from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ITD_agent.segmentation.finetuning.io_utils import dump_json, load_yaml
from ITD_agent.segmentation.model_registry.mmdet_specs import is_mmdet_algorithm


SUMMARY_FILENAME = "public_segmentation_model_finetune_summary.json"
LEGACY_SUMMARY_FILENAME = "public_segmentation_finetune_pipeline_summary.json"


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def load_json_file(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def resolve_segmentation_trainer_module(cfg: dict) -> str:
    algorithm = str(cfg.get("segmentation_algorithm") or cfg.get("segmentation_algorithm", "")).strip().lower()
    if is_mmdet_algorithm(algorithm):
        return "ITD_agent.segmentation.model_training.train_mmdet_instance"
    if algorithm == "maskdino_official":
        return "ITD_agent.segmentation.model_training.train_maskdino_instance"
    raise ValueError(f"Unsupported segmentation_algorithm for public finetune pipeline: {algorithm}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"config": args.config, "mode": "public_segmentation_model_finetune", "steps": []}

    trainer_module = resolve_segmentation_trainer_module(cfg)
    run_cmd([sys.executable, "-u", "-m", trainer_module, "--config", args.config])

    train_summary_path = out_dir / cfg.get("segmentation_train_out_dirname", "segmentation_training") / "train_summary.json"
    if not train_summary_path.exists():
        raise RuntimeError(f"训练完成后未找到 train_summary.json: {train_summary_path}")

    train_summary = load_json_file(train_summary_path)
    ckpt = train_summary.get("best_ckpt")
    if not ckpt:
        raise RuntimeError(f"train_summary.json 中未找到 best_ckpt: {train_summary_path}")
    if not Path(ckpt).exists():
        raise RuntimeError(f"best_ckpt 文件不存在: {ckpt}")

    summary["train_summary_json"] = str(train_summary_path)
    summary["best_ckpt"] = ckpt
    summary["steps"].append(
        {
            "step": "train_segmentation_model",
            "step_alias": "train_segmentation_model",
            "algorithm": train_summary.get("segmentation_algorithm") or train_summary.get("segmentation_algorithm"),
            "ckpt": ckpt,
            "generated_config": train_summary.get("generated_config"),
            "train_summary_json": str(train_summary_path),
        }
    )
    test_summary_json = train_summary.get("test_summary_json")
    if test_summary_json:
        summary["test_summary_json"] = str(test_summary_json)
        summary["steps"].append(
            {
                "step": "test_segmentation_model",
                "step_alias": "test_segmentation_model",
                "test_summary_json": str(test_summary_json),
                "eval_returncode": train_summary.get("eval_returncode"),
            }
        )

    run_cmd(
        [
            sys.executable,
            "-u",
            "-m",
            "ITD_agent.segmentation.model_training.infer_segmentation_finetuned",
            "--config",
            args.config,
            "--ckpt",
            ckpt,
        ]
    )

    integration_summary_path = out_dir / "finetuned_infer" / "integration_summary.json"
    integration_summary = None
    if integration_summary_path.exists():
        integration_summary = load_json_file(integration_summary_path)
        summary["steps"].append(
            {
                "step": "integration_check",
                "integration_summary_json": str(integration_summary_path),
                "can_rerun": integration_summary.get("can_rerun"),
                "reason": integration_summary.get("reason"),
            }
        )

    ft_cfg = out_dir / "finetuned_infer" / "exp_segmentation_finetuned.yaml"
    enable_rerun = to_bool(cfg.get("enable_rerun_after_finetune", False))
    summary["enable_rerun_after_finetune"] = enable_rerun

    if ft_cfg.exists():
        summary["steps"].append({"step": "make_finetuned_config", "config": str(ft_cfg)})
    else:
        summary["steps"].append(
            {
                "step": "make_finetuned_config",
                "status": "skipped",
                "reason": (
                    integration_summary.get("reason")
                    if integration_summary is not None
                    else f"missing generated config: {ft_cfg}"
                ),
            }
        )

    if enable_rerun:
        if not ft_cfg.exists():
            raise RuntimeError(f"未生成 exp_segmentation_finetuned.yaml: {ft_cfg}")
        if integration_summary is not None and not integration_summary.get("can_rerun", False):
            raise RuntimeError(
                "enable_rerun_after_finetune=true，但 integration_summary 显示不能 rerun: "
                f"{integration_summary.get('reason', 'unknown reason')}"
            )
        run_cmd([sys.executable, "-u", "-m", "scripts.run_ITD_agent_experiment", "--config", str(ft_cfg)])
        summary["steps"].append({"step": "rerun_experiment", "config": str(ft_cfg)})

        base_cfg_path = cfg.get("base_config")
        before_csv = None
        if base_cfg_path:
            base_cfg = load_yaml(base_cfg_path)
            if base_cfg.get("details_csv"):
                before_csv = str(base_cfg["details_csv"])

        after_cfg = load_yaml(str(ft_cfg))
        after_csv = str(after_cfg.get("details_csv") or Path(after_cfg["output_dir"]) / "evaluation_details.csv")
        if not Path(after_csv).exists():
            raise RuntimeError(f"rerun 完成后未找到 after_csv: {after_csv}")
        summary["steps"].append({"step": "rerun_output_check", "after_csv": str(after_csv)})

        if before_csv and Path(before_csv).exists():
            compare_dir = out_dir / "compare"
            compare_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "ITD_agent.segmentation.finetuning.evaluate_finetune_gain",
                    "--before_csv",
                    str(before_csv),
                    "--after_csv",
                    str(after_csv),
                    "--out_dir",
                    str(compare_dir),
                    "--config",
                    args.config,
                ]
            )
            summary["compare_json"] = str(compare_dir / "finetune_gain_summary.json")
            summary["steps"].append(
                {
                    "step": "compare",
                    "compare_dir": str(compare_dir),
                    "compare_json": str(compare_dir / "finetune_gain_summary.json"),
                    "before_csv": str(before_csv),
                    "after_csv": str(after_csv),
                }
            )
        else:
            summary["steps"].append(
                {
                    "step": "compare",
                    "status": "skipped",
                    "reason": "base_config.details_csv missing or not found",
                }
            )
    else:
        print("[SKIP] rerun after finetune is disabled by config", flush=True)
        summary["steps"].append(
            {
                "step": "rerun_experiment",
                "status": "skipped",
                "reason": "enable_rerun_after_finetune=false",
            }
        )

    dump_json(summary, out_dir / SUMMARY_FILENAME)
    dump_json(summary, out_dir / LEGACY_SUMMARY_FILENAME)
    print(f"[OK] public segmentation-model finetune pipeline done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
