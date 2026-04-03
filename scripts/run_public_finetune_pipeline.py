from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ITD_agent.segmentation.finetuning.io_utils import dump_json, load_yaml


SUMMARY_FILENAME = "public_data_processing_finetune_summary.json"
LEGACY_SUMMARY_FILENAME = "public_finetune_pipeline_summary.json"


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"config": args.config, "mode": "public_data_processing_finetune", "steps": []}

    run_cmd([sys.executable, "-m", "ITD_agent.segmentation.finetuning.train_data_processing_light", "--config", args.config])

    train_summary_path = out_dir / "training" / "train_summary.json"
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
            "step": "train_data_processing_light",
            "backend": train_summary.get("backend"),
            "ckpt": ckpt,
            "train_summary_json": str(train_summary_path),
        }
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "ITD_agent.segmentation.finetuning.infer_data_processing_finetuned",
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

    ft_cfg = out_dir / "finetuned_infer" / "exp_finetuned.yaml"
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
            raise RuntimeError(f"未生成 exp_finetuned.yaml: {ft_cfg}")
        if integration_summary is not None and not integration_summary.get("can_rerun", False):
            raise RuntimeError(
                "enable_rerun_after_finetune=true，但 integration_summary 显示不能 rerun: "
                f"{integration_summary.get('reason', 'unknown reason')}"
            )
        run_cmd([sys.executable, "-m", "scripts.run_ITD_agent_experiment", "--config", str(ft_cfg)])
        summary["steps"].append({"step": "rerun_experiment", "config": str(ft_cfg)})
    else:
        print("[SKIP] rerun after finetune is disabled by config")
        summary["steps"].append(
            {
                "step": "rerun_experiment",
                "status": "skipped",
                "reason": "enable_rerun_after_finetune=false",
            }
        )

    dump_json(summary, out_dir / SUMMARY_FILENAME)
    dump_json(summary, out_dir / LEGACY_SUMMARY_FILENAME)
    print(f"[OK] public data-processing finetune pipeline done: {out_dir}")


if __name__ == "__main__":
    main()
