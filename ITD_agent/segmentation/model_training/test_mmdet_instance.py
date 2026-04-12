from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, ensure_dir, load_json, load_yaml
from ITD_agent.segmentation.model_training.train_mmdet_instance import _resolve_training_env

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_generated_config(training_dir: Path, train_summary: dict[str, Any]) -> Path:
    generated = train_summary.get("generated_infer_config") or train_summary.get("generated_config")
    if not generated:
        raise RuntimeError("train_summary 中缺少 generated_config / generated_infer_config")
    generated_path = Path(str(generated)).resolve()
    if not generated_path.exists():
        raise FileNotFoundError(f"generated_config 不存在: {generated_path}")
    return generated_path


def _resolve_checkpoint(train_summary: dict[str, Any], cli_checkpoint: str | None) -> Path:
    value = cli_checkpoint or train_summary.get("best_ckpt")
    if not value:
        raise RuntimeError("未提供 checkpoint，且 train_summary 中不存在 best_ckpt")
    ckpt = Path(str(value)).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    return ckpt


def _fallback_generated_config(training_dir: Path, algorithm_name: str) -> Path:
    generated = training_dir / "generated_configs" / f"{algorithm_name}_public_itd.py"
    if not generated.exists():
        raise FileNotFoundError(f"generated_config 不存在，且 train_summary.json 缺失: {generated}")
    return generated.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--show-dir")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_cfg = _resolve_training_env(cfg)
    output_dir = Path(cfg["output_dir"])
    training_dir = output_dir / cfg.get("segmentation_train_out_dirname", "segmentation_training")
    eval_dir = ensure_dir(training_dir / "evaluation")
    test_work_dir = ensure_dir(eval_dir / "work_dir")

    train_summary_path = training_dir / "train_summary.json"
    train_summary: dict[str, Any] = {}
    if train_summary_path.exists():
        train_summary = load_json(train_summary_path)

    if train_summary:
        generated_config = _resolve_generated_config(training_dir, train_summary)
    else:
        generated_config = _fallback_generated_config(training_dir, env_cfg["algorithm_name"])
    checkpoint = _resolve_checkpoint(train_summary, args.checkpoint)
    repo_root = Path(str(train_summary.get("repo_root") or env_cfg["repo_root"])).resolve()
    conda_sh = str(train_summary.get("conda_sh") or env_cfg["conda_sh"])
    conda_env = str(train_summary.get("conda_env") or env_cfg["conda_env"])
    test_py = repo_root / "tools" / "test.py"
    if not test_py.exists():
        raise FileNotFoundError(f"未找到 MMDetection test.py: {test_py}")

    pred_pkl = eval_dir / "predictions.pkl"
    cmd_parts = [
        "python",
        str(test_py),
        str(generated_config),
        str(checkpoint),
        "--work-dir",
        str(test_work_dir),
        "--out",
        str(pred_pkl),
    ]
    if args.show_dir:
        cmd_parts.extend(["--show-dir", str(Path(args.show_dir).resolve())])

    bash_cmd = (
        f"source {conda_sh} && "
        f"conda activate {conda_env} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export PYTHONPATH={repo_root}:{PROJECT_ROOT}:${{PYTHONPATH:-}} && "
        f"{' '.join(shlex.quote(str(x)) for x in cmd_parts)}"
    )
    print("[RUN segmentation mmdet test]")
    print(bash_cmd)
    result = subprocess.run(["bash", "-lc", bash_cmd], cwd=str(repo_root))

    summary = {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "segmentation_algorithm": env_cfg["algorithm_name"],
        "generated_config": str(generated_config),
        "checkpoint": str(checkpoint),
        "repo_root": str(repo_root),
        "conda_sh": conda_sh,
        "conda_env": conda_env,
        "test_work_dir": str(test_work_dir),
        "predictions_pkl": str(pred_pkl),
    }
    if args.show_dir:
        summary["show_dir"] = str(Path(args.show_dir).resolve())

    summary_path = eval_dir / "test_summary.json"
    dump_json(summary, summary_path)
    if not train_summary_path.exists():
        recovered_summary = {
            "status": "recovered_after_train",
            "segmentation_algorithm": env_cfg["algorithm_name"],
            "repo_root": str(repo_root),
            "conda_sh": conda_sh,
            "conda_env": conda_env,
            "generated_config": str(generated_config),
            "generated_infer_config": str(generated_config),
            "best_ckpt": str(checkpoint),
            "test_summary_json": str(summary_path),
            "eval_after_train": True,
            "eval_returncode": int(result.returncode),
        }
        dump_json(recovered_summary, train_summary_path)
    if result.returncode != 0:
        raise RuntimeError(f"segmentation mmdet test failed: {summary_path}")

    print(f"[OK] segmentation mmdet test done: {summary_path}")


if __name__ == "__main__":
    main()
