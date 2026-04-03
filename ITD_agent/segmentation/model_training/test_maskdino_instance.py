from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, ensure_dir, load_json, load_yaml


def _resolve_checkpoint(train_summary: dict[str, Any], cli_checkpoint: str | None) -> Path:
    value = cli_checkpoint or train_summary.get("best_ckpt")
    if not value:
        raise RuntimeError("未提供 checkpoint，且 train_summary 中不存在 best_ckpt")
    ckpt = Path(str(value)).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    return ckpt


def _resolve_train_summary(training_dir: Path) -> tuple[Path, dict[str, Any]]:
    train_summary_path = training_dir / "train_summary.json"
    if not train_summary_path.exists():
        raise FileNotFoundError(f"未找到 train_summary.json: {train_summary_path}")
    return train_summary_path, load_json(train_summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = Path(cfg["output_dir"])
    training_dir = output_dir / cfg.get("segmentation_train_out_dirname", "segmentation_training")
    train_summary_path, train_summary = _resolve_train_summary(training_dir)

    generated_config = Path(str(train_summary["generated_config"])).resolve()
    if not generated_config.exists():
        raise FileNotFoundError(f"generated_config 不存在: {generated_config}")

    checkpoint = _resolve_checkpoint(train_summary, args.checkpoint)
    repo_root = Path(str(train_summary["repo_root"])).resolve()
    conda_sh = str(train_summary["conda_sh"])
    conda_env = str(train_summary["conda_env"])
    num_gpus = int(train_summary.get("num_gpus", 1))

    test_json = Path(str(train_summary["test_json"])).resolve()
    if not test_json.exists():
        raise FileNotFoundError(f"test_json 不存在: {test_json}")

    eval_dir = ensure_dir(training_dir / "evaluation")
    test_work_dir = ensure_dir(eval_dir / "work_dir")
    image_root = Path(str(train_summary["image_root"])).resolve()

    thing_classes = train_summary.get("class_names") or ["crown"]
    if isinstance(thing_classes, str):
        thing_classes = [x.strip() for x in thing_classes.split(",") if x.strip()]

    entry_module = "ITD_agent.segmentation.model_training.maskdino_train_entry"
    cmd_parts = [
        "python",
        "-m",
        entry_module,
        "--eval-only",
        "--num-gpus",
        str(num_gpus),
        "--config-file",
        str(generated_config),
        "--train-json",
        str(Path(str(train_summary["train_json"])).resolve()),
        "--val-json",
        str(Path(str(train_summary["val_json"])).resolve()),
        "--test-json",
        str(test_json),
        "--image-root",
        str(image_root),
        "--train-dataset-name",
        str(train_summary["train_dataset_name"]),
        "--val-dataset-name",
        str(train_summary["val_dataset_name"]),
        "--test-dataset-name",
        str(train_summary["test_dataset_name"]),
        "--thing-classes",
        ",".join(str(x) for x in thing_classes),
        "MODEL.WEIGHTS",
        str(checkpoint),
        "DATASETS.TEST",
        f"(\"{train_summary['test_dataset_name']}\",)",
        "OUTPUT_DIR",
        str(test_work_dir),
    ]

    bash_cmd = (
        f"source {conda_sh} && "
        f"conda activate {conda_env} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export MASKDINO_REPO_ROOT={repo_root} && "
        f"export PYTHONPATH={Path(__file__).resolve().parents[1]}:{repo_root}:${{PYTHONPATH:-}} && "
        f"{' '.join(shlex.quote(str(x)) for x in cmd_parts)}"
    )
    print("[RUN segmentation MaskDINO test]")
    print(bash_cmd)
    result = subprocess.run(["bash", "-lc", bash_cmd], cwd=str(repo_root))

    summary = {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "segmentation_algorithm": train_summary.get("segmentation_algorithm", "maskdino_official"),
        "generated_config": str(generated_config),
        "checkpoint": str(checkpoint),
        "repo_root": str(repo_root),
        "conda_sh": conda_sh,
        "conda_env": conda_env,
        "test_work_dir": str(test_work_dir),
        "test_json": str(test_json),
        "test_dataset_name": str(train_summary["test_dataset_name"]),
    }
    summary_path = eval_dir / "test_summary.json"
    dump_json(summary, summary_path)
    if result.returncode != 0:
        dump_json(train_summary, train_summary_path)
        raise RuntimeError(f"segmentation MaskDINO test failed: {summary_path}")

    print(f"[OK] segmentation MaskDINO test done: {summary_path}")


if __name__ == "__main__":
    main()
