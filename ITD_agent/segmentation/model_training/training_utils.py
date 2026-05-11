from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import to_bool


def ensure_public_segmentation_dataset(args_config: str, cfg: dict[str, Any]) -> tuple[Path, Path]:
    dataset_dir = Path(cfg["output_dir"]) / cfg.get("segmentation_dataset_dirname", "external_segmentation_dataset")
    summary_path = dataset_dir / "prepare_summary.json"
    force_rebuild = to_bool(cfg.get("segmentation_dataset_force_rebuild"), default=False)

    if dataset_dir.exists() and summary_path.exists() and not force_rebuild:
        return dataset_dir, summary_path

    if dataset_dir.exists() and force_rebuild:
        print(f"[INFO] rebuilding segmentation dataset in place: {dataset_dir}", flush=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "ITD_agent.segmentation.model_training.prepare_public_coco_segmentation_dataset",
        "--config",
        args_config,
    ]
    print("[RUN segmentation dataset prepare]", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ITD_agent.segmentation.model_training.prepare_public_coco_segmentation_dataset failed")
    if not summary_path.exists():
        raise RuntimeError(f"未找到 segmentation 数据准备摘要: {summary_path}")
    return dataset_dir, summary_path


def grad_accum_steps(cfg: dict[str, Any]) -> int:
    raw = cfg.get("segmentation_train_grad_accum_steps", 1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def find_best_checkpoint(search_root: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_root.rglob(pattern))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        priority = 0
        if "best" in name:
            priority += 100
        if name == "model_final.pth":
            priority += 100
        if name.startswith("model_"):
            priority += 20
        if "coco" in name and "segm" in name:
            priority += 20
        if "latest" in name:
            priority += 10
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return priority, mtime

    return sorted(candidates, key=score, reverse=True)[0]
