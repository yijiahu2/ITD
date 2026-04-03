from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_finetune_dataset_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"未找到微调数据包: {bundle_path}")
    with open(bundle_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_training_sources(bundle: dict[str, Any]) -> dict[str, Any]:
    selection = bundle.get("selection_summary") or {}
    return {
        "target_module": bundle.get("target_module"),
        "target_model_role": bundle.get("target_model_role"),
        "failure_category": bundle.get("failure_category"),
        "supervision_mode": bundle.get("supervision_mode"),
        "training_ready_sample_count": int(selection.get("training_ready_sample_count") or 0),
        "weak_supervision_candidate_count": int(selection.get("weak_supervision_candidate_count") or 0),
        "label_preparation_queue_count": int(selection.get("label_preparation_queue_count") or 0),
        "replay_sample_count": int(selection.get("replay_sample_count") or 0),
        "public_dataset_candidate_count": int(selection.get("public_dataset_candidate_count") or 0),
    }
