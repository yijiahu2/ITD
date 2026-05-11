from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def synthesize_training_samples(
    *,
    accepted_samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if not bool(cfg.get("enabled", False)):
        return {"synthetic_samples": [], "report": {"enabled": False}}

    max_ratio = float(cfg.get("max_synthetic_ratio", 0.3))
    max_count = int(len(accepted_samples) * max_ratio)

    synthetic_samples = []
    for sample in accepted_samples[:max_count]:
        synthetic = _build_synthetic_sample_stub(sample, cfg)
        synthetic_samples.append(synthetic)

    root = Path(output_dir) / "dataset_bundle" / "synthesis"
    root.mkdir(parents=True, exist_ok=True)

    samples_path = root / "synthetic_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for item in synthetic_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "enabled": True,
        "source_sample_count": len(accepted_samples),
        "synthetic_sample_count": len(synthetic_samples),
        "max_synthetic_ratio": max_ratio,
        "policy": "train_only_no_val_test_replay",
    }
    (root / "synthesis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"synthetic_samples": synthetic_samples, "report": report}


def _build_synthetic_sample_stub(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(sample.get("metadata") or {})
    failure_category = sample.get("failure_category") or metadata.get("failure_category")
    strategies = _strategies_for_failure(str(failure_category))
    return {
        **sample,
        "sample_id": f"synthetic_{sample.get('sample_id')}",
        "source_sample_id": sample.get("sample_id"),
        "source_type": "synthetic_augmentation",
        "label_status": "synthetic",
        "split_policy": "train_only",
        "augmentation_strategies": strategies,
        "ready_for_training": True,
    }


def _strategies_for_failure(failure_category: str) -> list[str]:
    mapping = {
        "false_negative": ["small_crown_copy_paste", "scale_jitter", "brightness_shadow_perturbation"],
        "under_segmentation": ["dense_canopy_crop", "boundary_jitter", "local_contrast_jitter"],
        "over_segmentation": ["large_crown_context_crop", "boundary_smoothing"],
        "false_positive": ["hard_negative_crop", "shadow_background_sampling"],
    }
    return mapping.get(failure_category, ["safe_color_jitter"])
