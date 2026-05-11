from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import write_json, write_jsonl


def load_review_assets(*, review_asset_dir: str | Path, target_cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    review_dir = Path(review_asset_dir)
    out_dir = Path(output_dir)
    samples = _load_jsonl(review_dir / "finetune_pool" / "samples.jsonl")
    distillation = _load_jsonl(review_dir / "distillation" / "distillation_candidates.jsonl")
    routing = _load_jsonl(review_dir / "routing" / "routing_candidates.jsonl")
    report = _load_json(review_dir / "reports" / "review_summary.json")
    asset_summary = _load_json(review_dir / "reports" / "asset_summary.json")

    target_role = str(target_cfg.get("target_model_role") or "expert_model")
    target_family = target_cfg.get("target_expert_family")
    failure_category = target_cfg.get("failure_category")
    imported_root = out_dir / "finetune_bundle" / "imported_finetune_pool"
    records_dir = imported_root / "records"
    imported_samples = [
        _convert_review_sample_to_finetune_pool_sample(
            sample,
            target_model_role=target_role,
            target_expert_family=str(target_family) if target_family else None,
        )
        for sample in samples
        if _matches_target(sample, target_role=target_role, failure_category=str(failure_category) if failure_category else None)
    ]
    write_jsonl(records_dir / "samples.jsonl", imported_samples)
    write_jsonl(records_dir / "public_dataset_candidates.jsonl", [])
    snapshot = {
        "timestamp": report.get("created_at") or report.get("review_run_id"),
        "run_name": report.get("source_run_id") or "unknown_run",
        "trigger_ready": bool(imported_samples),
        "recommended_target_module": "segmentation_model",
        "recommended_target_model_role": target_role,
        "recommended_target_expert_family": target_family,
        "recommended_failure_category": failure_category,
        "trigger_reason": "loaded_from_review_assets",
        "sample_counts": dict(Counter(str(item.get("failure_category") or "unknown") for item in imported_samples)),
        "ready_counts": {
            "training_ready": sum(1 for item in imported_samples if item.get("ready_for_training")),
            "replay": sum(1 for item in imported_samples if item.get("source_type") == "replay_good_sample"),
        },
        "metadata": {"review_asset_dir": str(review_dir), "asset_summary": asset_summary},
    }
    write_json(records_dir / "latest_trigger_snapshot.json", snapshot)
    write_json(
        out_dir / "trigger" / "review_asset_load_report.json",
        {
            "review_asset_dir": str(review_dir),
            "source_sample_count": len(samples),
            "imported_sample_count": len(imported_samples),
            "distillation_candidate_count": len(distillation),
            "routing_candidate_count": len(routing),
            "imported_finetune_pool_root": str(imported_root),
        },
    )
    return {
        "review_asset_dir": str(review_dir),
        "review_summary": report,
        "asset_summary": asset_summary,
        "finetune_samples": samples,
        "imported_finetune_pool_root": str(imported_root),
        "imported_finetune_samples": imported_samples,
        "distillation_candidates": distillation,
        "routing_candidates": routing,
        "snapshot": snapshot,
    }


def _matches_target(sample: dict[str, Any], *, target_role: str, failure_category: str | None) -> bool:
    if failure_category and str(sample.get("target_error_type") or sample.get("failure_category") or "") != failure_category:
        return False
    return str(sample.get("review_status") or "").lower() in {"approved", "approve", "accepted"}


def _convert_review_sample_to_finetune_pool_sample(
    sample: dict[str, Any],
    *,
    target_model_role: str,
    target_expert_family: str | None,
) -> dict[str, Any]:
    failure_category = sample.get("target_error_type") or sample.get("failure_category")
    artifact_refs = {
        "image": sample.get("image_crop_path"),
        "gt_mask": sample.get("gt_mask_path"),
        "main_pred_mask": sample.get("main_pred_path"),
        "expert_pred_mask": sample.get("expert_pred_path"),
        "metadata": sample.get("metadata_path"),
    }
    label_status = "manual" if sample.get("gt_mask_path") else "pseudo"
    return {
        "sample_id": sample.get("sample_id"),
        "run_name": sample.get("source_run_id"),
        "timestamp": sample.get("created_at"),
        "source_type": sample.get("sample_type") or "review_approved_sample",
        "target_module": "segmentation_model",
        "target_model_role": target_model_role,
        "target_expert_family": target_expert_family,
        "failure_category": failure_category,
        "scene_profile": {},
        "artifact_refs": artifact_refs,
        "label_status": label_status,
        "ready_for_training": str(sample.get("review_status") or "").lower() == "approved",
        "tags": [str(item) for item in [failure_category, sample.get("sample_type")] if item],
        "metrics_snapshot": {"quality_score": sample.get("quality_score")},
        "metadata": {
            "source": "finetune_pool_review",
            "source_run_id": sample.get("source_run_id"),
            "source_trajectory_id": sample.get("source_trajectory_id"),
            "source_roi_id": sample.get("source_roi_id"),
            "image_id": sample.get("image_id"),
            "roi": sample.get("roi") or {},
            "review_status": sample.get("review_status"),
            "export_status": sample.get("export_status"),
            "raw_review_sample": sample,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]
