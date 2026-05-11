from __future__ import annotations

from typing import Any


def build_learning_events_from_run_result(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    result = run_result.get("result") or run_result
    return [
        {
            "event_type": "run_completed",
            "run_id": result.get("run_id"),
            "output_dir": result.get("output_dir"),
            "score_before": result.get("score_before"),
            "score_after": result.get("score_after"),
            "scene_signature": result.get("scene_signature"),
            "parameter_signature": result.get("parameter_signature"),
            "repeat_count": result.get("repeat_count", 1),
            "residual_type": result.get("residual_type"),
        }
    ]


def build_learning_events_from_review_result(review_result: dict[str, Any]) -> list[dict[str, Any]]:
    result = review_result.get("result") or review_result
    asset_counts = result.get("asset_counts") or {}
    return [
        {
            "event_type": "review_completed",
            "output_dir": result.get("output_dir"),
            "skill_record_count": int(asset_counts.get("skill_records") or 0),
            "finetune_sample_count": int(asset_counts.get("finetune_samples") or 0),
        }
    ]


def build_learning_events_from_training_result(training_result: dict[str, Any]) -> list[dict[str, Any]]:
    result = training_result.get("result") or training_result
    return [
        {
            "event_type": "training_completed",
            "output_dir": result.get("output_dir"),
            "training_status": (result.get("training_result") or {}).get("status"),
            "promotion_decision": (result.get("promotion_decision") or {}).get("decision"),
        }
    ]
