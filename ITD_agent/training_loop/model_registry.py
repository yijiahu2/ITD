from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import append_jsonl
from ITD_agent.training_loop.contracts import ModelVersionRecord, TrainingPlan, TrainingRunResult
from ITD_agent.training_loop.model_card_builder import build_model_card


def register_model_version(
    *,
    plan: TrainingPlan,
    result: TrainingRunResult,
    status: str,
    metrics_summary: dict[str, Any],
    replay_guard_summary: dict[str, Any],
    output_dir: str | Path,
    evidence: dict[str, Any],
) -> ModelVersionRecord:
    model_version_id = build_model_version_id(plan)
    checkpoint = result.best_checkpoint_path or ""
    base_record = {
        "model_version_id": model_version_id,
        "model_id": plan.target_model_id,
        "model_role": plan.target_model_role,
        "algorithm_name": plan.algorithm_name,
        "checkpoint_path": checkpoint,
        "source_training_job_id": plan.training_job_id,
        "status": status,
        "metrics_summary": metrics_summary,
        "replay_guard_summary": replay_guard_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    card_path = build_model_card(record=base_record, output_dir=output_dir, evidence=evidence)
    record = ModelVersionRecord(
        model_version_id=model_version_id,
        model_id=plan.target_model_id,
        model_role=plan.target_model_role,
        algorithm_name=plan.algorithm_name,
        checkpoint_path=checkpoint,
        source_training_job_id=plan.training_job_id,
        status=status,
        metrics_summary=metrics_summary,
        replay_guard_summary=replay_guard_summary,
        model_card_path=card_path,
    )
    append_jsonl(Path(output_dir) / "model_registry" / "model_versions.jsonl", {**base_record, "model_card_path": card_path})
    return record


def build_model_version_id(plan: TrainingPlan) -> str:
    compact_job = plan.training_job_id.replace(":", "").replace(".", "")
    return f"{plan.algorithm_name}_v3_{compact_job}"
