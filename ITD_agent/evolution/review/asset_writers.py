from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import append_jsonl, write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_review_run_record(
    *,
    conn: sqlite3.Connection,
    review_run_id: str,
    source_run_id: str,
    config_path: str,
    output_dir: str,
    status: str,
    summary: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO v2_review_runs
        (review_run_id, source_run_id, created_at, status, config_path, output_dir, summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (review_run_id, source_run_id, _now(), status, config_path, output_dir, _dump(summary or {})),
    )


def write_review_event(
    *,
    conn: sqlite3.Connection,
    review_run_id: str,
    source_run_id: str,
    decision: dict[str, Any],
    review_type: str,
    guardrail_result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    event_id = f"event_{review_run_id}_{review_type}_{decision.get('candidate_id')}_{abs(hash(_dump(decision))) % 100000000}"
    conn.execute(
        """
        INSERT OR REPLACE INTO v2_review_events
        (review_event_id, review_run_id, source_run_id, source_trajectory_id, candidate_id, candidate_type,
         review_type, decision, reason, evidence_refs_json, guardrail_result_json, error_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            review_run_id,
            source_run_id,
            decision.get("trajectory_id"),
            decision.get("candidate_id"),
            decision.get("candidate_type"),
            review_type,
            decision.get("decision"),
            decision.get("reason"),
            _dump(decision.get("evidence_refs") or {}),
            _dump(guardrail_result or {}),
            _dump(error or {}),
            _now(),
        ),
    )


def write_memory_record(*, conn: sqlite3.Connection, output_dir: Path, decision: dict[str, Any]) -> dict[str, Any]:
    payload = dict(decision.get("payload") or {})
    memory_id = f"memory_{decision['candidate_id']}"
    record = {
        "memory_id": memory_id,
        "created_at": _now(),
        **payload,
    }
    append_jsonl(output_dir / "memory" / "memory_records.jsonl", record)
    memory_type = str(record.get("memory_type") or "unknown")
    if memory_type == "failure_pattern_memory":
        append_jsonl(output_dir / "memory" / "failure_pattern_records.jsonl", record)
    elif memory_type == "expert_success_memory":
        append_jsonl(output_dir / "memory" / "expert_success_records.jsonl", record)
    elif memory_type == "rollback_memory":
        append_jsonl(output_dir / "memory" / "rollback_records.jsonl", record)
    conn.execute(
        """
        INSERT OR REPLACE INTO memory_records
        (memory_id, source_run_id, source_trajectory_id, memory_type, level1_error_type, failure_family,
         summary, evidence_refs_json, metrics_snapshot_json, artifact_refs_json, confidence, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            record.get("source_run_id"),
            record.get("source_trajectory_id"),
            record.get("memory_type"),
            record.get("level1_error_type"),
            record.get("failure_family"),
            record.get("summary"),
            _dump(decision.get("evidence_refs") or {}),
            _dump(record.get("metrics_snapshot") or {}),
            _dump(record.get("artifact_refs") or {}),
            record.get("confidence"),
            record.get("status") or "active",
            record.get("created_at"),
        ),
    )
    return record


def write_skill_record(*, conn: sqlite3.Connection, output_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    append_jsonl(output_dir / "skills" / "skill_records.jsonl", record)
    draft_root = output_dir / "skills" / "drafts" / str((record.get("trigger_conditions") or {}).get("failure_family") or record["skill_id"])
    write_json(
        draft_root / "SKILL.md",
        {
            "name": record["name"],
            "status": record["status"],
            "skill_type": record["skill_type"],
            "evidence_summary": record["evidence_summary"],
            "safety_constraints": record["safety_constraints"],
            "recommended_action": record["recommended_action"],
        },
    )
    append_jsonl(draft_root / "references" / "evidence_cases.jsonl", record)
    (draft_root / "templates").mkdir(parents=True, exist_ok=True)
    (draft_root / "scripts").mkdir(parents=True, exist_ok=True)
    conn.execute(
        """
        INSERT OR REPLACE INTO skill_records
        (skill_id, skill_type, name, source_run_ids_json, source_trajectory_ids_json,
         trigger_conditions_json, recommended_action_json, evidence_summary_json,
         safety_constraints_json, status, version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["skill_id"],
            record["skill_type"],
            record["name"],
            _dump(record.get("source_run_ids") or []),
            _dump(record.get("source_trajectory_ids") or []),
            _dump(record.get("trigger_conditions") or {}),
            _dump(record.get("recommended_action") or {}),
            _dump(record.get("evidence_summary") or {}),
            _dump(record.get("safety_constraints") or {}),
            record["status"],
            record["version"],
            record["created_at"],
        ),
    )
    return record


def write_finetune_sample(*, conn: sqlite3.Connection, output_dir: Path, decision: dict[str, Any], source_image_path: Path | None = None) -> dict[str, Any]:
    payload = dict(decision.get("payload") or {})
    sample_id = f"sample_{decision['candidate_id']}"
    sample_dir = output_dir / "finetune_pool" / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_suffix = source_image_path.suffix if source_image_path and source_image_path.suffix else ".png"
    image_path = sample_dir / f"image{image_suffix}"
    if image_path.exists():
        pass
    elif source_image_path and source_image_path.exists():
        try:
            os.link(source_image_path, image_path)
        except OSError:
            shutil.copyfile(source_image_path, image_path)
    else:
        image_path.write_text("source image unavailable; see metadata artifact_refs", encoding="utf-8")
    gt_mask_path = sample_dir / "gt_mask.json"
    main_pred_path = sample_dir / "main_pred_mask.json"
    expert_pred_path = sample_dir / "expert_pred_mask.json"
    write_json(gt_mask_path, {"roi": payload.get("roi"), "source": "coco_gt_reference"})
    write_json(main_pred_path, {"roi": payload.get("roi"), "source": "main_prediction_reference"})
    write_json(expert_pred_path, {"roi": payload.get("roi"), "source": "expert_prediction_reference"})
    metadata = {
        "sample_id": sample_id,
        "created_at": _now(),
        **payload,
        "image_crop_path": str(image_path),
        "gt_mask_path": str(gt_mask_path),
        "main_pred_path": str(main_pred_path),
        "expert_pred_path": str(expert_pred_path),
    }
    metadata_path = sample_dir / "metadata.json"
    write_json(metadata_path, metadata)
    record = {
        **metadata,
        "metadata_path": str(metadata_path),
        "export_status": "exported",
    }
    append_jsonl(output_dir / "finetune_pool" / "samples.jsonl", record)
    conn.execute(
        """
        INSERT OR REPLACE INTO finetune_samples
        (sample_id, source_run_id, source_trajectory_id, source_roi_id, image_id, sample_type,
         target_model_role, target_error_type, image_crop_path, gt_mask_path, main_pred_path,
         expert_pred_path, metadata_path, quality_score, review_status, export_status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sample_id,
            record.get("source_run_id"),
            record.get("source_trajectory_id"),
            record.get("source_roi_id"),
            record.get("image_id"),
            record.get("sample_type"),
            record.get("target_model_role"),
            record.get("target_error_type"),
            record.get("image_crop_path"),
            record.get("gt_mask_path"),
            record.get("main_pred_path"),
            record.get("expert_pred_path"),
            record.get("metadata_path"),
            record.get("quality_score"),
            record.get("review_status"),
            record.get("export_status"),
            record.get("created_at"),
        ),
    )
    return record


def write_routing_candidate(*, conn: sqlite3.Connection, output_dir: Path, decision: dict[str, Any]) -> dict[str, Any]:
    payload = dict(decision.get("payload") or {})
    candidate_id = f"routing_{decision['candidate_id']}"
    record = {"routing_candidate_id": candidate_id, "created_at": _now(), **payload}
    append_jsonl(output_dir / "routing" / "routing_candidates.jsonl", record)
    conn.execute(
        """
        INSERT OR REPLACE INTO routing_candidates
        (routing_candidate_id, source_run_id, source_trajectory_id, level1_error_type, failure_family,
         expert_model, expert_decision, improvement_summary_json, safety_summary_json,
         recommendation, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            record.get("source_run_id"),
            record.get("source_trajectory_id"),
            record.get("level1_error_type"),
            record.get("failure_family"),
            record.get("expert_model"),
            record.get("expert_decision"),
            _dump(record.get("improvement_summary") or {}),
            _dump(record.get("safety_summary") or {}),
            record.get("recommendation"),
            record.get("status"),
            record.get("created_at"),
        ),
    )
    return record


def write_distillation_candidate(*, conn: sqlite3.Connection, output_dir: Path, decision: dict[str, Any]) -> dict[str, Any]:
    payload = dict(decision.get("payload") or {})
    candidate_id = f"distillation_{decision['candidate_id']}"
    record = {"distillation_candidate_id": candidate_id, "created_at": _now(), **payload}
    append_jsonl(output_dir / "distillation" / "distillation_candidates.jsonl", record)
    conn.execute(
        """
        INSERT OR REPLACE INTO distillation_candidates
        (distillation_candidate_id, source_run_id, source_trajectory_id, source_roi_id,
         expert_model, quality_tier, evidence_refs_json, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            record.get("source_run_id"),
            record.get("source_trajectory_id"),
            record.get("source_roi_id"),
            record.get("expert_model"),
            record.get("quality_tier"),
            _dump(record.get("evidence_refs") or {}),
            record.get("status"),
            record.get("created_at"),
        ),
    )
    return record
