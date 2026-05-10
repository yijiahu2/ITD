from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact_store import sha256_file
from .db import connect_state_db


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_run_record(
    *,
    db_path: str | Path,
    run_id: str,
    mode: str,
    mainline_profile: str,
    config_path: str,
    output_dir: str,
    status: str,
    summary: dict[str, Any],
) -> None:
    with connect_state_db(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, created_at, mode, mainline_profile, config_path, output_dir, status, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                mode,
                mainline_profile,
                config_path,
                output_dir,
                status,
                _dump(summary),
            ),
        )


def write_state_records(*, db_path: str | Path, trajectory: dict[str, Any], trajectory_path: str) -> None:
    with connect_state_db(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO trajectories
            (trajectory_id, run_id, created_at, main_model, final_status, final_result_source, trajectory_path, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trajectory["trajectory_id"],
                trajectory["run_id"],
                trajectory.get("created_at"),
                (trajectory.get("main_model_stage") or {}).get("model_id"),
                "completed",
                (trajectory.get("fusion_stage") or {}).get("final_result_source"),
                trajectory_path,
                trajectory.get("review_status", "pending"),
            ),
        )
        for roi in (trajectory.get("roi_stage") or {}).get("roi_candidates") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO roi_candidates
                (roi_id, trajectory_id, image_id, level1_error_type, failure_family, severity_score,
                 confidence_level, review_status, expert_eligible, training_eligible, distill_eligible,
                 bbox_json, tags_json, geometry_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    roi["roi_id"],
                    trajectory["trajectory_id"],
                    roi.get("image_id"),
                    roi.get("level1_error_type"),
                    roi.get("failure_family"),
                    roi.get("severity_score"),
                    roi.get("confidence_level"),
                    roi.get("review_status"),
                    int(bool(roi.get("expert_eligible"))),
                    int(bool(roi.get("training_eligible"))),
                    int(bool(roi.get("distill_eligible"))),
                    _dump(roi.get("bbox_px") or []),
                    _dump(roi.get("tags") or []),
                    _dump(roi.get("geometry") or {}),
                ),
            )
        for task in (trajectory.get("expert_task_stage") or {}).get("expert_tasks") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO expert_tasks
                (expert_task_id, trajectory_id, expert_model, failure_family, level1_error_type,
                 roi_ids_json, tile_window_json, status, trigger_reason_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["expert_task_id"],
                    trajectory["trajectory_id"],
                    task.get("expert_model"),
                    task.get("failure_family"),
                    task.get("level1_error_type"),
                    _dump(task.get("roi_ids") or []),
                    _dump(task.get("tile_window_px") or []),
                    task.get("status", "pending"),
                    _dump(task.get("trigger_reason") or {}),
                ),
            )
        for review in (trajectory.get("expert_review_stage") or {}).get("expert_reviews") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO expert_reviews
                (review_id, expert_task_id, decision, improvement_json, safety_json,
                 accepted_roi_ids_json, rejected_roi_ids_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review["review_id"],
                    review["expert_task_id"],
                    review["decision"],
                    _dump(review.get("improvement") or {}),
                    _dump(review.get("safety") or {}),
                    _dump(review.get("accepted_roi_ids") or []),
                    _dump(review.get("rejected_roi_ids") or []),
                ),
            )
        for idx, event in enumerate((trajectory.get("fusion_stage") or {}).get("fusion_events") or [], start=1):
            conn.execute(
                """
                INSERT OR REPLACE INTO fusion_events
                (fusion_event_id, trajectory_id, decision, fused_result_path, summary_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"fusion_{trajectory['trajectory_id']}_{idx:04d}",
                    trajectory["trajectory_id"],
                    event.get("decision"),
                    event.get("fused_result_path"),
                    _dump(event),
                ),
            )
        for candidate in (trajectory.get("pending_review_candidates") or {}).get("training_candidates") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_candidates
                (candidate_id, trajectory_id, roi_id, sample_type, target_model_role,
                 failure_category, quality_status, approved, artifact_refs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate["candidate_id"],
                    trajectory["trajectory_id"],
                    candidate.get("roi_id"),
                    candidate.get("sample_type"),
                    candidate.get("target_model_role"),
                    candidate.get("failure_category"),
                    candidate.get("quality_status"),
                    int(bool(candidate.get("approved"))),
                    _dump(candidate.get("artifact_refs") or {}),
                ),
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts
            (artifact_id, run_id, trajectory_id, artifact_type, path, format, metadata_json, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"artifact_{trajectory['trajectory_id']}_trajectory",
                trajectory["run_id"],
                trajectory["trajectory_id"],
                "trajectory_json",
                trajectory_path,
                "json",
                _dump({"review_status": trajectory.get("review_status")}),
                sha256_file(trajectory_path),
            ),
        )
