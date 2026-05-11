from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SUMMARY_TABLES = [
    "runs",
    "trajectories",
    "roi_candidates",
    "expert_tasks",
    "expert_reviews",
    "fusion_events",
    "training_candidates",
    "artifacts",
]

REVIEW_SUMMARY_TABLES = [
    "review_runs",
    "memory_records",
    "skill_records",
    "finetune_samples",
    "routing_candidates",
    "distillation_candidates",
    "review_events",
]


def summarize_state(db_path: str | Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in SUMMARY_TABLES}
        for table in REVIEW_SUMMARY_TABLES:
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = 0
        latest_run = conn.execute(
            "SELECT run_id, created_at, mode, mainline_profile, status FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return {
        "db_path": str(db_path),
        "counts": counts,
        "latest_run": {
            "run_id": latest_run[0],
            "created_at": latest_run[1],
            "mode": latest_run[2],
            "mainline_profile": latest_run[3],
            "status": latest_run[4],
        }
        if latest_run
        else None,
    }


def summarize_review_assets(db_path: str | Path, review_run_id: str | None = None) -> dict[str, Any]:
    where = ""
    params: tuple[Any, ...] = ()
    if review_run_id:
        where = " WHERE review_run_id = ?"
        params = (review_run_id,)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        counts = {}
        for table in ["memory_records", "skill_records", "finetune_samples", "routing_candidates", "distillation_candidates"]:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        review_counts = conn.execute(
            f"SELECT decision, COUNT(*) AS count FROM review_events{where} GROUP BY decision",
            params,
        ).fetchall()
        latest = conn.execute(
            "SELECT review_run_id, source_run_id, created_at, status, output_dir FROM review_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    latest_dict = dict(latest) if latest else None
    latest_report = None
    if latest_dict and latest_dict.get("output_dir"):
        report_path = Path(str(latest_dict["output_dir"])) / "reports" / "review_summary.json"
        if report_path.exists():
            try:
                latest_report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                latest_report = None
    return {
        "db_path": str(db_path),
        "asset_counts": counts,
        "review_decision_counts": {row["decision"]: row["count"] for row in review_counts},
        "latest_review_run": latest_dict,
        "latest_review_report": latest_report,
    }


def list_review_pending(db_path: str | Path, limit: int = 50) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT review_event_id, review_run_id, source_trajectory_id, candidate_id,
                   candidate_type, review_type, decision, reason
            FROM review_events
            WHERE decision IN ('defer', 'need_human_review', 'reject')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"db_path": str(db_path), "pending_or_blocked_review_events": [dict(row) for row in rows]}


def list_pending_reviews(db_path: str | Path, limit: int = 50) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        training_rows = conn.execute(
            """
            SELECT candidate_id, trajectory_id, roi_id, failure_category, quality_status, artifact_refs_json
            FROM training_candidates
            WHERE approved = 0
            ORDER BY candidate_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        trajectories = conn.execute(
            """
            SELECT trajectory_id, run_id, final_result_source, trajectory_path, review_status
            FROM trajectories
            WHERE review_status = 'pending'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {
        "db_path": str(db_path),
        "pending_training_candidates": [
            {
                **dict(row),
                "artifact_refs": json.loads(row["artifact_refs_json"] or "{}"),
            }
            for row in training_rows
        ],
        "pending_trajectories": [dict(row) for row in trajectories],
    }
