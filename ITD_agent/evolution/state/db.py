from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    mainline_profile TEXT NOT NULL,
    config_path TEXT,
    output_dir TEXT,
    status TEXT NOT NULL,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    main_model TEXT,
    final_status TEXT,
    final_result_source TEXT,
    trajectory_path TEXT NOT NULL,
    review_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roi_candidates (
    roi_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    image_id TEXT,
    level1_error_type TEXT,
    failure_family TEXT,
    severity_score REAL,
    confidence_level TEXT,
    review_status TEXT,
    expert_eligible INTEGER,
    training_eligible INTEGER,
    distill_eligible INTEGER,
    bbox_json TEXT,
    tags_json TEXT,
    geometry_json TEXT
);

CREATE TABLE IF NOT EXISTS expert_tasks (
    expert_task_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    expert_model TEXT,
    failure_family TEXT,
    level1_error_type TEXT,
    roi_ids_json TEXT,
    tile_window_json TEXT,
    status TEXT,
    trigger_reason_json TEXT
);

CREATE TABLE IF NOT EXISTS expert_reviews (
    review_id TEXT PRIMARY KEY,
    expert_task_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    improvement_json TEXT,
    safety_json TEXT,
    accepted_roi_ids_json TEXT,
    rejected_roi_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS fusion_events (
    fusion_event_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    decision TEXT,
    fused_result_path TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS training_candidates (
    candidate_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    roi_id TEXT,
    sample_type TEXT,
    target_model_role TEXT,
    failure_category TEXT,
    quality_status TEXT,
    approved INTEGER DEFAULT 0,
    artifact_refs_json TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    trajectory_id TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    format TEXT,
    metadata_json TEXT,
    sha256 TEXT
);

CREATE TABLE IF NOT EXISTS v2_review_runs (
    review_run_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    config_path TEXT,
    output_dir TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS memory_records (
    memory_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    source_trajectory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    level1_error_type TEXT,
    failure_family TEXT,
    summary TEXT,
    evidence_refs_json TEXT,
    metrics_snapshot_json TEXT,
    artifact_refs_json TEXT,
    confidence TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_records (
    skill_id TEXT PRIMARY KEY,
    skill_type TEXT NOT NULL,
    name TEXT NOT NULL,
    source_run_ids_json TEXT,
    source_trajectory_ids_json TEXT,
    trigger_conditions_json TEXT,
    recommended_action_json TEXT,
    evidence_summary_json TEXT,
    safety_constraints_json TEXT,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS finetune_samples (
    sample_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    source_trajectory_id TEXT NOT NULL,
    source_roi_id TEXT,
    image_id TEXT,
    sample_type TEXT NOT NULL,
    target_model_role TEXT,
    target_error_type TEXT,
    image_crop_path TEXT,
    gt_mask_path TEXT,
    main_pred_path TEXT,
    expert_pred_path TEXT,
    metadata_path TEXT,
    quality_score REAL,
    review_status TEXT NOT NULL,
    export_status TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_candidates (
    routing_candidate_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    source_trajectory_id TEXT NOT NULL,
    level1_error_type TEXT,
    failure_family TEXT,
    expert_model TEXT,
    expert_decision TEXT,
    improvement_summary_json TEXT,
    safety_summary_json TEXT,
    recommendation TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS distillation_candidates (
    distillation_candidate_id TEXT PRIMARY KEY,
    source_run_id TEXT NOT NULL,
    source_trajectory_id TEXT NOT NULL,
    source_roi_id TEXT,
    expert_model TEXT,
    quality_tier TEXT,
    evidence_refs_json TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS v2_review_events (
    review_event_id TEXT PRIMARY KEY,
    review_run_id TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    source_trajectory_id TEXT,
    candidate_id TEXT,
    candidate_type TEXT,
    review_type TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    evidence_refs_json TEXT,
    guardrail_result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL
);
"""


def connect_state_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    return conn
