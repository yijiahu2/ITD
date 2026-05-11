from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evolution.adaptive_inference import run_adaptive_inference_stage
from ITD_agent.evolution.config_preflight import preflight_runtime_config
from ITD_agent.evolution.state.queries import (
    list_pending_reviews,
    list_review_pending,
    summarize_review_assets,
    summarize_state,
)
from ITD_agent.finetune_pool.review.finetune_bundle_exporter import export_finetune_bundle
from ITD_agent.finetune_pool.review.review_runner import run_review_stage
from ITD_agent.orchestration.orchestrator import run_itd_agent
from ITD_agent.training_loop.training_runner import run_training_loop


def run_full_workflow(config_path: str | Path) -> dict[str, Any]:
    return run_itd_agent(str(config_path))


def run_adaptive_workflow(config_path: str | Path) -> dict[str, Any]:
    return run_adaptive_inference_stage(str(config_path))


def preflight_workflow(config_path: str | Path) -> dict[str, Any]:
    return preflight_runtime_config(config_path)


def run_review_workflow(config_path: str | Path) -> dict[str, Any]:
    return run_review_stage(str(config_path))


def run_training_workflow(config_path: str | Path) -> dict[str, Any]:
    return run_training_loop(str(config_path))


def summarize_state_db(db_path: str | Path) -> dict[str, Any]:
    return summarize_state(db_path)


def list_pending_state_items(db_path: str | Path, *, limit: int = 50) -> dict[str, Any]:
    return list_pending_reviews(db_path, limit=limit)


def list_pending_review_items(db_path: str | Path, *, limit: int = 50) -> dict[str, Any]:
    return list_review_pending(db_path, limit=limit)


def summarize_review_state_assets(db_path: str | Path, *, review_run_id: str | None = None) -> dict[str, Any]:
    return summarize_review_assets(db_path, review_run_id=review_run_id)


def export_review_bundle(*, review_output_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    return export_finetune_bundle(review_output_dir=review_output_dir, out_dir=output_dir)
