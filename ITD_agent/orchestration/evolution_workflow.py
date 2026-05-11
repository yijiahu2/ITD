from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ITD_agent.finetune_pool.review.io_utils import load_structured, write_json
from ITD_agent.learning_gate.dispatcher import dispatch_learning_events
from ITD_agent.learning_gate.event_builder import (
    build_learning_events_from_review_result,
    build_learning_events_from_run_result,
    build_learning_events_from_training_result,
)
from ITD_agent.orchestration import workflow


def run_controlled_evolution(config_path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    closed_loop = cfg.get("closed_loop") or {}
    generated_cfg_root = Path(str(closed_loop.get("output_dir") or "outputs/controlled_evolution")) / "generated_configs"

    result: dict[str, Any] = {
        "command": "evolve",
        "mode": "controlled_self_evolution",
        "config_path": str(config_path),
        "run": None,
        "review": None,
        "training": None,
        "learning_events": [],
    }

    run_result = _run_stage_from_config(cfg["run_config"])
    result["run"] = run_result

    run_events = build_learning_events_from_run_result(run_result)
    dispatch_report = dispatch_learning_events(
        events=run_events,
        cfg=cfg.get("learning_gate") or {},
        output_dir=closed_loop.get("output_dir") or "outputs/controlled_evolution",
    )
    result["learning_events"].append({"stage": "post_run", "report": dispatch_report})

    if bool(closed_loop.get("review_after_run", True)):
        review_config_path = _materialize_review_config(
            template_config_path=cfg["review_config"],
            run_result=run_result,
            output_dir=generated_cfg_root,
        )
        review_result = workflow.review(review_config_path)
        result["review"] = review_result
        review_events = build_learning_events_from_review_result(review_result)
        dispatch_report = dispatch_learning_events(
            events=review_events,
            cfg=cfg.get("learning_gate") or {},
            output_dir=closed_loop.get("output_dir") or "outputs/controlled_evolution",
        )
        result["learning_events"].append({"stage": "post_review", "report": dispatch_report})

    if bool(closed_loop.get("train_after_review", False)):
        training_config_path = _materialize_training_config(
            template_config_path=cfg["training_config"],
            run_result=run_result,
            review_result=review_result if bool(closed_loop.get("review_after_run", True)) else None,
            output_dir=generated_cfg_root,
        )
        training_result = workflow.train(training_config_path)
        result["training"] = training_result
        training_events = build_learning_events_from_training_result(training_result)
        dispatch_report = dispatch_learning_events(
            events=training_events,
            cfg=cfg.get("learning_gate") or {},
            output_dir=closed_loop.get("output_dir") or "outputs/controlled_evolution",
        )
        result["learning_events"].append({"stage": "post_training", "report": dispatch_report})

    return result


def _run_stage_from_config(config_path: str | Path) -> dict[str, Any]:
    stage_cfg = load_structured(config_path)
    mode = str(stage_cfg.get("mode") or "").strip().lower()
    if mode == "adaptive_inference":
        return workflow.adaptive_inference(config_path)
    return workflow.run(config_path)


def _materialize_review_config(
    *,
    template_config_path: str | Path,
    run_result: dict[str, Any],
    output_dir: str | Path,
) -> str:
    review_cfg = load_structured(template_config_path)
    run_payload = run_result.get("result") or run_result
    run_output_dir = Path(str(run_payload.get("output_dir") or ""))
    source = dict(review_cfg.get("source") or {})
    output = dict(review_cfg.get("output") or {})
    source["run_id"] = run_payload.get("run_id")
    source["state_db_path"] = str(run_output_dir / "state.sqlite")
    source["artifact_root"] = str(run_output_dir)
    output["output_dir"] = str(run_output_dir / "review")
    review_cfg["source"] = source
    review_cfg["output"] = output
    return write_json(Path(output_dir) / "review_config.generated.json", review_cfg)


def _materialize_training_config(
    *,
    template_config_path: str | Path,
    run_result: dict[str, Any],
    review_result: dict[str, Any] | None,
    output_dir: str | Path,
) -> str:
    training_cfg = load_structured(template_config_path)
    run_payload = run_result.get("result") or run_result
    review_payload = (review_result or {}).get("result") or review_result or {}
    source = dict(training_cfg.get("source") or {})
    source["run_id"] = run_payload.get("run_id")
    if review_payload.get("output_dir"):
        source["review_asset_dir"] = review_payload.get("output_dir")
    training_cfg["source"] = source
    return write_json(Path(output_dir) / "training_config.generated.json", training_cfg)
