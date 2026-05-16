from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ITD_agent.orchestration.evolution_workflow import run_controlled_evolution
from ITD_agent.orchestration.run_context import build_config_context, build_export_context, build_state_context
from ITD_agent.orchestration.stage_runner import (
    export_review_bundle,
    list_pending_review_items,
    list_pending_state_items,
    preflight_workflow,
    run_adaptive_workflow,
    run_full_workflow,
    run_review_workflow,
    run_training_workflow,
    summarize_review_state_assets,
    summarize_state_db,
)


@dataclass(frozen=True)
class WorkflowResult:
    command: str
    context: dict[str, Any]
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_workflow(command: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    commands = {
        "evolve-infer": evolve_infer,
        "evolve": evolve,
        "run": run,
        "coco-png-infer": coco_png_infer,
        "adaptive-inference": adaptive_inference,
        "review": review,
        "train": train,
        "state": state,
        "export": export,
    }
    if command not in commands:
        raise ValueError(f"Unsupported workflow command: {command}")
    return commands[command](*args, **kwargs)


def run_stage(command: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return run_workflow(command, *args, **kwargs)


def evolve(config_path: str | Path) -> dict[str, Any]:
    return run_controlled_evolution(str(config_path))


def run(config_path: str | Path) -> dict[str, Any]:
    ctx = build_config_context(config_path)
    result = run_full_workflow(config_path)
    return {"command": "run", "context": ctx.to_dict(), "result": result}


def coco_png_infer(
    *,
    template: str | Path,
    dataset_root: str | Path,
    image_root: str | Path,
    annotation: str | Path,
    output_dir: str | Path,
    run_name: str,
    split: str = "validation",
    max_images: int | None = None,
    image_ids: list[str] | None = None,
    image_names: list[str] | None = None,
    max_expert_rounds: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    from ITD_agent.orchestration.coco_png_pipeline import run_coco_png_pipeline

    result = run_coco_png_pipeline(
        template=template,
        dataset_root=dataset_root,
        image_root=image_root,
        annotation=annotation,
        output_dir=output_dir,
        run_name=run_name,
        split=split,
        max_images=max_images,
        image_ids=image_ids,
        image_names=image_names,
        max_expert_rounds=max_expert_rounds,
        device=device,
    )
    return {
        "command": "coco-png-infer",
        "context": {
            "template": str(template),
            "dataset_root": str(dataset_root),
            "image_root": str(image_root),
            "annotation": str(annotation),
            "output_dir": str(output_dir),
            "run_name": run_name,
        },
        "result": result,
    }


def adaptive_inference(config_path: str | Path) -> dict[str, Any]:
    ctx = build_config_context(config_path)
    result = run_adaptive_workflow(config_path)
    return {"command": "adaptive-inference", "context": ctx.to_dict(), "result": result}


def evolve_infer(config_path: str | Path) -> dict[str, Any]:
    ctx = build_config_context(config_path)
    result = run_adaptive_workflow(config_path)
    return {"command": "evolve-infer", "context": ctx.to_dict(), "result": result}


def preflight(config_path: str | Path) -> dict[str, Any]:
    ctx = build_config_context(config_path)
    result = preflight_workflow(config_path)
    return {"command": "preflight", "context": ctx.to_dict(), "result": result}


def review(config_path: str | Path) -> dict[str, Any]:
    ctx = build_config_context(config_path)
    result = run_review_workflow(config_path)
    return {"command": "review", "context": ctx.to_dict(), "result": result}


def train(config_path: str | Path) -> dict[str, Any]:
    ctx = build_config_context(config_path)
    result = run_training_workflow(config_path)
    return {"command": "train", "context": ctx.to_dict(), "result": result}


def state(db_path: str | Path, *, detail: str = "summary", limit: int = 50, review_run_id: str | None = None) -> dict[str, Any]:
    ctx = build_state_context(db_path)
    if detail == "pending":
        result = list_pending_state_items(db_path, limit=limit)
    elif detail == "review-pending":
        result = list_pending_review_items(db_path, limit=limit)
    elif detail == "review-assets":
        result = summarize_review_state_assets(db_path, review_run_id=review_run_id)
    else:
        result = summarize_state_db(db_path)
    return {"command": "state", "detail": detail, "context": ctx.to_dict(), "result": result}


def export(run_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    ctx = build_export_context(run_dir, output_path)
    source = Path(run_dir)
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    copied = _copy_exportable_artifacts(source, destination)
    manifest = {"source_run_dir": str(source), "output_dir": str(destination), "copied": copied}
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"command": "export", "context": ctx.to_dict(), "result": manifest}


def export_finetune_pool(review_output_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    result = export_review_bundle(review_output_dir=review_output_dir, output_dir=output_dir)
    return {"command": "export-review-bundle", "result": result}


def _copy_exportable_artifacts(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for name in [
        "ITD_agent_run_summary.json",
        "run_summary.json",
        "final_evaluation_report.md",
        "final_evaluation_report.json",
        "state.sqlite",
    ]:
        src = source / name
        if src.exists() and src.is_file():
            dst = destination / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    for dirname in ["trajectories", "reports", "final_outputs"]:
        src_dir = source / dirname
        if src_dir.exists() and src_dir.is_dir():
            dst_dir = destination / dirname
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            copied.append(str(dst_dir))
    return copied
