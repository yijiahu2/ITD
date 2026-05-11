from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ITD_agent.evaluation_analysis.coco_error_decomposition import decompose_coco_errors
from ITD_agent.evaluation_analysis.expert_result_comparator import compare_expert_with_main
from ITD_agent.evaluation_analysis.geometry_failure_tags import build_geometry_failure_tags
from ITD_agent.evaluation_analysis.geometry_metrics import build_geometry_profile
from ITD_agent.evolution.expert.expert_task_builder import build_expert_tasks
from ITD_agent.evolution.expert.expert_task_runner import run_expert_tasks
from ITD_agent.evolution.expert.tile_image import materialize_expert_tile, offset_instances_to_full_image
from ITD_agent.evolution.fusion.local_roi_fusion import fuse_or_rollback
from ITD_agent.evolution.real_inference_adapter import (
    derive_dataset_input,
    resolve_image_path_for_coco,
    run_real_segmentation_for_sample,
)
from ITD_agent.evolution.roi.roi_candidate_builder import build_roi_candidates
from ITD_agent.evolution.roi.roi_clusterer import cluster_rois_for_expert_tiles
from ITD_agent.evolution.roi.roi_status_assigner import assign_roi_status, is_global_failure
from ITD_agent.evolution.state.repositories import write_run_record, write_state_records
from ITD_agent.evolution.trajectory.trajectory_builder import start_trajectory, summarize_trajectory
from ITD_agent.evolution.trajectory.trajectory_writer import write_trajectory
from ITD_agent.training_loop.sample_intake import intake_training_candidates_dry_run
from ITD_agent.training_loop.trigger_policy import evaluate_dry_run_trigger


def _load_structured(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON config when PyYAML is unavailable.") from exc
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _instances_by_image(coco: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations") or []:
        image_id = str(ann.get("image_id"))
        grouped[image_id] = [*grouped.get(image_id, []), dict(ann)]
    return grouped


def _select_images(images: list[dict[str, Any]], input_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    selected = list(images)
    include_ids = input_cfg.get("image_ids")
    if include_ids:
        include = {str(item) for item in include_ids}
        selected = [image for image in selected if str(image.get("id")) in include]
    if input_cfg.get("filter_resolvable_images", False) and input_cfg.get("image_root"):
        resolvable: list[dict[str, Any]] = []
        for image in selected:
            try:
                resolve_image_path_for_coco(image, input_cfg["image_root"])
            except FileNotFoundError:
                continue
            resolvable.append(image)
        selected = resolvable
    max_images = input_cfg.get("max_images")
    if max_images not in (None, "", 0):
        selected = selected[: int(max_images)]
    return selected


def _get_model_cfg(cfg: dict[str, Any], model_id: str | None) -> dict[str, Any]:
    if not model_id:
        return {}
    model_cfgs = cfg.get("model_configs") or {}
    return dict(model_cfgs.get(model_id) or {})


def _run_main_model(
    *,
    cfg: dict[str, Any],
    image: dict[str, Any],
    image_path: Path,
    pred_instances: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    main_cfg = cfg.get("main_model") or {}
    execution_mode = str(main_cfg.get("execution_mode") or "prediction_json")
    if execution_mode != "real":
        return {
            "status": "completed",
            "model_id": main_cfg.get("model_id", "legacy_cellpose_sam"),
            "execution_mode": execution_mode,
            "instances": pred_instances,
            "artifacts": {"prediction_json": (cfg.get("input") or {}).get("prediction_json")},
        }
    model_id = str(main_cfg.get("model_id") or "legacy_cellpose_sam")
    model_cfg = {**_get_model_cfg(cfg, model_id), **main_cfg}
    return run_real_segmentation_for_sample(
        base_config_path=str(cfg["runtime"]["base_config"]),
        image=image,
        image_path=image_path,
        output_dir=output_dir / "main_model",
        model_cfg=model_cfg,
        score_mode=str((cfg.get("runtime") or {}).get("prediction_score_mode") or "semantic_prior_mean_prob"),
    )


def _run_real_expert_tasks(
    *,
    cfg: dict[str, Any],
    image: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    tasks: list[Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    tile_enabled = bool((cfg.get("expert_models") or {}).get("tile_execution", False))
    for task in tasks:
        expert_model = str(task.expert_model)
        tile_info = {
            "status": "disabled",
            "tile_image_path": str(image_path),
            "offset_xy": [0.0, 0.0],
            "tile_window_px": list(task.tile_window_px),
        }
        task_image = image
        task_image_path = image_path
        if tile_enabled:
            tile_info = materialize_expert_tile(
                image_path=image_path,
                tile_window_px=list(task.tile_window_px),
                output_dir=output_dir / "expert_tiles" / task.expert_task_id,
                image_id=str(image.get("id")),
            )
            task_image_path = Path(str(tile_info["tile_image_path"]))
            task_image = {
                **image,
                "file_name": task_image_path.name,
                "width": int(tile_info.get("width") or image.get("width") or 1024),
                "height": int(tile_info.get("height") or image.get("height") or 1024),
            }
        cache_key = f"{expert_model}:{tile_info.get('tile_image_path')}:{tile_info.get('tile_window_px')}"
        if cache_key not in cache:
            expert_cfg = {**_get_model_cfg(cfg, expert_model), "model_id": expert_model}
            cache[cache_key] = run_real_segmentation_for_sample(
                base_config_path=str(cfg["runtime"]["base_config"]),
                image=task_image,
                image_path=task_image_path,
                output_dir=output_dir / "expert_models" / expert_model / task.expert_task_id,
                model_cfg=expert_cfg,
                score_mode=str((cfg.get("runtime") or {}).get("prediction_score_mode") or "semantic_prior_mean_prob"),
            )
        instances = list(cache[cache_key]["instances"])
        if tile_enabled and tile_info.get("status") == "materialized":
            instances = offset_instances_to_full_image(instances, list(tile_info.get("offset_xy") or [0.0, 0.0]))
        results.append(
            {
                "expert_task_id": task.expert_task_id,
                "expert_model": expert_model,
                "execution_mode": "real",
                "oracle_mock": False,
                "status": cache[cache_key]["status"],
                "instances": instances,
                "artifacts": dict(cache[cache_key]["artifacts"]),
                "tile_execution": tile_info,
            }
        )
    return results


def _decide_main_action(metrics: dict[str, Any], roi_count: int) -> dict[str, Any]:
    if not roi_count and metrics.get("false_negative_count", 0) == 0 and metrics.get("false_positive_count", 0) == 0:
        return {"decision": "accept_main", "reason": "no_supervised_errors_detected"}
    return {"decision": "escalate_expert", "reason": "actionable ROI clusters found"}


def _run_one_sample(
    *,
    cfg: dict[str, Any],
    run_id: str,
    image: dict[str, Any],
    gt_instances: list[dict[str, Any]],
    pred_instances: list[dict[str, Any]],
    config_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    image_id = str(image["id"])
    width = int(image.get("width") or 1024)
    height = int(image.get("height") or 1024)
    trajectory = start_trajectory(run_id=run_id, image=image, annotation_json=cfg["input"]["annotation_json"])
    trajectory["input_snapshot"]["gt_instance_count"] = len(gt_instances)
    trajectory["mainline_profile"] = str(cfg.get("mainline_profile") or "A_DOM_ONLY")
    image_path: Path | None = None
    if (cfg.get("main_model") or {}).get("execution_mode") == "real" or (cfg.get("expert_models") or {}).get("execution_mode") == "real":
        image_path = resolve_image_path_for_coco(image, cfg["input"]["image_root"])

    main_run = _run_main_model(
        cfg=cfg,
        image=image,
        image_path=image_path or Path(str(image.get("file_name", ""))),
        pred_instances=pred_instances,
        output_dir=output_dir / "samples" / f"image_{image_id}",
    )
    main_instances = list(main_run["instances"])
    trajectory["main_model_stage"] = {
        "model_id": main_run.get("model_id") or (cfg.get("main_model") or {}).get("model_id", "legacy_cellpose_sam"),
        "execution_result": {
            "status": main_run.get("status", "completed"),
            "execution_mode": (cfg.get("main_model") or {}).get("execution_mode", "prediction_json"),
            "config_path": config_path,
            "runtime_cfg": main_run.get("runtime_cfg") or {},
        },
        "prediction_artifacts": {
            **(main_run.get("artifacts") or {}),
            "instance_count": len(main_instances),
        },
    }

    matching_cfg = ((cfg.get("evaluation") or {}).get("matching") or {})
    error_decomp = decompose_coco_errors(
        gt_instances=gt_instances,
        pred_instances=main_instances,
        iou_threshold=float(matching_cfg.get("iou_threshold", 0.5)),
        weak_overlap_threshold=float(matching_cfg.get("weak_overlap_threshold", 0.1)),
    )
    geometry_profile = build_geometry_profile(main_instances)
    failure_tags = build_geometry_failure_tags(geometry_profile)
    geometry_review = {"geometry_profile": geometry_profile, "failure_tags": failure_tags}
    trajectory["main_eval_stage"] = {
        "coco_metrics": error_decomp.metrics,
        "error_decomposition": error_decomp.to_dict(),
    }
    trajectory["geometry_review_stage"] = geometry_review

    roi_policy = cfg.get("roi_policy") or {}
    rois = build_roi_candidates(
        image_id=image_id,
        image_size=(width, height),
        error_decomposition=error_decomp,
        geometry_review=geometry_review,
    )
    rois = assign_roi_status(rois, roi_policy)
    trajectory["main_decision_stage"] = _decide_main_action(error_decomp.metrics, len(rois))
    trajectory["roi_stage"]["roi_candidates"] = [roi.to_dict() for roi in rois]

    if trajectory["main_decision_stage"]["decision"] != "escalate_expert" or is_global_failure(rois, roi_policy):
        trajectory["fusion_stage"] = {
            "fusion_events": [{"decision": "main_only", "reason": trajectory["main_decision_stage"]["reason"]}],
            "final_result_source": "main_only",
            "instances": list(main_instances),
        }
    else:
        clusters = cluster_rois_for_expert_tiles(
            roi_candidates=rois,
            image_size=(width, height),
            tile_size=int(roi_policy.get("expert_tile_size_px", 1024)),
        )
        trajectory["roi_stage"]["roi_clusters"] = [cluster.to_dict() for cluster in clusters]
        expert_cfg = cfg.get("expert_models") or {}
        execution_mode = str(expert_cfg.get("execution_mode") or "mock")
        tasks = build_expert_tasks(
            trajectory_id=trajectory["trajectory_id"],
            roi_clusters=clusters,
            routing_policy=cfg.get("expert_routing_policy") or {},
            execution_mode=execution_mode,
        )
        trajectory["expert_task_stage"] = {
            "expert_tasks": [task.to_dict() for task in tasks],
            "routing_events": [task.routing_event for task in tasks],
        }
        if execution_mode == "real":
            expert_results = _run_real_expert_tasks(
                cfg=cfg,
                image=image,
                image_path=image_path,
                output_dir=output_dir / "samples" / f"image_{image_id}",
                tasks=tasks,
            )
        else:
            expert_results = run_expert_tasks(
                expert_tasks=tasks,
                gt_instances=gt_instances,
                main_instances=main_instances,
                execution_mode=execution_mode,
                expert_models_cfg=expert_cfg,
            )
        reviews = compare_expert_with_main(
            expert_tasks=tasks,
            expert_results=expert_results,
            main_instances=main_instances,
            gt_instances=gt_instances,
        )
        trajectory["expert_review_stage"] = {"expert_results": expert_results, "expert_reviews": reviews}
        fusion_result = fuse_or_rollback(
            main_instances=main_instances,
            expert_results=expert_results,
            expert_reviews=reviews,
            min_improvement_epsilon=float((cfg.get("adaptive_inference") or {}).get("min_improvement_epsilon", 0.01)),
        )
        trajectory["fusion_stage"] = fusion_result

    final_distill_roi_ids = {
        roi_id
        for review in trajectory["expert_review_stage"].get("expert_reviews", [])
        for roi_id in review.get("accepted_roi_ids", [])
    }
    rois_for_training = []
    for roi in rois:
        if roi.roi_id in final_distill_roi_ids:
            from dataclasses import replace

            rois_for_training.append(replace(roi, distill_eligible=True))
        else:
            rois_for_training.append(roi)
    training_candidates = intake_training_candidates_dry_run(
        trajectory_id=trajectory["trajectory_id"],
        roi_candidates=rois_for_training,
    )
    trajectory["pending_review_candidates"] = {
        "memory_candidates": [],
        "skill_candidates": [],
        "training_candidates": [candidate.to_dict() for candidate in training_candidates],
        "distillation_candidates": [
            {"trajectory_id": trajectory["trajectory_id"], "roi_id": roi.roi_id}
            for roi in rois_for_training
            if roi.distill_eligible
        ],
        "dry_run_training_trigger": evaluate_dry_run_trigger(training_candidates),
    }

    trajectory_path = write_trajectory(trajectory, output_dir)
    write_state_records(db_path=output_dir / "state.sqlite", trajectory=trajectory, trajectory_path=trajectory_path)
    return summarize_trajectory(trajectory, trajectory_path)


def run_evolve_infer_v1(config_path: str) -> dict[str, Any]:
    cfg = _load_structured(config_path)
    cfg["input"] = derive_dataset_input(cfg.get("input") or {})
    output_dir = Path(cfg.get("output_dir") or "outputs/evolve_coco_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    coco = _load_structured(cfg["input"]["annotation_json"])
    predictions = _load_structured(cfg["input"]["prediction_json"]) if cfg["input"].get("prediction_json") else {"annotations": []}
    gt_by_image = _instances_by_image(coco)
    pred_by_image = _instances_by_image(predictions)
    requires_real_images = (cfg.get("main_model") or {}).get("execution_mode") == "real" or (cfg.get("expert_models") or {}).get("execution_mode") == "real"
    selection_input = {
        **cfg["input"],
        "filter_resolvable_images": cfg["input"].get("filter_resolvable_images", requires_real_images),
    }
    images = _select_images(list(coco.get("images") or []), selection_input)
    summaries = [
        _run_one_sample(
            cfg=cfg,
            run_id=run_id,
            image=dict(image),
            gt_instances=gt_by_image.get(str(image.get("id")), []),
            pred_instances=pred_by_image.get(str(image.get("id")), []),
            config_path=config_path,
            output_dir=output_dir,
        )
        for image in images
    ]
    summary = {
        "run_id": run_id,
        "mode": str(cfg.get("mode") or "supervised_coco_evolve_v1"),
        "mainline_profile": str(cfg.get("mainline_profile") or "A_DOM_ONLY"),
        "output_dir": str(output_dir),
        "trajectory_count": len(summaries),
        "trajectories": summaries,
        "totals": {
            "roi_candidates": sum(item["roi_candidate_count"] for item in summaries),
            "expert_tasks": sum(item["expert_task_count"] for item in summaries),
            "training_candidates": sum(item["training_candidate_count"] for item in summaries),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_record(
        db_path=output_dir / "state.sqlite",
        run_id=run_id,
        mode=summary["mode"],
        mainline_profile=summary["mainline_profile"],
        config_path=config_path,
        output_dir=str(output_dir),
        status="completed",
        summary=summary,
    )
    return summary
