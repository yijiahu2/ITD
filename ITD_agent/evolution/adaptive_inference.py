from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from input_layer.adapters import build_input_manifest
from input_layer.mainline_profiles import A_DOM_ONLY, normalize_mainline_profile
from input_layer.registry import register_input_bundle
from output_layer.contracts import FinalTreeCrownResult
from output_layer.publisher import publish_final_tree_crown_outputs

from ITD_agent.data_processing.context_object import build_coco_public_dataset_context
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
from ITD_agent.finetune_pool.query import load_finetune_pool_snapshot, load_recent_failed_cases
from ITD_agent.memory_store.query import (
    load_recent_execution_traces,
    load_recent_failure_patterns,
    load_recent_success_strategies,
)
from ITD_agent.planning.scheduler import build_evolve_infer_plan_context
from ITD_agent.skill_store.query import load_skill_records
from ITD_agent.segmentation.coco_utils import normalize_coco_instances
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


def _preview(values: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    return [dict(item) for item in values[: max(int(limit), 0)]]


def _build_experience_context(cfg: dict[str, Any]) -> dict[str, Any]:
    experience_cfg = cfg.get("experience_retrieval") or cfg.get("experience_context") or {}
    if experience_cfg.get("enabled", True) is False:
        return {
            "enabled": False,
            "memory_context": {"recent_success": [], "recent_failure": [], "recent_execution": []},
            "skill_context": {"records": []},
            "finetune_pool_context": {"snapshot": {}, "recent_failed_cases": []},
        }

    memory_limit = int(experience_cfg.get("memory_limit", 5))
    skill_limit = int(experience_cfg.get("skill_limit", 20))
    finetune_limit = int(experience_cfg.get("finetune_case_limit", 5))
    skill_records = load_skill_records(
        db_path=experience_cfg.get("state_db_path"),
        review_output_dir=experience_cfg.get("review_output_dir"),
        statuses=["draft", "shadow", "active"],
        limit=skill_limit,
    )
    return {
        "enabled": True,
        "source": "experience_retrieval",
        "memory_context": {
            "recent_success": _preview(load_recent_success_strategies(limit=memory_limit), limit=memory_limit),
            "recent_failure": _preview(load_recent_failure_patterns(limit=memory_limit), limit=memory_limit),
            "recent_execution": _preview(load_recent_execution_traces(limit=memory_limit), limit=memory_limit),
        },
        "skill_context": {
            "record_count": len(skill_records),
            "records": _preview(skill_records, limit=skill_limit),
        },
        "finetune_pool_context": {
            "snapshot": load_finetune_pool_snapshot(),
            "recent_failed_cases": _preview(load_recent_failed_cases(limit=finetune_limit), limit=finetune_limit),
        },
    }


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


def _decide_main_action(metrics: dict[str, Any], roi_count: int, *, global_failure: bool = False) -> dict[str, Any]:
    if not roi_count and metrics.get("false_negative_count", 0) == 0 and metrics.get("false_positive_count", 0) == 0:
        return {
            "decision": "accept_main",
            "reason": "no_supervised_errors_detected",
            "available_decisions": ["accept_main", "retry_main_plan", "escalate_expert", "record_failure"],
            "retry_main_plan_supported": False,
        }
    if global_failure:
        return {
            "decision": "record_failure",
            "reason": "global_failure_guard_triggered",
            "available_decisions": ["accept_main", "retry_main_plan", "escalate_expert", "record_failure"],
            "retry_main_plan_supported": False,
        }
    return {
        "decision": "escalate_expert",
        "reason": "actionable ROI clusters found",
        "available_decisions": ["accept_main", "retry_main_plan", "escalate_expert", "record_failure"],
        "retry_main_plan_supported": False,
    }


def _summarize_expert_decisions(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(review.get("decision") or "unknown") for review in reviews)
    if not reviews:
        return {
            "decision_counts": {},
            "primary_decision": "not_run",
            "available_decisions": ["accept", "partial_accept", "reject", "retry_expert_plan", "try_next_expert", "record_uncertain"],
            "retry_expert_plan_supported": False,
            "try_next_expert_supported": False,
        }
    if counts.get("accept"):
        primary = "accept"
    elif counts.get("partial_accept"):
        primary = "partial_accept"
    elif counts.get("reject") == len(reviews):
        primary = "reject"
    else:
        primary = "record_uncertain"
    return {
        "decision_counts": dict(counts),
        "primary_decision": primary,
        "available_decisions": ["accept", "partial_accept", "reject", "retry_expert_plan", "try_next_expert", "record_uncertain"],
        "retry_expert_plan_supported": False,
        "try_next_expert_supported": False,
    }


def _annotation_from_instance(instance: dict[str, Any], *, image_id: int, ann_id: int) -> dict[str, Any]:
    bbox = instance.get("bbox") or instance.get("bbox_xywh") or [0, 0, 0, 0]
    item = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(instance.get("category_id") or 1),
        "bbox": list(bbox),
        "score": float(instance.get("score", 1.0)),
        "area": float(instance.get("area") or (float(bbox[2]) * float(bbox[3]) if len(bbox) >= 4 else 0.0)),
    }
    if "segmentation" in instance:
        item["segmentation"] = instance["segmentation"]
    if "source" in instance:
        item["source"] = instance["source"]
    if "pred_id" in instance:
        item["source_pred_id"] = instance["pred_id"]
    return item


def _load_trajectory(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _build_manifest_cfg(cfg: dict[str, Any], input_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mainline_profile": normalize_mainline_profile(cfg.get("mainline_profile") or A_DOM_ONLY),
        "inputs": {
            "remote_sensing": {"images": []},
            "terrain": {"dem": []},
            "canopy": {"chm": []},
            "surface": {"dsm": []},
            "survey_data": {"tables": []},
            "industry_vectors": {"vectors": []},
            "domain_knowledge": {"items": []},
            "public_datasets": {
                "datasets": [
                    {
                        "id": str(input_cfg.get("dataset_id") or "coco_mainline_a"),
                        "format": "coco",
                        "annotation_path": input_cfg.get("annotation_json"),
                        "image_root": input_cfg.get("image_root"),
                        "root": input_cfg.get("image_root") or input_cfg.get("dataset_root"),
                        "required": True,
                        "metadata": {
                            "split": input_cfg.get("split"),
                            "resolved_split_dir": input_cfg.get("resolved_split_dir"),
                            "gt_visibility_policy": "evaluation_analysis_only",
                        },
                    }
                ]
            },
        },
    }


def _prepare_input_and_data_context(
    *,
    cfg: dict[str, Any],
    config_path: str,
    output_dir: Path,
    coco: dict[str, Any],
    selected_images: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_cfg = _build_manifest_cfg(cfg, cfg["input"])
    manifest = build_input_manifest(manifest_cfg, config_path=config_path)
    if not manifest.remote_sensing:
        manifest.preparation = None
    registry = register_input_bundle(manifest, {"output_dir": str(output_dir)})
    validation = manifest.validation.to_dict() if manifest.validation else {"status": "unknown", "issues": []}
    errors = [issue for issue in validation.get("issues", []) if issue.get("level") == "error"]
    if errors:
        raise ValueError(f"COCO input validation failed: {errors}")

    public_dataset_summary = build_coco_public_dataset_context(
        annotation_json=cfg["input"]["annotation_json"],
        image_root=cfg["input"].get("image_root"),
        selected_images=selected_images,
    )
    data_context = {
        "stage": "data_processing",
        "mainline_profile": manifest.metadata.get("mainline_profile"),
        "sample_count": len(selected_images),
        "public_dataset_summary": public_dataset_summary,
        "image_profile_summary": {
            "image_count": len(coco.get("images") or []),
            "selected_image_count": len(selected_images),
            "source": "coco.images",
        },
        "artifact_paths": {
            **registry,
            "public_dataset_summary_json": _write_json(
                output_dir / "data_processing" / "summaries" / "public_dataset_summary.json",
                public_dataset_summary,
            ),
        },
    }
    _write_json(output_dir / "data_processing" / "summaries" / "evolve_infer_data_context.json", data_context)
    return {
        "manifest": manifest.to_dict(),
        "registry": registry,
        "validation": validation,
        "data_context": data_context,
    }


def _coco_quality_score(metrics: dict[str, Any]) -> float:
    matched = float(metrics.get("matched_count") or 0.0)
    penalties = sum(
        float(metrics.get(key) or 0.0)
        for key in [
            "false_negative_count",
            "false_positive_count",
            "under_segmentation_count",
            "over_segmentation_count",
        ]
    )
    return matched - penalties


def _build_pending_review_candidates(
    *,
    trajectory: dict[str, Any],
    rois: list[Any],
    training_candidates: list[Any],
) -> dict[str, Any]:
    failure_counts = Counter(roi.level1_error_type for roi in rois)
    memory_candidates = [
        {
            "candidate_type": "memory_candidate",
            "trajectory_id": trajectory["trajectory_id"],
            "status": "pending_review",
            "evidence": {
                "main_decision": (trajectory.get("main_decision_stage") or {}).get("decision"),
                "final_result_source": (trajectory.get("fusion_stage") or {}).get("final_result_source"),
                "failure_counts": dict(failure_counts),
            },
            "write_policy": "candidate_only_no_auto_memory_write",
        }
    ] if rois else []
    skill_candidates = [
        {
            "candidate_type": "skill_candidate",
            "trajectory_id": trajectory["trajectory_id"],
            "failure_family": family,
            "status": "pending_review",
            "write_policy": "candidate_only_no_auto_skill_update",
        }
        for family in sorted({roi.failure_family for roi in rois if roi.review_status == "actionable"})
    ]
    routing_update_candidates = [
        {
            "candidate_type": "routing_update_candidate",
            "trajectory_id": trajectory["trajectory_id"],
            "expert_task_id": review.get("expert_task_id"),
            "expert_decision": review.get("decision"),
            "level1_error_type": next(
                (
                    task.get("level1_error_type")
                    for task in (trajectory.get("expert_task_stage") or {}).get("expert_tasks") or []
                    if task.get("expert_task_id") == review.get("expert_task_id")
                ),
                None,
            ),
            "status": "pending_review",
            "write_policy": "candidate_only_no_auto_routing_update",
        }
        for review in (trajectory.get("expert_review_stage") or {}).get("expert_reviews") or []
        if review.get("decision") in {"reject", "record_uncertain", "partial_accept"}
    ]
    final_distill_roi_ids = {
        roi_id
        for review in (trajectory.get("expert_review_stage") or {}).get("expert_reviews", [])
        for roi_id in review.get("accepted_roi_ids", [])
    }
    return {
        "memory_candidates": memory_candidates,
        "skill_candidates": skill_candidates,
        "training_candidates": [candidate.to_dict() for candidate in training_candidates],
        "routing_update_candidates": routing_update_candidates,
        "distillation_candidates": [
            {"trajectory_id": trajectory["trajectory_id"], "roi_id": roi.roi_id, "status": "pending_review"}
            for roi in rois
            if roi.roi_id in final_distill_roi_ids
        ],
        "dry_run_training_trigger": evaluate_dry_run_trigger(training_candidates),
        "auto_update_policy": {
            "write_memory": False,
            "write_skill": False,
            "start_training": False,
            "update_model_weights": False,
            "update_routing_policy": False,
        },
    }


def _write_foreground_outputs(
    *,
    run_id: str,
    cfg: dict[str, Any],
    coco: dict[str, Any],
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    final_root = output_dir / "final_outputs"
    selected_image_ids = {str(item.get("image_id")) for item in summaries}
    selected_images = [dict(image) for image in (coco.get("images") or []) if str(image.get("id")) in selected_image_ids]
    selected_image_by_id = {str(image.get("id")): dict(image) for image in selected_images}
    annotations: list[dict[str, Any]] = []
    instances_by_image: dict[str, list[dict[str, Any]]] = {}
    final_metrics_by_image: dict[str, dict[str, Any]] = {}
    final_source_counts: Counter[str] = Counter()
    all_final_instances: list[dict[str, Any]] = []
    trajectory_paths: list[str] = []
    first_input_dom_path: str | None = None
    first_geometry_metrics: dict[str, Any] | None = None
    ann_id = 1
    for summary in summaries:
        trajectory = _load_trajectory(summary["trajectory_path"])
        trajectory_paths.append(summary["trajectory_path"])
        image_id = int(trajectory["image_id"])
        instances = [dict(item) for item in (trajectory.get("fusion_stage") or {}).get("instances") or []]
        instances_by_image[str(image_id)] = instances
        all_final_instances.extend(instances)
        final_source_counts[str((trajectory.get("fusion_stage") or {}).get("final_result_source") or "unknown")] += 1
        final_metrics_by_image[str(image_id)] = (trajectory.get("final_evaluation_stage") or {}).get("coco_metrics") or {}
        if first_geometry_metrics is None:
            first_geometry_metrics = (trajectory.get("geometry_review_stage") or {}).get("geometry_profile") or {}
        if first_input_dom_path is None:
            image_info = selected_image_by_id.get(str(image_id)) or {}
            if (cfg.get("input") or {}).get("image_root"):
                try:
                    first_input_dom_path = str(resolve_image_path_for_coco(image_info, cfg["input"]["image_root"]))
                except Exception:
                    first_input_dom_path = None
        for instance in instances:
            annotations.append(_annotation_from_instance(instance, image_id=image_id, ann_id=ann_id))
            ann_id += 1

    prediction_payload = {
        "run_id": run_id,
        "mode": str(cfg.get("mode") or "adaptive_inference"),
        "images": selected_images,
        "annotations": annotations,
        "categories": list(coco.get("categories") or []),
    }
    instances_payload = {
        "run_id": run_id,
        "instances_by_image": instances_by_image,
    }
    metrics_payload = {
        "run_id": run_id,
        "image_count": len(summaries),
        "final_instance_count": len(annotations),
        "final_result_source_counts": dict(final_source_counts),
        "metrics_by_image": final_metrics_by_image,
        "totals": {
            "roi_candidates": sum(item["roi_candidate_count"] for item in summaries),
            "expert_tasks": sum(item["expert_task_count"] for item in summaries),
            "training_candidates": sum(item["training_candidate_count"] for item in summaries),
        },
    }
    prediction_path = _write_json(final_root / "final_prediction.json", prediction_payload)
    instances_path = _write_json(final_root / "final_instances.json", instances_payload)
    metrics_path = _write_json(final_root / "final_metrics.json", metrics_payload)
    report_payload = {
        "run_id": run_id,
        "mainline_profile": str(cfg.get("mainline_profile") or "A_DOM_ONLY"),
        "status": "frozen",
        "summary": metrics_payload,
        "trajectory_paths": trajectory_paths,
        "background_evolution_policy": "candidate_only_no_auto_training_or_model_update",
    }
    report_json_path = _write_json(final_root / "final_report.json", report_payload)
    report_md_path = final_root / "final_report.md"
    report_md_path.write_text(
        "\n".join(
            [
                "# Evolve Infer Final Report",
                "",
                f"- run_id: `{run_id}`",
                f"- mainline_profile: `{report_payload['mainline_profile']}`",
                f"- image_count: `{metrics_payload['image_count']}`",
                f"- final_instance_count: `{metrics_payload['final_instance_count']}`",
                f"- roi_candidates: `{metrics_payload['totals']['roi_candidates']}`",
                f"- expert_tasks: `{metrics_payload['totals']['expert_tasks']}`",
                f"- training_candidates: `{metrics_payload['totals']['training_candidates']}`",
                "- training_policy: `candidate_only_no_auto_training_or_model_update`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    bundle = {
        "run_id": run_id,
        "foreground_goal": "single_tree_crown_detection_and_extraction",
        "status": "frozen",
        "output_type": "coco_instance_prediction",
        "final_prediction_json": prediction_path,
        "final_instances_json": instances_path,
        "final_metrics_json": metrics_path,
        "final_report_json": report_json_path,
        "final_report_md": str(report_md_path),
        "trajectory_count": len(summaries),
        "state_db": str(output_dir / "state.sqlite"),
        "trajectory_paths": [item["trajectory_path"] for item in summaries],
        "background_evolution": {
            "trajectory_store": str(output_dir / "trajectories"),
            "state_db": str(output_dir / "state.sqlite"),
            "pending_review_candidate_count": metrics_payload["totals"]["training_candidates"],
            "training_trigger_mode": "dry_run_only",
        },
    }
    bundle_path = _write_json(final_root / "final_result_bundle.json", bundle)
    output_layer_publish = publish_final_tree_crown_outputs(
        result=FinalTreeCrownResult(
            run_id=run_id,
            output_dir=str(final_root),
            input_dom_path=first_input_dom_path,
            instances=all_final_instances,
            semantic_mask_tif=None,
            semantic_mask_png=None,
            coordinate_mode="geospatial" if first_input_dom_path and Path(first_input_dom_path).suffix.lower() in {".tif", ".tiff"} else "pixel",
            image_width=int(selected_images[0].get("width") or 0) if selected_images else None,
            image_height=int(selected_images[0].get("height") or 0) if selected_images else None,
            categories=list(coco.get("categories") or []),
            gt_metrics=metrics_payload,
            geometry_metrics=first_geometry_metrics,
            trajectory_paths=trajectory_paths,
            report_markdown=report_md_path.read_text(encoding="utf-8"),
            report_json=report_payload,
            metadata={
                "source_adapter": "coco_evolve_infer",
                "final_prediction_json": prediction_path,
                "final_instances_json": instances_path,
                "final_metrics_json": metrics_path,
                "state_db": str(output_dir / "state.sqlite"),
            },
        ),
        publish_root=final_root,
    )
    final_bundle = {
        **bundle,
        "final_result_bundle_json": bundle_path,
        "output_layer": output_layer_publish,
    }
    _write_json(final_root / "final_result_bundle.json", final_bundle)
    return final_bundle


def _build_final_evaluation(gt_instances: list[dict[str, Any]], final_instances: list[dict[str, Any]], matching_cfg: dict[str, Any]) -> dict[str, Any]:
    final_decomp = decompose_coco_errors(
        gt_instances=gt_instances,
        pred_instances=final_instances,
        iou_threshold=float(matching_cfg.get("iou_threshold", 0.5)),
        weak_overlap_threshold=float(matching_cfg.get("weak_overlap_threshold", 0.1)),
    )
    return {
        "coco_metrics": {**final_decomp.metrics, "quality_score": _coco_quality_score(final_decomp.metrics)},
        "error_decomposition": final_decomp.to_dict(),
    }


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
    gt_instances = normalize_coco_instances(gt_instances, image_id=image.get("id"), source="coco_gt")
    pred_instances = normalize_coco_instances(pred_instances, image_id=image.get("id"), source="main_prediction_input")
    trajectory = start_trajectory(run_id=run_id, image=image, annotation_json=cfg["input"]["annotation_json"])
    trajectory["input_snapshot"]["gt_instance_count"] = len(gt_instances)
    trajectory["mainline_profile"] = str(cfg.get("mainline_profile") or "A_DOM_ONLY")
    trajectory["experience_context"] = cfg.get("_experience_context") or {}
    trajectory["data_processing_stage"] = {
        **(cfg.get("_data_processing_context") or {}),
        "sample_context": {
            "image_id": image_id,
            "width": width,
            "height": height,
            "file_name": image.get("file_name"),
            "gt_visibility_policy": "evaluation_analysis_only",
        },
    }
    trajectory["planning_stage"] = cfg.get("_planning_context") or {}
    trajectory["artifact_paths"] = {
        key: value
        for key, value in {
            **(((cfg.get("_input_context") or {}).get("registry") or {})),
            **(((cfg.get("_data_processing_context") or {}).get("artifact_paths") or {})),
        }.items()
        if isinstance(value, str)
    }
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
    main_instances = normalize_coco_instances(list(main_run["instances"]), image_id=image.get("id"), source="main_model")
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
            "standardized_instance_format": "coco_xywh_instance_v1",
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
    main_metrics = {**error_decomp.metrics, "quality_score": _coco_quality_score(error_decomp.metrics)}
    trajectory["main_eval_stage"] = {
        "coco_metrics": main_metrics,
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
    global_failure = is_global_failure(rois, roi_policy)
    trajectory["main_decision_stage"] = _decide_main_action(error_decomp.metrics, len(rois), global_failure=global_failure)
    trajectory["roi_stage"]["roi_candidates"] = [roi.to_dict() for roi in rois]

    if trajectory["main_decision_stage"]["decision"] != "escalate_expert":
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
        expert_results = [
            {
                **result,
                "instances": normalize_coco_instances(
                    list(result.get("instances") or []),
                    image_id=image.get("id"),
                    source=f"expert_model:{result.get('expert_model') or 'unknown'}",
                ),
            }
            for result in expert_results
        ]
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
        trajectory["expert_decision_stage"] = _summarize_expert_decisions(reviews)
        trajectory["fusion_stage"] = fusion_result

    final_instances = normalize_coco_instances(
        list((trajectory.get("fusion_stage") or {}).get("instances") or []),
        image_id=image.get("id"),
        source="final_frozen",
    )
    trajectory["final_evaluation_stage"] = _build_final_evaluation(gt_instances, final_instances, matching_cfg)
    final_score = float((trajectory["final_evaluation_stage"].get("coco_metrics") or {}).get("quality_score") or 0.0)
    main_score = float(main_metrics.get("quality_score") or 0.0)
    if (
        (trajectory.get("fusion_stage") or {}).get("final_result_source") not in {"main_only", "rollback_to_main"}
        and final_score + float((cfg.get("adaptive_inference") or {}).get("min_improvement_epsilon", 0.01)) < main_score
    ):
        trajectory["fusion_stage"] = {
            "decision": "rollback_to_main",
            "final_result_source": "rollback_to_main",
            "instances": list(main_instances),
            "fusion_events": [
                *list((trajectory.get("fusion_stage") or {}).get("fusion_events") or []),
                {
                    "decision": "rollback_to_main",
                    "reason": "fusion_degraded_after_reassessment",
                    "main_quality_score": main_score,
                    "candidate_quality_score": final_score,
                },
            ],
        }
        final_instances = list(main_instances)
        trajectory["final_evaluation_stage"] = _build_final_evaluation(gt_instances, final_instances, matching_cfg)

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
    trajectory["pending_review_candidates"] = _build_pending_review_candidates(
        trajectory=trajectory,
        rois=rois_for_training,
        training_candidates=training_candidates,
    )

    trajectory_path = write_trajectory(trajectory, output_dir)
    write_state_records(db_path=output_dir / "state.sqlite", trajectory=trajectory, trajectory_path=trajectory_path)
    return summarize_trajectory(trajectory, trajectory_path)


def run_adaptive_inference_stage(config_path: str) -> dict[str, Any]:
    cfg = _load_structured(config_path)
    cfg["mainline_profile"] = normalize_mainline_profile(cfg.get("mainline_profile") or A_DOM_ONLY)
    cfg["input"] = derive_dataset_input(cfg.get("input") or {})
    output_dir = Path(cfg.get("output_dir") or "outputs/adaptive_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    cfg["_experience_context"] = _build_experience_context(cfg)
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
    cfg["_input_context"] = _prepare_input_and_data_context(
        cfg=cfg,
        config_path=config_path,
        output_dir=output_dir,
        coco=coco,
        selected_images=images,
    )
    cfg["_data_processing_context"] = cfg["_input_context"]["data_context"]
    cfg["_planning_context"] = build_evolve_infer_plan_context(
        cfg=cfg,
        input_manifest=cfg["_input_context"]["manifest"],
        data_processing_context=cfg["_data_processing_context"],
        experience_context=cfg["_experience_context"],
    )
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
    final_outputs = _write_foreground_outputs(
        run_id=run_id,
        cfg=cfg,
        coco=coco,
        summaries=summaries,
        output_dir=output_dir,
    )
    summary = {
        "run_id": run_id,
        "mode": str(cfg.get("mode") or "adaptive_inference"),
        "mainline_profile": str(cfg.get("mainline_profile") or "A_DOM_ONLY"),
        "output_dir": str(output_dir),
        "foreground_goal": "single_tree_crown_detection_and_extraction",
        "final_outputs": final_outputs,
        "input_layer": {
            "validation": cfg["_input_context"]["validation"],
            "registry": cfg["_input_context"]["registry"],
            "mainline_profile": cfg["mainline_profile"],
        },
        "data_processing": cfg["_data_processing_context"],
        "planning_context": cfg["_planning_context"],
        "experience_context": cfg["_experience_context"],
        "background_evolution": {
            "trajectory_store": str(output_dir / "trajectories"),
            "state_db": str(output_dir / "state.sqlite"),
            "review_status": "pending",
            "training_trigger_mode": "dry_run_only",
        },
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
