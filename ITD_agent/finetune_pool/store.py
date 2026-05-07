from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile

from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv
from ITD_agent.finetune_pool.contracts import FinetunePoolSample, PublicDatasetCandidate
from ITD_agent.finetune_pool.policy import build_finetune_trigger_snapshot, infer_failure_category
from ITD_agent.model_roles import EXPERT_MODEL_ROLE, MAIN_MODEL_ROLE, is_expert_model_role, normalize_model_role
from ITD_agent.planning.scheduler.expert_taxonomy import infer_expert_family_from_entry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINETUNE_POOL_ROOT = PROJECT_ROOT / "ITD_agent" / "finetune_pool"
LEGACY_FINETUNE_POOL_ROOT = PROJECT_ROOT / "ITD_agent" / "ITD_agent" / "finetune_pool"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _scene_profile(runtime_cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    mainline_profile = resolve_mainline_profile(runtime_cfg)
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(mainline_profile)
    allow_external_knowledge = bool(capabilities.get("allow_external_knowledge"))
    allow_public_datasets = bool(capabilities.get("allow_public_datasets"))
    data_processing = (summary.get("data_processing") or {}).get("processing_summary") or {}
    input_assessment = (
        ((summary.get("data_processing") or {}).get("input_assessment") or {})
        or ((summary.get("evaluation_analysis") or {}).get("input_assessment") or {})
    )
    scene_analysis = input_assessment.get("scene_analysis") or {}
    image_texture_analysis = scene_analysis.get("image_texture_analysis") or {}
    image_quality_analysis = scene_analysis.get("image_quality_analysis") or {}
    image_profiles = data_processing.get("image_profiles") or []
    manifest_summary = ((data_processing.get("metadata") or {}).get("input_manifest_summary") or {})
    knowledge_profiles = manifest_summary.get("domain_knowledge_items") or []
    public_profiles = manifest_summary.get("public_datasets") or []
    image_resolution = None
    if image_profiles:
        image_resolution = image_profiles[0].get("resolution_x_m") or image_profiles[0].get("resolution_y_m")
    return {
        "mainline_profile": mainline_profile,
        "input_modalities": (((summary.get("input_manifest") or {}).get("metadata") or {}).get("input_modalities") or {}),
        "run_name": summary.get("run_name") or runtime_cfg.get("run_name"),
        "forest_type": runtime_cfg.get("forest_type") or scene_analysis.get("forest_type"),
        "terrain_type": runtime_cfg.get("terrain_type") if capabilities.get("allow_dem") else None,
        "image_resolution": image_resolution,
        "knowledge_profile_types": sorted({str(item.get("normalized_type")) for item in knowledge_profiles if item.get("normalized_type")}) if allow_external_knowledge else [],
        "public_dataset_roles": sorted({role for item in public_profiles for role in (item.get("usage_roles") or [])}) if allow_public_datasets else [],
        "stand_condition_labels": ((scene_analysis.get("stand_condition") or {}).get("labels") or []),
        "texture_labels": image_texture_analysis.get("labels") or [],
        "image_texture_levels": image_texture_analysis.get("levels") or {},
        "quality_labels": image_quality_analysis.get("labels") or [],
        "image_quality_levels": image_quality_analysis.get("levels") or {},
    }


def _common_artifact_refs(runtime_cfg: dict[str, Any], summary: dict[str, Any], details_csv: str | None) -> dict[str, Any]:
    segmentation = summary.get("segmentation_model") or {}
    final_outputs = summary.get("final_outputs") or {}
    return {
        "input_image": runtime_cfg.get("input_image"),
        "reference_vector_path": runtime_cfg.get("reference_vector_path") or runtime_cfg.get("inventory_vector_path") or runtime_cfg.get("xiaoban_shp"),
        "dem_tif": runtime_cfg.get("dem_tif"),
        "details_csv": details_csv,
        "summary_json": summary.get("summary_json"),
        "metrics_json": summary.get("metrics_json"),
        "tree_crowns_shp": final_outputs.get("tree_crowns_shp") or segmentation.get("tree_crowns_shp"),
        "tree_points_shp": final_outputs.get("tree_points_shp") or segmentation.get("tree_points_shp"),
        "merged_inst_shp": summary.get("merged_inst_shp") or segmentation.get("y_inst_shp"),
    }


def _infer_target_model_role(summary: dict[str, Any]) -> str:
    roi_rounds = ((summary.get("planning_scheduler") or {}).get("roi_rounds") or [])
    return EXPERT_MODEL_ROLE if roi_rounds else MAIN_MODEL_ROLE


def _infer_target_expert_family(runtime_cfg: dict[str, Any], summary: dict[str, Any]) -> str:
    planning = summary.get("planning_scheduler") or {}
    finetune_plan = planning.get("finetune_training_plan") or {}
    explicit = str(finetune_plan.get("target_expert_family") or "").strip()
    if explicit:
        return explicit
    expert_plan = (planning.get("expert_model_call_plan") or planning.get("child_model_call_plan") or {})
    preferred_name = str(
        expert_plan.get("preferred_expert_model")
        or expert_plan.get("preferred_child_model")
        or ""
    ).strip()
    seg_models = (((runtime_cfg.get("ITD_agent") or {}).get("segmentation_models") or {}))
    expert_models = (seg_models.get("expert_models") or seg_models.get("child_models") or [])
    for entry in expert_models:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name") or entry.get("algorithm") or "").strip() == preferred_name:
            return infer_expert_family_from_entry(entry)
    return "cross_domain_generalist"


def _build_failure_samples(
    *,
    runtime_cfg: dict[str, Any],
    summary: dict[str, Any],
    details_summary: dict[str, Any],
    now_iso: str,
) -> list[FinetunePoolSample]:
    top_cases = details_summary.get("top_k_reference_units") or []
    run_name = str(summary.get("run_name") or runtime_cfg.get("run_name") or "unknown_run")
    target_module = (((summary.get("planning_scheduler") or {}).get("finetune_recommendation") or {}).get("target_module")) or "segmentation_model"
    target_role = _infer_target_model_role(summary)
    target_expert_family = _infer_target_expert_family(runtime_cfg, summary)
    scene_profile = _scene_profile(runtime_cfg, summary)
    artifact_refs = _common_artifact_refs(runtime_cfg, summary, summary.get("details_csv"))
    samples: list[FinetunePoolSample] = []
    for idx, case in enumerate(top_cases):
        source_type = "failed_roi_sample" if is_expert_model_role(target_role) or idx < 3 else "hard_case_sample"
        category = infer_failure_category(case)
        tags = [source_type, target_role, category]
        tags.append(target_expert_family)
        for key in ["landform_type", "slope_class", "aspect_class", "slope_position_class"]:
            if case.get(key):
                tags.append(str(case.get(key)))
        sample = FinetunePoolSample(
            sample_id=f"{run_name}-{idx + 1}-{uuid4().hex[:8]}",
            run_name=run_name,
            timestamp=now_iso,
            source_type=source_type,
            target_module=target_module,
            target_model_role=target_role,
            target_expert_family=target_expert_family,
            failure_category=category,
            scene_profile=scene_profile,
            artifact_refs={**artifact_refs, "problem_case": case},
            label_status="weak",
            ready_for_training=(source_type == "failed_roi_sample"),
            tags=sorted(set(tags)),
            metrics_snapshot={
                "error_score": case.get("error_score"),
                "tree_count_error_abs": case.get("tree_count_error_abs"),
                "mean_crown_width_error_abs": case.get("mean_crown_width_error_abs"),
                "closure_error_abs": case.get("closure_error_abs"),
                "density_error_abs": case.get("density_error_abs"),
            },
            metadata={"reference_unit_id": case.get("reference_unit_id")},
        )
        samples.append(sample)
    return samples


def _build_replay_good_sample(
    *,
    runtime_cfg: dict[str, Any],
    summary: dict[str, Any],
    now_iso: str,
) -> FinetunePoolSample | None:
    metrics = summary.get("metrics") or {}
    final_eval = summary.get("final_evaluation") or {}
    run_name = str(summary.get("run_name") or runtime_cfg.get("run_name") or "unknown_run")
    benchmark_metrics = final_eval.get("benchmark_metrics") or final_eval.get("metrics") or {}
    reference_metrics = final_eval.get("reference_quality_metrics") or metrics
    good_benchmark = benchmark_metrics and float(benchmark_metrics.get("ap50") or 0.0) >= 0.75
    good_reference = (
        float(reference_metrics.get("tree_count_error_ratio") or 1.0) <= 0.1
        and float(reference_metrics.get("mean_crown_width_error_ratio") or 1.0) <= 0.12
        and float(reference_metrics.get("closure_error_abs") or 1.0) <= 0.08
    )
    if not good_benchmark and not good_reference:
        return None
    return FinetunePoolSample(
        sample_id=f"{run_name}-replay-{uuid4().hex[:8]}",
        run_name=run_name,
        timestamp=now_iso,
        source_type="replay_good_sample",
        target_module="segmentation_model",
        target_model_role="main_model",
        target_expert_family="cross_domain_generalist",
        failure_category="replay_good",
        scene_profile=_scene_profile(runtime_cfg, summary),
        artifact_refs=_common_artifact_refs(runtime_cfg, summary, summary.get("details_csv")),
        label_status="pseudo",
        ready_for_training=True,
        tags=["replay_good_sample", "segmentation_model", "main_model"],
        metrics_snapshot={"benchmark_metrics": benchmark_metrics, "reference_metrics": reference_metrics},
    )


def _build_public_dataset_candidates(summary: dict[str, Any]) -> list[PublicDatasetCandidate]:
    processing_summary = ((summary.get("data_processing") or {}).get("processing_summary") or {})
    profiles = ((((processing_summary.get("metadata") or {}).get("input_manifest_summary") or {}).get("public_datasets")) or [])
    candidates: list[PublicDatasetCandidate] = []
    for idx, item in enumerate(profiles):
        metadata = item.get("metadata") or {}
        usage_roles = item.get("usage_roles") or [MAIN_MODEL_ROLE, EXPERT_MODEL_ROLE]
        expert_families = item.get("target_expert_families") or metadata.get("target_expert_families") or ["cross_domain_generalist"]
        for role in usage_roles:
            for expert_family in expert_families:
                candidates.append(
                    PublicDatasetCandidate(
                        candidate_id=f"{item.get('source_id') or 'dataset'}-{role}-{expert_family}-{idx}",
                        dataset_id=str(item.get("source_id") or f"dataset_{idx}"),
                        dataset_name=str(item.get("dataset_name") or metadata.get("dataset_name") or item.get("source_id") or f"dataset_{idx}"),
                        target_model_role=normalize_model_role(role),
                        target_expert_family=str(expert_family),
                        supported_failure_categories=list(metadata.get("supported_failure_categories") or []),
                        domain_tags=list(item.get("domain_tags") or metadata.get("domain_tags") or []),
                        terrain_tags=list(item.get("terrain_tags") or metadata.get("terrain_tags") or []),
                        forest_type=item.get("forest_type") or metadata.get("forest_type"),
                        sensor_type=item.get("sensor_type") or metadata.get("sensor_type"),
                        resolution_range=item.get("resolution_range") or metadata.get("resolution_range"),
                        annotation_type=item.get("annotation_type"),
                        label_quality=item.get("label_quality") or metadata.get("label_quality") or ("high" if item.get("finetune_ready") else "unknown"),
                        data_volume=metadata.get("data_volume"),
                        index_ref={
                            "root_path": item.get("root_path"),
                            "annotation_path": item.get("annotation_path"),
                            "usage_roles": usage_roles,
                            "target_expert_families": expert_families,
                        },
                        metadata=metadata,
                    )
                )
    return candidates


def _update_indexes(root: Path, samples: list[dict[str, Any]]) -> None:
    by_failure = _load_json(root / "index" / "by_failure_category.json")
    by_target = _load_json(root / "index" / "by_target_model.json")
    by_expert = _load_json(root / "index" / "by_expert_family.json")
    for item in samples:
        sample_id = str(item.get("sample_id"))
        failure = str(item.get("failure_category") or "uncategorized")
        role = normalize_model_role(item.get("target_model_role"), default=MAIN_MODEL_ROLE)
        expert_family = str(item.get("target_expert_family") or "cross_domain_generalist")
        by_failure.setdefault(failure, [])
        by_target.setdefault(role, [])
        by_expert.setdefault(expert_family, [])
        if sample_id not in by_failure[failure]:
            by_failure[failure].append(sample_id)
        if sample_id not in by_target[role]:
            by_target[role].append(sample_id)
        if sample_id not in by_expert[expert_family]:
            by_expert[expert_family].append(sample_id)
    _write_json(root / "index" / "by_failure_category.json", by_failure)
    _write_json(root / "index" / "by_target_model.json", by_target)
    _write_json(root / "index" / "by_expert_family.json", by_expert)


def register_finetune_pool_assets(
    *,
    runtime_cfg: dict[str, Any],
    summary: dict[str, Any],
    details_csv: str | None,
    input_manifest: dict[str, Any],
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
) -> dict[str, Any]:
    root = Path(finetune_pool_root)
    now_iso = datetime.now().isoformat(timespec="seconds")
    root.mkdir(parents=True, exist_ok=True)
    details_summary = summarize_details_csv(details_csv, top_k=10) if details_csv and Path(details_csv).exists() else {"top_k_reference_units": []}

    failure_samples = _build_failure_samples(
        runtime_cfg=runtime_cfg,
        summary=summary,
        details_summary=details_summary,
        now_iso=now_iso,
    )
    replay_sample = _build_replay_good_sample(runtime_cfg=runtime_cfg, summary=summary, now_iso=now_iso)
    public_candidates = _build_public_dataset_candidates(summary)
    all_samples = [item.to_dict() for item in failure_samples]
    if replay_sample:
        all_samples.append(replay_sample.to_dict())

    sample_log = None
    for item in all_samples:
        sample_log = _append_jsonl(root / "records" / "samples.jsonl", item)
    public_log = None
    for item in public_candidates:
        public_log = _append_jsonl(root / "records" / "public_dataset_candidates.jsonl", item.to_dict())

    snapshot = build_finetune_trigger_snapshot(
        samples=all_samples,
        public_candidates=[item.to_dict() for item in public_candidates],
        run_name=str(summary.get("run_name") or runtime_cfg.get("run_name") or "unknown_run"),
        timestamp=now_iso,
        target_module=(((summary.get("planning_scheduler") or {}).get("finetune_recommendation") or {}).get("target_module")),
        runtime_cfg=runtime_cfg,
    )
    snapshot_path = root / "records" / "latest_trigger_snapshot.json"
    _write_json(snapshot_path, snapshot.to_dict())
    _append_jsonl(root / "records" / "training_triggers.jsonl", snapshot.to_dict())
    _append_jsonl(
        root / "records" / "clusters.jsonl",
        {
            "timestamp": now_iso,
            "run_name": summary.get("run_name") or runtime_cfg.get("run_name"),
            "clusters": snapshot.cluster_summaries,
        },
    )
    _update_indexes(root, all_samples)

    run_name = str(summary.get("run_name") or runtime_cfg.get("run_name") or "unknown_run")
    manifest_payload = {
        "timestamp": now_iso,
        "run_name": run_name,
        "input_manifest": input_manifest,
        "details_csv": details_csv,
        "summary_json": summary.get("summary_json"),
        "sample_count": len(all_samples),
        "public_dataset_candidate_count": len(public_candidates),
        "trigger_snapshot": snapshot.to_dict(),
    }
    manifest_path = root / "cases" / f"{run_name}.json"
    _write_json(manifest_path, manifest_payload)

    return {
        "finetune_pool_root": str(root),
        "sample_log": sample_log,
        "public_dataset_candidate_log": public_log,
        "trigger_snapshot_json": str(snapshot_path),
        "case_manifest": str(manifest_path),
        "registered_samples": len(all_samples),
        "registered_failure_samples": len(failure_samples),
        "registered_public_dataset_candidates": len(public_candidates),
        "trigger_ready": snapshot.trigger_ready,
        "trigger_reason": snapshot.trigger_reason,
        "recommended_target_model_role": snapshot.recommended_target_model_role,
        "recommended_failure_category": snapshot.recommended_failure_category,
    }


def register_failed_cases_for_finetune_pool(
    *,
    runtime_cfg: dict[str, Any],
    summary: dict[str, Any],
    details_csv: str | None,
    input_manifest: dict[str, Any],
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
) -> dict[str, Any]:
    return register_finetune_pool_assets(
        runtime_cfg=runtime_cfg,
        summary=summary,
        details_csv=details_csv,
        input_manifest=input_manifest,
        finetune_pool_root=finetune_pool_root,
    )


def load_recent_failed_cases(
    *,
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
    limit: int = 10,
) -> list[dict[str, Any]]:
    path = Path(finetune_pool_root) / "records" / "samples.jsonl"
    rows = _load_jsonl(path)
    rows = [row for row in rows if row.get("source_type") in {"failed_roi_sample", "hard_case_sample"}]
    return rows[-limit:]
