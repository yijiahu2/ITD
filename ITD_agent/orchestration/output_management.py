from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from ITD_agent.contracts import FinalDeliverables
from ITD_agent.finetune_pool import export_finetune_dataset_bundle
from ITD_agent.finetune_pool.store import register_finetune_pool_assets
from ITD_agent.memory_store.store import (
    record_execution,
    record_failure_pattern,
    record_run_retrospective,
    record_success_strategy,
)
from ITD_agent.orchestration.runtime_paths import get_eval_output_paths
from ITD_agent.orchestration.runtime_support import (
    copy_optional_file,
    remove_path,
    remove_vector_dataset,
)
from output_layer.publisher import publish_segmentation_deliverables


RUN_SUMMARY_FILENAME = "ITD_agent_run_summary.json"
LEGACY_RUN_SUMMARY_FILENAME = "run_experiment_summary.json"
RUN_REPORT_FILENAME = "final_evaluation_report.md"
RUN_REPORT_JSON_FILENAME = "final_evaluation_report.json"
LEGACY_RUN_REPORT_FILENAME = "run_experiment_report.md"


def keep_legacy_output_aliases(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("keep_legacy_output_aliases", False))


def _resolve_inst_shp(summary: dict[str, Any]) -> str | None:
    return summary.get("merged_inst_shp") or (summary.get("segmentation_model") or {}).get("y_inst_shp")


def _resolve_report_path(summary: dict[str, Any], fallback_dir: Path) -> str | None:
    report_path = summary.get("report_md")
    if report_path:
        return str(report_path)
    for name in (RUN_REPORT_FILENAME, LEGACY_RUN_REPORT_FILENAME):
        fallback = fallback_dir / name
        if fallback.exists():
            return str(fallback)
    return None


def _resolve_report_json_path(summary: dict[str, Any], fallback_dir: Path) -> str | None:
    report_json = summary.get("report_json")
    if report_json:
        return str(report_json)
    fallback = fallback_dir / RUN_REPORT_JSON_FILENAME
    if fallback.exists():
        return str(fallback)
    return None


def get_persistent_output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("persistent_output_dir") or cfg["output_dir"]).resolve()


def runtime_uses_temp_dir(cfg: dict[str, Any]) -> bool:
    return Path(cfg["output_dir"]).resolve() != get_persistent_output_dir(cfg)


def _copy_runtime_artifact(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    try:
        if src.resolve() == dst.resolve():
            return str(dst)
    except Exception:
        pass
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def cleanup_unused_outputs(
    cfg: dict[str, Any],
    semantic_prior_info: dict[str, Any],
    segmentation_info: dict[str, Any],
    report_path: Optional[str],
    report_json_path: Optional[str] = None,
) -> dict[str, Any]:
    removed: dict[str, Any] = {"removed_files": [], "removed_vector_datasets": []}
    if cfg.get("keep_debug_outputs", False):
        return removed

    if not cfg.get("keep_semantic_prior_artifacts", False):
        for key in ["m_sem_tif", "m_sem_png"]:
            path = semantic_prior_info.get(key)
            if path and remove_path(path):
                removed["removed_files"].append(path)
                semantic_prior_info[key] = None

        m_sem_shp = Path(cfg["output_dir"]) / "M_sem.shp"
        removed_vec = remove_vector_dataset(m_sem_shp)
        if removed_vec:
            removed["removed_vector_datasets"].append({"label": "M_sem", "paths": removed_vec})

    for key in ["y_inst_tif", "y_inst_color_png"]:
        path = segmentation_info.get(key)
        if path and remove_path(path):
            removed["removed_files"].append(path)
            segmentation_info[key] = None

    if report_path and remove_path(report_path):
        removed["removed_files"].append(report_path)
    if report_json_path and remove_path(report_json_path):
        removed["removed_files"].append(report_json_path)
    legacy_report = Path(cfg["metrics_json"]).resolve().parent / LEGACY_RUN_REPORT_FILENAME
    if not keep_legacy_output_aliases(cfg) and legacy_report.exists() and remove_path(legacy_report):
        removed["removed_files"].append(str(legacy_report))
    return removed


def materialize_public_output_aliases(
    cfg: dict[str, Any],
    semantic_prior_info: dict[str, Any],
    final_inst_shp: str,
    eval_info: dict[str, Any],
    summary_json: str,
    report_md: str,
) -> dict[str, Any]:
    eval_paths = get_eval_output_paths(cfg)
    aliases: dict[str, Any] = {}

    aliases["evaluation_metrics_json"] = copy_optional_file(
        eval_info.get("metrics_json"),
        eval_paths["evaluation_metrics_json"],
    )
    aliases["evaluation_details_csv"] = copy_optional_file(
        eval_info.get("details_csv"),
        eval_paths["evaluation_details_csv"],
    )
    eval_info["evaluation_metrics_json"] = aliases["evaluation_metrics_json"]
    eval_info["evaluation_details_csv"] = aliases["evaluation_details_csv"]
    if keep_legacy_output_aliases(cfg):
        aliases["summary_json"] = copy_optional_file(summary_json, Path(summary_json).resolve().parent / LEGACY_RUN_SUMMARY_FILENAME)
        aliases["report_md"] = copy_optional_file(report_md, Path(report_md).resolve().parent / LEGACY_RUN_REPORT_FILENAME)
    return aliases


def sync_runtime_artifacts_to_persistent_root(
    *,
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any],
) -> dict[str, Any]:
    runtime_root = Path(runtime_cfg["output_dir"]).resolve()
    persistent_root = get_persistent_output_dir(runtime_cfg)
    sync_info: dict[str, Any] = {
        "runtime_root": str(runtime_root),
        "persistent_root": str(persistent_root),
        "used_temp_runtime": runtime_uses_temp_dir(runtime_cfg),
        "copied": {},
    }
    if runtime_root == persistent_root:
        return sync_info

    keep_debug = bool(runtime_cfg.get("keep_debug_outputs", False))
    copied: dict[str, Any] = {}
    copy_map = [
        ("input_registry", runtime_root / "input_registry", persistent_root / "input_registry"),
        ("planning_scheduler", runtime_root / "planning_scheduler", persistent_root / "planning_scheduler"),
        ("data_processing_summaries", runtime_root / "data_processing" / "summaries", persistent_root / "data_processing" / "summaries"),
        ("metrics_json", Path(summary.get("metrics_json") or ""), persistent_root / "evaluation_metrics.json"),
        ("details_csv", Path(summary.get("details_csv") or ""), persistent_root / "evaluation_details.csv"),
        ("summary_json", Path(summary.get("summary_json") or ""), persistent_root / RUN_SUMMARY_FILENAME),
    ]
    if keep_debug:
        copy_map.append(("data_processing_requests", runtime_root / "data_processing" / "requests", persistent_root / "data_processing" / "requests"))
        copy_map.append(("roi_refinement", runtime_root / "roi_refinement", persistent_root / "roi_refinement"))

    for key, src, dst in copy_map:
        try:
            copied_path = _copy_runtime_artifact(src, dst)
        except Exception:
            copied_path = None
        if copied_path:
            copied[key] = copied_path

    sync_info["copied"] = copied
    return sync_info


def cleanup_temp_runtime_dir(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_root = Path(runtime_cfg["output_dir"]).resolve()
    cleanup_info = {
        "used_temp_runtime": runtime_uses_temp_dir(runtime_cfg),
        "temp_runtime_root": str(runtime_root),
        "removed": False,
        "skipped": None,
    }
    if not runtime_uses_temp_dir(runtime_cfg):
        cleanup_info["skipped"] = "runtime_equals_persistent"
        return cleanup_info
    if runtime_cfg.get("cleanup_policy") == "debug" or runtime_cfg.get("keep_debug_outputs", False):
        cleanup_info["skipped"] = "debug_policy"
        return cleanup_info
    if not bool(runtime_cfg.get("cleanup_temp_runtime", True)):
        cleanup_info["skipped"] = "cleanup_disabled"
        return cleanup_info
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
        cleanup_info["removed"] = True
    return cleanup_info


def finalize_run_outputs(
    *,
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any],
    input_manifest: dict[str, Any],
    publish_root: str | Path | None = None,
) -> dict[str, Any]:
    metrics_json = summary.get("metrics_json") or (summary.get("evaluation") or {}).get("metrics_json")
    details_csv = summary.get("details_csv") or (summary.get("evaluation") or {}).get("details_csv")
    summary_json = summary.get("summary_json")
    inst_shp = _resolve_inst_shp(summary)

    default_publish_root = get_persistent_output_dir(runtime_cfg) / "final_outputs"
    publish_dir = Path(publish_root) if publish_root is not None else default_publish_root
    report_path = _resolve_report_path(summary, Path(metrics_json).resolve().parent if metrics_json else publish_dir)
    report_json_path = _resolve_report_json_path(summary, Path(metrics_json).resolve().parent if metrics_json else publish_dir)

    deliverables = publish_segmentation_deliverables(
        inst_shp=inst_shp,
        publish_root=publish_dir,
        report_path=report_path,
        report_json_path=report_json_path,
        metrics_json=metrics_json,
        details_csv=details_csv,
        summary_json=summary_json,
        run_name=summary.get("run_name") or runtime_cfg.get("run_name"),
        background_raster=runtime_cfg.get("input_image"),
    )
    summary["input_manifest"] = input_manifest
    summary["final_outputs"] = FinalDeliverables(
        publish_root=str(publish_dir),
        tree_crowns_shp=deliverables.get("tree_crowns_shp"),
        tree_points_shp=deliverables.get("tree_points_shp"),
        segmentation_visualization_png=deliverables.get("segmentation_visualization_png"),
        final_evaluation_report_md=deliverables.get("final_evaluation_report_md"),
        final_evaluation_report_json=deliverables.get("final_evaluation_report_json"),
    ).to_dict()
    memory_info = record_execution(summary=summary, input_manifest=input_manifest)
    strategy_info = record_success_strategy(summary=summary)
    failure_info = record_failure_pattern(summary=summary)
    retrospective_info = record_run_retrospective(summary=summary)
    finetune_info = register_finetune_pool_assets(
        runtime_cfg=runtime_cfg,
        summary=summary,
        details_csv=details_csv,
        input_manifest=input_manifest,
    )
    finetune_plan = ((summary.get("planning_scheduler") or {}).get("finetune_training_plan") or {})
    dataset_bundle_info = export_finetune_dataset_bundle(
        summary=summary,
        runtime_cfg=runtime_cfg,
        finetune_plan=finetune_plan,
        output_path=get_persistent_output_dir(runtime_cfg) / "finetune" / "finetune_dataset_bundle.json",
    )
    finetune_plan["dataset_bundle_path"] = dataset_bundle_info.get("dataset_bundle_path")
    finetune_plan["dataset_selection_summary"] = dataset_bundle_info.get("selection_summary") or {}
    finetune_plan["supervision_mode"] = dataset_bundle_info.get("supervision_mode") or finetune_plan.get("supervision_mode") or "hybrid"
    finetune_plan["target_model_role"] = dataset_bundle_info.get("target_model_role")
    if dataset_bundle_info.get("failure_category"):
        finetune_plan["failure_category"] = dataset_bundle_info.get("failure_category")
    summary["memory_store"] = memory_info
    summary["strategy_memory"] = strategy_info
    summary["failure_memory"] = failure_info
    summary["retrospective_memory"] = retrospective_info
    summary["finetune_pool"] = finetune_info
    summary["finetune_dataset_bundle"] = dataset_bundle_info
    return summary
