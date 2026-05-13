from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile
from output_layer.contracts import FinalDeliverables, FinalTreeCrownResult
from output_layer.publisher import publish_final_tree_crown_outputs


def resolve_final_instance_shp(summary: dict[str, Any]) -> str | None:
    return summary.get("merged_inst_shp") or (summary.get("segmentation_model") or {}).get("y_inst_shp")


def _resolve_output_input_type(summary: dict[str, Any], runtime_cfg: dict[str, Any]) -> str:
    explicit = runtime_cfg.get("output_input_type") or runtime_cfg.get("input_type") or (summary.get("final_outputs_config") or {}).get("input_type")
    if explicit:
        return str(explicit)
    if summary.get("final_prediction_json") or summary.get("coco_predictions_json"):
        return "coco_gt"
    if summary.get("gt_metrics") or summary.get("metrics") or (summary.get("evaluation") or {}).get("metrics"):
        return "dom_with_gt"
    return "auto"


def _resolve_no_gt_quality_metrics(summary: dict[str, Any]) -> dict[str, Any] | None:
    final_eval = summary.get("final_evaluation") or {}
    online_quality = (final_eval.get("online_quality") or {}).get("metrics") or {}
    geometry_diag = online_quality.get("geometry_diagnostics") or {}
    geometry = online_quality.get("geometry_plausibility") or {}
    semantic_consistency = online_quality.get("semantic_instance_consistency") or {}
    merged = {
        **(summary.get("no_gt_quality_metrics") or {}),
        **geometry,
        **geometry_diag,
    }
    if semantic_consistency:
        merged.setdefault("semantic_instance_consistency", semantic_consistency.get("overlap_iou"))
        merged.setdefault("semantic_coverage_gap", semantic_consistency.get("semantic_gap"))
    return merged or None


def build_final_deliverables(
    *,
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any],
    publish_root: str | Path,
    report_path: str | None = None,
    report_json_path: str | None = None,
) -> dict[str, Any]:
    data_processing = summary.get("data_processing") or {}
    mainline_profile = resolve_mainline_profile(runtime_cfg)
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(mainline_profile)
    report_markdown = Path(report_path).read_text(encoding="utf-8") if report_path and Path(report_path).exists() else None
    report_json = None
    if report_json_path and Path(report_json_path).exists():
        try:
            import json

            report_json = json.loads(Path(report_json_path).read_text(encoding="utf-8"))
        except Exception:
            report_json = None
    final_result = FinalTreeCrownResult(
        run_id=str(summary.get("run_name") or runtime_cfg.get("run_name") or "unknown_run"),
        output_dir=str(publish_root),
        input_type=_resolve_output_input_type(summary, runtime_cfg),
        has_gt=bool(summary.get("gt_metrics") or summary.get("metrics") or (summary.get("evaluation") or {}).get("metrics")) or None,
        input_dom_path=runtime_cfg.get("input_image"),
        crown_vector_path=resolve_final_instance_shp(summary),
        coco_predictions_path=summary.get("final_prediction_json") or summary.get("coco_predictions_json"),
        semantic_mask_tif=data_processing.get("m_sem_tif"),
        semantic_mask_png=data_processing.get("m_sem_png"),
        instance_mask_tif=data_processing.get("instance_mask_tif") or data_processing.get("y_inst_tif"),
        instance_mask_png=data_processing.get("instance_mask_png") or data_processing.get("y_inst_png"),
        chm_raster=runtime_cfg.get("chm_tif") if bool(capabilities.get("output_height_structure")) else None,
        gt_metrics=summary.get("metrics") or (summary.get("evaluation") or {}).get("metrics"),
        gt_matches=summary.get("gt_matches") or (summary.get("evaluation") or {}).get("gt_matches") or [],
        geometry_metrics=(summary.get("final_evaluation") or {}).get("geometry_metrics") or summary.get("geometry_metrics"),
        no_gt_quality_metrics=_resolve_no_gt_quality_metrics(summary),
        visualizations=summary.get("visualizations") or (summary.get("final_evaluation") or {}).get("visualizations") or {},
        visualization_config=runtime_cfg.get("visualization") or runtime_cfg.get("output_visualization") or {},
        export_config=runtime_cfg.get("final_output_exports") or runtime_cfg.get("output_exports") or {},
        report_markdown=report_markdown,
        report_json=report_json,
        metadata={
            "mainline_profile": mainline_profile,
            "mainline_capabilities": capabilities,
            "metrics_json": summary.get("metrics_json") or (summary.get("evaluation") or {}).get("metrics_json"),
            "details_csv": summary.get("details_csv") or (summary.get("evaluation") or {}).get("details_csv"),
            "summary_json": summary.get("summary_json"),
            "source_adapter": "legacy_dom_summary",
        },
    )
    deliverables = publish_final_tree_crown_outputs(result=final_result, publish_root=publish_root)
    return FinalDeliverables(
        publish_root=str(publish_root),
        tree_crowns_shp=deliverables.get("tree_crowns_shp"),
        tree_points_shp=deliverables.get("tree_points_shp"),
        semantic_prior_tif=deliverables.get("semantic_prior_tif"),
        semantic_prior_png=deliverables.get("semantic_prior_png"),
        segmentation_visualization_png=deliverables.get("segmentation_visualization_png"),
        final_evaluation_report_md=deliverables.get("final_evaluation_report_md"),
        final_evaluation_report_json=deliverables.get("final_evaluation_report_json"),
        tree_crowns_height_structure_gpkg=deliverables.get("tree_crowns_height_structure_gpkg"),
        height_structure_summary_json=deliverables.get("height_structure_summary_json"),
        metadata={
            "mainline_profile": mainline_profile,
            "mainline_capabilities": capabilities,
            "height_structure_summary": deliverables.get("height_structure_summary") or {},
        },
    ).to_dict()
