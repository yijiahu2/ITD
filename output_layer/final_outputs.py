from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile
from output_layer.contracts import FinalDeliverables
from output_layer.publisher import publish_segmentation_deliverables


def resolve_final_instance_shp(summary: dict[str, Any]) -> str | None:
    return summary.get("merged_inst_shp") or (summary.get("segmentation_model") or {}).get("y_inst_shp")


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
    deliverables = publish_segmentation_deliverables(
        inst_shp=resolve_final_instance_shp(summary),
        publish_root=publish_root,
        semantic_prior_tif=data_processing.get("m_sem_tif"),
        semantic_prior_png=data_processing.get("m_sem_png"),
        report_path=report_path,
        report_json_path=report_json_path,
        metrics_json=summary.get("metrics_json") or (summary.get("evaluation") or {}).get("metrics_json"),
        details_csv=summary.get("details_csv") or (summary.get("evaluation") or {}).get("details_csv"),
        summary_json=summary.get("summary_json"),
        run_name=summary.get("run_name") or runtime_cfg.get("run_name"),
        background_raster=runtime_cfg.get("input_image"),
        mainline_profile=mainline_profile,
        chm_raster=runtime_cfg.get("chm_tif"),
        enable_height_structure=bool(capabilities.get("output_height_structure")),
    )
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
