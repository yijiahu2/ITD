from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FinalTreeCrownResult:
    run_id: str
    output_dir: str
    input_type: str = "auto"
    has_gt: bool | None = None
    input_dom_path: str | None = None
    crown_vector_path: str | None = None
    instances: list[dict[str, Any]] = field(default_factory=list)
    coco_predictions_path: str | None = None
    semantic_mask_tif: str | None = None
    semantic_mask_png: str | None = None
    instance_mask_tif: str | None = None
    instance_mask_png: str | None = None
    instance_mask_paths: list[str] = field(default_factory=list)
    chm_raster: str | None = None
    coordinate_mode: str = "unknown"
    image_width: int | None = None
    image_height: int | None = None
    categories: list[dict[str, Any]] = field(default_factory=list)
    dom_spatial_reference: dict[str, Any] = field(default_factory=dict)
    gt_metrics: dict[str, Any] | None = None
    gt_matches: list[dict[str, Any]] = field(default_factory=list)
    geometry_metrics: dict[str, Any] | None = None
    no_gt_quality_metrics: dict[str, Any] | None = None
    visualizations: dict[str, str] = field(default_factory=dict)
    visualization_config: dict[str, Any] = field(default_factory=dict)
    export_config: dict[str, Any] = field(default_factory=dict)
    trajectory_paths: list[str] = field(default_factory=list)
    report_markdown: str | None = None
    report_json: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalDeliverables:
    publish_root: str
    tree_crowns_shp: str | None = None
    tree_points_shp: str | None = None
    semantic_prior_tif: str | None = None
    semantic_prior_png: str | None = None
    segmentation_visualization_png: str | None = None
    final_evaluation_report_md: str | None = None
    final_evaluation_report_json: str | None = None
    tree_crowns_height_structure_gpkg: str | None = None
    height_structure_summary_json: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
