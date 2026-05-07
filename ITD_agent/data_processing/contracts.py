from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RasterTilePlan:
    mode: str
    requires_geometry_crop: bool
    requires_sliding_window: bool
    reason: str
    tile_size: int | None = None
    overlap: int | None = None
    tile_overlap_ratio: float | None = None
    estimated_tile_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImagePriorProfile:
    source_id: str
    path: str
    width: int | None = None
    height: int | None = None
    crs: str | None = None
    resolution_x_m: float | None = None
    resolution_y_m: float | None = None
    area_ha: float | None = None
    band_count: int | None = None
    dtype: str | None = None
    quality_summary: dict[str, Any] = field(default_factory=dict)
    texture_summary: dict[str, Any] = field(default_factory=dict)
    tile_plan: RasterTilePlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.tile_plan:
            data["tile_plan"] = self.tile_plan.to_dict()
        return data


@dataclass
class DEMProcessingProfile:
    source_id: str
    path: str
    crs: str | None = None
    width: int | None = None
    height: int | None = None
    resolution_x_m: float | None = None
    resolution_y_m: float | None = None
    area_ha: float | None = None
    alignment_with_image: dict[str, Any] = field(default_factory=dict)
    terrain_products: dict[str, Any] = field(default_factory=dict)
    crop_strategy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeightRasterProfile:
    source_id: str
    path: str
    role: str
    crs: str | None = None
    width: int | None = None
    height: int | None = None
    resolution_x_m: float | None = None
    resolution_y_m: float | None = None
    area_ha: float | None = None
    alignment_with_image: dict[str, Any] = field(default_factory=dict)
    dom_cropped_path: str | None = None
    height_summary: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LogicalBlockPlanEntry:
    block_id: str
    dom_id: str
    block_index: int
    block_window: list[int] = field(default_factory=list)
    block_geo_bounds: list[float] = field(default_factory=list)
    width: int = 0
    height: int = 0
    edge_block_flag: bool = False
    overlap_with_neighbors_px: int = 0
    expected_tile_count: int = 0
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingBlockProfile:
    block_id: str
    dom_id: str
    block_index: int
    block_window: list[int] = field(default_factory=list)
    block_geo_bounds: list[float] = field(default_factory=list)
    width: int = 0
    height: int = 0
    edge_block_flag: bool = False
    overlap_with_neighbors_px: int = 0
    valid_pixel_ratio: float | None = None
    skip_block_candidate: bool = False
    low_valid_area_flag: bool = False
    brightness_mean: float | None = None
    brightness_std: float | None = None
    shadow_ratio_estimate: float | None = None
    overexposed_ratio: float | None = None
    underexposed_ratio: float | None = None
    laplacian_variance: float | None = None
    tenengrad: float | None = None
    blur_score: float | None = None
    stripe_noise_score: float | None = None
    stripe_noise_direction: str | None = None
    color_cast_score: float | None = None
    gradient_mean: float | None = None
    gradient_std: float | None = None
    texture_entropy: float | None = None
    texture_contrast: float | None = None
    texture_homogeneity: float | None = None
    texture_complexity_score: float | None = None
    low_texture_flag: bool = False
    dense_texture_flag: bool = False
    heterogeneity_coarse_grid: list[int] = field(default_factory=lambda: [7, 7])
    brightness_variance_across_cells: float | None = None
    shadow_spatial_variance: float | None = None
    gradient_variance_across_cells: float | None = None
    valid_ratio_variance_across_cells: float | None = None
    block_heterogeneity_score: float | None = None
    block_heterogeneity_level: str | None = None
    risk_tags: list[str] = field(default_factory=list)
    localized_risk_tags: list[str] = field(default_factory=list)
    quality_class: str | None = None
    priority_score: float | None = None
    expected_failure_modes: list[str] = field(default_factory=list)
    policy_template_name: str | None = None
    diam_list: str | None = None
    augment: bool = False
    iou_merge_thr: float | None = None
    enable_tile_fast_check: bool = False
    fusion_priority: str | None = None
    expert_model_candidates: list[str] = field(default_factory=list)
    memory_candidate_policy: str | None = None
    finetune_candidate_policy: str | None = None
    expected_tile_count: int = 0
    empty_tile_estimate: int = 0
    high_risk_tile_estimate: int = 0
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TileRunContext:
    tile_id: str
    dom_id: str
    block_id: str
    tile_index: int
    read_window: list[int] = field(default_factory=list)
    model_window: list[int] = field(default_factory=list)
    valid_write_window: list[int] = field(default_factory=list)
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0
    padding_ratio: float = 0.0
    edge_tile_flag: bool = False
    clip_to_valid_write_window: bool = True
    discard_padding_output: bool = True
    working_dom_path: str | None = None
    valid_mask_path: str | None = None
    crs: str | None = None
    transform_ref: str | None = None
    gsd_m: float | None = None
    gsd_status: str | None = None
    band_mapping: dict[str, int] = field(default_factory=dict)
    normalization_policy: str | None = None
    nodata_policy: str | None = None
    inherited_risk_tags: list[str] = field(default_factory=list)
    inherited_quality_class: str | None = None
    inherited_priority_score: float | None = None
    inherited_block_heterogeneity_level: str | None = None
    inherited_expected_failure_modes: list[str] = field(default_factory=list)
    inherited_diam_list: str | None = None
    inherited_augment: bool = False
    inherited_iou_merge_thr: float | None = None
    inherited_fusion_priority: str | None = None
    enable_tile_fast_check: bool = False
    valid_pixel_ratio: float | None = None
    empty_tile_flag: bool = False
    brightness_proxy: float | None = None
    shadow_proxy: float | None = None
    gradient_proxy: float | None = None
    local_texture_proxy: float | None = None
    tile_delta_detected: bool = False
    tile_delta_reason: list[str] = field(default_factory=list)
    tile_risk_tags: list[str] = field(default_factory=list)
    skip: bool = False
    skip_reason: str | None = None
    final_diam_list: str | None = None
    final_augment: bool = False
    final_iou_merge_thr: float | None = None
    final_bsize: int = 256
    final_fusion_priority: str | None = None
    expert_model_name: str | None = None
    export_sample_flag: bool = False
    memory_candidate_flag: bool = False
    finetune_candidate_flag: bool = False
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RemoteSensingPreflightSummary:
    dom_id: str
    working_dom_path: str | None = None
    valid_mask_path: str | None = None
    block_plan: list[LogicalBlockPlanEntry] = field(default_factory=list)
    block_profiles: list[ProcessingBlockProfile] = field(default_factory=list)
    tile_context_count: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dom_id": self.dom_id,
            "working_dom_path": self.working_dom_path,
            "valid_mask_path": self.valid_mask_path,
            "block_plan": [item.to_dict() for item in self.block_plan],
            "block_profiles": [item.to_dict() for item in self.block_profiles],
            "tile_context_count": self.tile_context_count,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass
class IntermediateArtifactRef:
    artifact_id: str
    artifact_type: str
    path: str
    producer: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingTaskRequest:
    request_id: str
    action: str
    source_type: str
    source_id: str | None = None
    target_path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusedSegmentationBundle:
    merged_instance_path: str | None = None
    deduped_instance_path: str | None = None
    tree_points_path: str | None = None
    removed_duplicates: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataProcessingSummary:
    image_profiles: list[ImagePriorProfile] = field(default_factory=list)
    dem_profiles: list[DEMProcessingProfile] = field(default_factory=list)
    height_raster_profiles: list[HeightRasterProfile] = field(default_factory=list)
    requested_tasks: list[ProcessingTaskRequest] = field(default_factory=list)
    intermediate_artifacts: list[IntermediateArtifactRef] = field(default_factory=list)
    fusion_bundle: FusedSegmentationBundle | None = None
    storage_layout: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_profiles": [item.to_dict() for item in self.image_profiles],
            "dem_profiles": [item.to_dict() for item in self.dem_profiles],
            "height_raster_profiles": [item.to_dict() for item in self.height_raster_profiles],
            "requested_tasks": [item.to_dict() for item in self.requested_tasks],
            "intermediate_artifacts": [item.to_dict() for item in self.intermediate_artifacts],
            "fusion_bundle": self.fusion_bundle.to_dict() if self.fusion_bundle else None,
            "storage_layout": self.storage_layout,
            "metadata": self.metadata,
        }
