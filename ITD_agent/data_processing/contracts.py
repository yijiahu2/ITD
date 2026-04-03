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
class SurveyTableProfile:
    source_id: str
    path: str
    columns: list[str] = field(default_factory=list)
    key_fields: list[str] = field(default_factory=list)
    field_mapping: dict[str, str] = field(default_factory=dict)
    recognized_fields: dict[str, str] = field(default_factory=dict)
    row_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IndustryVectorProfile:
    source_id: str
    path: str
    geometry_type: str | None = None
    crs: str | None = None
    feature_count: int | None = None
    columns: list[str] = field(default_factory=list)
    key_fields: list[str] = field(default_factory=list)
    field_mapping: dict[str, str] = field(default_factory=dict)
    recognized_fields: dict[str, str] = field(default_factory=dict)
    extent_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeProfile:
    source_id: str
    path: str
    raw_type: str
    normalized_type: str
    use_scope: str
    spatial_scale: str | None = None
    tags: list[str] = field(default_factory=list)
    extraction_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicDatasetProfile:
    source_id: str
    dataset_format: str
    root_path: str | None = None
    annotation_path: str | None = None
    usage_roles: list[str] = field(default_factory=list)
    annotation_type: str | None = None
    finetune_ready: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    survey_table_profiles: list[SurveyTableProfile] = field(default_factory=list)
    industry_vector_profiles: list[IndustryVectorProfile] = field(default_factory=list)
    knowledge_profiles: list[KnowledgeProfile] = field(default_factory=list)
    public_dataset_profiles: list[PublicDatasetProfile] = field(default_factory=list)
    requested_tasks: list[ProcessingTaskRequest] = field(default_factory=list)
    intermediate_artifacts: list[IntermediateArtifactRef] = field(default_factory=list)
    fusion_bundle: FusedSegmentationBundle | None = None
    storage_layout: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_profiles": [item.to_dict() for item in self.image_profiles],
            "dem_profiles": [item.to_dict() for item in self.dem_profiles],
            "survey_table_profiles": [item.to_dict() for item in self.survey_table_profiles],
            "industry_vector_profiles": [item.to_dict() for item in self.industry_vector_profiles],
            "knowledge_profiles": [item.to_dict() for item in self.knowledge_profiles],
            "public_dataset_profiles": [item.to_dict() for item in self.public_dataset_profiles],
            "requested_tasks": [item.to_dict() for item in self.requested_tasks],
            "intermediate_artifacts": [item.to_dict() for item in self.intermediate_artifacts],
            "fusion_bundle": self.fusion_bundle.to_dict() if self.fusion_bundle else None,
            "storage_layout": self.storage_layout,
            "metadata": self.metadata,
        }
