from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RemoteSensingImageSource:
    id: str
    path: str
    sensor: str | None = None
    resolution_m: float | None = None
    crs: str | None = None
    bands: list[str] = field(default_factory=list)
    nodata: int | float | str | None = None
    acquired_at: str | None = None
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomInputContract:
    dom_id: str
    source_path: str
    working_dom_path: str
    input_type: str
    mainline_profile: str
    width: int
    height: int
    pixel_count: int
    bounds: list[float]
    crs: str | None
    transform: list[float] | None
    working_to_original_transform: dict[str, Any] | None
    gsd_x_m: float | None
    gsd_y_m: float | None
    recommended_gsd_m: float
    acceptable_gsd_range_m: list[float]
    gsd_status: str
    band_count: int
    dtype: str
    band_mapping: dict[str, int]
    normalization_policy: str
    nodata: int | float | str | None
    nodata_policy: str
    valid_mask_path: str
    global_valid_pixel_ratio_estimate: float | None
    processing_block_px: int
    processing_block_stride_px: int
    processing_block_overlap_px: int
    processing_edge_absorb_px: int
    processing_block_min_preferred_px: int
    processing_block_max_preferred_px: int
    tile_px: int
    tile_overlap_px: int
    tile_stride_px: int
    allow_elastic_model_input: bool
    pad_if_smaller_than_model_input: bool
    snap_last_tile_to_edge: bool
    discard_padding_output: bool
    bsize: int
    processing_mode: str
    estimated_block_count: int
    estimated_tile_count: int
    output_clip_policy: str
    warnings: list[str] = field(default_factory=list)
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DEMSource:
    id: str
    path: str
    resolution_m: float | None = None
    crs: str | None = None
    vertical_unit: str | None = None
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeightRasterSource:
    id: str
    path: str
    role: str
    resolution_m: float | None = None
    crs: str | None = None
    vertical_unit: str | None = None
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurveyTableSource:
    id: str
    path: str
    format: str | None = None
    sheet_name: str | None = None
    key_fields: list[str] = field(default_factory=list)
    field_mapping: dict[str, str] = field(default_factory=dict)
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IndustryVectorSource:
    id: str
    path: str
    geometry_type: str | None = None
    crs: str | None = None
    key_fields: list[str] = field(default_factory=list)
    field_mapping: dict[str, str] = field(default_factory=dict)
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainKnowledgeItem:
    id: str
    type: str
    path: str
    title: str | None = None
    sheet_name: str | None = None
    tags: list[str] = field(default_factory=list)
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicDatasetSource:
    id: str
    format: str
    path: str | None = None
    root: str | None = None
    image_root: str | None = None
    annotation_path: str | None = None
    schema_mapping: dict[str, str] = field(default_factory=dict)
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DatasetSource = PublicDatasetSource


@dataclass
class ValidationIssue:
    level: str
    code: str
    source_type: str
    source_id: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    status: str = "ok"
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.issues if item.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.issues if item.level == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass
class PreparedAsset:
    source_type: str
    source_id: str
    raw_path: str | None = None
    prepared_path: str | None = None
    registry_key: str | None = None
    preparation_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedInputIndex:
    registry_root: str
    prepared_root: str
    assets: list[PreparedAsset] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_root": self.registry_root,
            "prepared_root": self.prepared_root,
            "assets": [item.to_dict() for item in self.assets],
            "metadata": self.metadata,
        }


@dataclass
class InputManifest:
    config_path: str | None = None
    remote_sensing: list[RemoteSensingImageSource] = field(default_factory=list)
    dom_input_contract: DomInputContract | None = None
    terrain_dem: list[DEMSource] = field(default_factory=list)
    canopy_height: list[HeightRasterSource] = field(default_factory=list)
    surface_models: list[HeightRasterSource] = field(default_factory=list)
    survey_tables: list[SurveyTableSource] = field(default_factory=list)
    industry_vectors: list[IndustryVectorSource] = field(default_factory=list)
    domain_knowledge_items: list[DomainKnowledgeItem] = field(default_factory=list)
    public_datasets: list[PublicDatasetSource] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: ValidationReport | None = None
    preparation: PreparedInputIndex | None = None

    @property
    def remote_sensing_images(self) -> list[str]:
        return [item.path for item in self.remote_sensing if item.path]

    @property
    def dem_paths(self) -> list[str]:
        return [item.path for item in self.terrain_dem if item.path]

    @property
    def chm_paths(self) -> list[str]:
        return [item.path for item in self.canopy_height if item.path]

    @property
    def dsm_paths(self) -> list[str]:
        return [item.path for item in self.surface_models if item.path]

    @property
    def survey_vector(self) -> str | None:
        for item in self.industry_vectors:
            if item.path:
                return item.path
        return None

    @property
    def reference_vector(self) -> str | None:
        return self.survey_vector

    @property
    def inventory_vector(self) -> str | None:
        return self.survey_vector

    @property
    def survey_table_paths(self) -> list[str]:
        return [item.path for item in self.survey_tables if item.path]

    @property
    def domain_knowledge(self) -> list[str]:
        return [item.path for item in self.domain_knowledge_items if item.path]

    @property
    def input_modalities(self) -> dict[str, bool]:
        return {
            "image": bool(self.remote_sensing),
            "dem": bool(self.terrain_dem),
            "chm": bool(self.canopy_height),
            "dsm": bool(self.surface_models),
            "inventory": bool(self.survey_tables or self.industry_vectors),
            "knowledge": bool(self.domain_knowledge_items),
            "public_datasets": bool(self.public_datasets),
        }

    def to_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata["input_modalities"] = self.input_modalities
        return {
            "config_path": self.config_path,
            "remote_sensing": [item.to_dict() for item in self.remote_sensing],
            "dom_input_contract": self.dom_input_contract.to_dict() if self.dom_input_contract else None,
            "terrain_dem": [item.to_dict() for item in self.terrain_dem],
            "canopy_height": [item.to_dict() for item in self.canopy_height],
            "surface_models": [item.to_dict() for item in self.surface_models],
            "survey_tables": [item.to_dict() for item in self.survey_tables],
            "industry_vectors": [item.to_dict() for item in self.industry_vectors],
            "domain_knowledge_items": [item.to_dict() for item in self.domain_knowledge_items],
            "public_datasets": [item.to_dict() for item in self.public_datasets],
            "metadata": metadata,
            "validation": self.validation.to_dict() if self.validation else None,
            "preparation": self.preparation.to_dict() if self.preparation else None,
            "remote_sensing_images": self.remote_sensing_images,
            "dem_paths": self.dem_paths,
            "chm_paths": self.chm_paths,
            "dsm_paths": self.dsm_paths,
            "survey_vector": self.survey_vector,
            "reference_vector": self.reference_vector,
            "inventory_vector": self.inventory_vector,
            "survey_tables_paths": self.survey_table_paths,
            "domain_knowledge": self.domain_knowledge,
        }
