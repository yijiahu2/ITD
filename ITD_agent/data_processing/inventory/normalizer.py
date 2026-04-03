from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import IndustryVectorProfile, SurveyTableProfile
from ITD_agent.data_processing.inventory.crown_metrics import (
    equivalent_crown_width,
    inventory_mean_crown_width_from_geometry,
    safe_float,
    standardize_inventory_crown_width,
)
from ITD_agent.data_processing.inventory.spatial_context import (
    aspect_stats_for_geom,
    build_bounds_gdf,
    crop_raster_to_geometry,
    enrich_xiaoban_clip_fields,
    load_dom_bounds,
    prepare_spatial_context,
    raster_stats_for_geom,
    summarize_xiaoban_terrain_classes,
)


STANDARD_FIELD_KEYS = [
    "xiaoban_id",
    "tree_count",
    "crown_width",
    "closure",
    "density",
    "area_ha",
]


def _read_table_columns(path: str, sheet_name: str | None = None) -> tuple[list[str], int | None]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        return [], None
    return [str(col) for col in df.columns], int(len(df))


def _recognized_fields(mapping: dict[str, str], available_columns: list[str]) -> dict[str, str]:
    return {
        std_key: raw_key
        for std_key, raw_key in mapping.items()
        if std_key in STANDARD_FIELD_KEYS and raw_key in available_columns
    }


def build_survey_table_profiles(manifest: InputManifest) -> list[SurveyTableProfile]:
    profiles: list[SurveyTableProfile] = []
    for item in manifest.survey_tables:
        path = Path(item.path)
        if not path.exists():
            profiles.append(SurveyTableProfile(source_id=item.id, path=item.path, metadata={"status": "missing"}))
            continue
        columns, row_count = _read_table_columns(item.path, sheet_name=item.sheet_name)
        profiles.append(
            SurveyTableProfile(
                source_id=item.id,
                path=item.path,
                columns=columns,
                key_fields=item.key_fields,
                field_mapping=item.field_mapping,
                recognized_fields=_recognized_fields(item.field_mapping, columns),
                row_count=row_count,
                metadata={"format": item.format, "sheet_name": item.sheet_name},
            )
        )
    return profiles


def build_industry_vector_profiles(manifest: InputManifest) -> list[IndustryVectorProfile]:
    profiles: list[IndustryVectorProfile] = []
    for item in manifest.industry_vectors:
        path = Path(item.path)
        if not path.exists():
            profiles.append(IndustryVectorProfile(source_id=item.id, path=item.path, metadata={"status": "missing"}))
            continue
        gdf = gpd.read_file(path, rows=10)
        try:
            import fiona

            feature_count = int(len(fiona.open(path)))
        except Exception:
            feature_count = int(len(gdf))
        bounds = list(gdf.total_bounds) if not gdf.empty else None
        geom_type = None
        if not gdf.empty:
            try:
                geom_type = str(gdf.geom_type.mode().iloc[0])
            except Exception:
                geom_type = item.geometry_type
        profiles.append(
            IndustryVectorProfile(
                source_id=item.id,
                path=item.path,
                geometry_type=geom_type or item.geometry_type,
                crs=str(gdf.crs) if gdf.crs else item.crs,
                feature_count=feature_count,
                columns=[str(col) for col in gdf.columns],
                key_fields=item.key_fields,
                field_mapping=item.field_mapping,
                recognized_fields=_recognized_fields(item.field_mapping, [str(col) for col in gdf.columns]),
                extent_summary={
                    "bounds": bounds,
                },
            )
        )
    return profiles


__all__ = [
    "aspect_stats_for_geom",
    "build_bounds_gdf",
    "build_industry_vector_profiles",
    "build_survey_table_profiles",
    "crop_raster_to_geometry",
    "enrich_xiaoban_clip_fields",
    "equivalent_crown_width",
    "inventory_mean_crown_width_from_geometry",
    "load_dom_bounds",
    "prepare_spatial_context",
    "raster_stats_for_geom",
    "safe_float",
    "standardize_inventory_crown_width",
    "summarize_xiaoban_terrain_classes",
]
