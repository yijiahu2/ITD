from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely import wkt as shapely_wkt

from ITD_agent.planning.agent.config_builder import load_yaml, save_yaml
from ITD_agent.data_processing.roi.extractor import (
    clip_xiaoban_to_geometry_with_fields as dp_clip_xiaoban_to_geometry_with_fields,
    crop_roi_terrain_bundle as dp_crop_roi_terrain_bundle,
    make_bad_roi_gdf as dp_make_bad_roi_gdf,
    prepare_roi_refinement_inputs,
)
from ITD_agent.data_processing.inventory.normalizer import crop_raster_to_geometry as dp_crop_raster_to_geometry
from ITD_agent.segmentation.executor import execute_segmentation_model
from ITD_agent.data_processing.instance_ops import (
    dedupe_instances_by_overlap,
    filter_instances_to_ids_by_overlap,
    merge_split_instances_by_proximity,
    overlap_share_with_geom,
    suppress_small_boundary_fragments,
)
from ITD_agent.data_processing.terrain.dem_pipeline import generate_terrain_products
from tools.process_runner import run_streaming
from tools.runtime_cache_client import run_semantic_prior_task_via_worker

# =========================
# 基础工具
# =========================

DEFAULT_BASE_PARAMS = {
    "diam_list": "160,256,384",
    "tile": 2048,
    "overlap": 128,
    "tile_overlap": 0.35,
    "augment": True,
    "iou_merge_thr": 0.50,
    "bsize": 256,
}

SAFE_TILE = [1536, 1792, 2048, 2304]
SAFE_OVERLAP = [128, 192, 256, 384, 512]
SAFE_TILE_OVERLAP = [0.25, 0.30, 0.35, 0.40, 0.45]
SAFE_IOU = [0.18, 0.22, 0.24, 0.28, 0.30, 0.35, 0.40, 0.50]


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_parent(Path(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_str(v, default=None):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return str(v)


def _normalize_model_name(name: Any) -> str:
    return safe_str(name, "").strip().lower()


def _nearest_choice(value: float, choices: List[float]) -> float:
    return min(choices, key=lambda item: abs(float(item) - float(value)))


def _normalize_diam_list(value: Any) -> str:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        parts = []

    normalized: list[int] = []
    for part in parts:
        try:
            diameter = int(round(float(part)))
        except Exception:
            continue
        diameter = max(64, min(512, diameter))
        if diameter not in normalized:
            normalized.append(diameter)

    if len(normalized) < 2:
        return DEFAULT_BASE_PARAMS["diam_list"]
    return ",".join(str(item) for item in normalized)


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_BASE_PARAMS)
    if params:
        out.update(params)

    try:
        out["tile"] = int(_nearest_choice(float(out.get("tile", DEFAULT_BASE_PARAMS["tile"])), SAFE_TILE))
    except Exception:
        out["tile"] = DEFAULT_BASE_PARAMS["tile"]

    try:
        out["overlap"] = int(_nearest_choice(float(out.get("overlap", DEFAULT_BASE_PARAMS["overlap"])), SAFE_OVERLAP))
    except Exception:
        out["overlap"] = DEFAULT_BASE_PARAMS["overlap"]

    try:
        out["tile_overlap"] = float(_nearest_choice(float(out.get("tile_overlap", DEFAULT_BASE_PARAMS["tile_overlap"])), SAFE_TILE_OVERLAP))
    except Exception:
        out["tile_overlap"] = DEFAULT_BASE_PARAMS["tile_overlap"]

    try:
        out["iou_merge_thr"] = float(_nearest_choice(float(out.get("iou_merge_thr", DEFAULT_BASE_PARAMS["iou_merge_thr"])), SAFE_IOU))
    except Exception:
        out["iou_merge_thr"] = DEFAULT_BASE_PARAMS["iou_merge_thr"]

    out["diam_list"] = _normalize_diam_list(out.get("diam_list"))

    out["augment"] = bool(out.get("augment", True))

    # 关键运行约束：强制固定
    out["bsize"] = 256
    return out


def _resolve_preferred_child_runtime_overrides(
    preferred_child_model: str | None,
    child_plan_summary: dict[str, Any] | None,
) -> Dict[str, Any]:
    if not preferred_child_model or not isinstance(child_plan_summary, dict):
        return {}
    call_plan = child_plan_summary.get("child_model_call_plan") or {}
    profiles = call_plan.get("candidate_profiles") or []
    preferred_name = _normalize_model_name(preferred_child_model)
    for item in profiles:
        if not isinstance(item, dict):
            continue
        if _normalize_model_name(item.get("name")) != preferred_name:
            continue
        overrides = item.get("runtime_overrides")
        if isinstance(overrides, dict):
            return dict(overrides)
        return {}
    return {}


def _merge_preferred_child_base_params(
    base_params: Dict[str, Any],
    preferred_child_model: str | None,
    child_plan_summary: dict[str, Any] | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    overrides = _resolve_preferred_child_runtime_overrides(preferred_child_model, child_plan_summary)
    if not overrides:
        return sanitize_params(base_params), {}
    merged = dict(base_params)
    for key in ["diam_list", "tile", "overlap", "tile_overlap", "augment", "iou_merge_thr", "bsize"]:
        if key in overrides:
            merged[key] = overrides[key]
    return sanitize_params(merged), sanitize_params(overrides)


def copy_vector_dataset(src_shp: str, dst_shp: str):
    src = Path(src_shp)
    dst = Path(dst_shp)
    ensure_parent(dst)

    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        s = src.with_suffix(ext)
        if s.exists():
            shutil.copy2(s, dst.with_suffix(ext))


# =========================
# terrain 准备 / ROI terrain 裁剪
# =========================

def prepare_terrain_rasters(
    dem_tif: Optional[str],
    slope_tif: Optional[str],
    aspect_tif: Optional[str],
    work_dir: str,
) -> Dict[str, Any]:
    """
    若只提供 dem_tif，则自动生成 slope/aspect。
    """
    result = {
        "dem_tif": dem_tif,
        "slope_tif": slope_tif,
        "aspect_tif": aspect_tif,
        "landform_tif": None,
        "slope_position_tif": None,
        "terrain_generated": False,
    }

    if dem_tif is None:
        return result

    terrain_dir = Path(work_dir) / "terrain_cache"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    auto_slope = terrain_dir / f"{Path(dem_tif).stem}_slope.tif"
    auto_aspect = terrain_dir / f"{Path(dem_tif).stem}_aspect.tif"
    auto_landform = terrain_dir / f"{Path(dem_tif).stem}_landform.tif"
    auto_slope_position = terrain_dir / f"{Path(dem_tif).stem}_slope_position.tif"

    if slope_tif is not None and aspect_tif is not None and auto_landform.exists() and auto_slope_position.exists():
        result["landform_tif"] = str(auto_landform)
        result["slope_position_tif"] = str(auto_slope_position)
        return result

    generate_terrain_products(
        dem_tif=dem_tif,
        slope_tif=str(auto_slope),
        aspect_tif=str(auto_aspect),
        landform_tif=str(auto_landform),
        slope_position_tif=str(auto_slope_position),
        z_factor=1.0,
    )

    result["slope_tif"] = str(auto_slope)
    result["aspect_tif"] = str(auto_aspect)
    result["landform_tif"] = str(auto_landform)
    result["slope_position_tif"] = str(auto_slope_position)
    result["terrain_generated"] = True
    return result


def crop_raster_to_geometry(
    src_raster: str,
    geom_gdf: gpd.GeoDataFrame,
    out_raster: str,
    all_touched: bool = False,
):
    dp_crop_raster_to_geometry(src_raster, geom_gdf, out_raster, all_touched=all_touched)


def clip_vector_to_geometry(src_vector: str, geom_gdf: gpd.GeoDataFrame, out_vector: str):
    gdf = gpd.read_file(src_vector)
    if gdf.crs is None:
        raise ValueError(f"Vector has no CRS: {src_vector}")

    geom = geom_gdf.to_crs(gdf.crs)
    clipped = gpd.overlay(gdf, geom, how="intersection")
    clipped = clipped[clipped.geometry.notnull() & (~clipped.geometry.is_empty)].copy()
    if clipped.empty:
        raise ValueError(f"Clipped vector is empty: {src_vector}")

    out_path = Path(out_vector)
    ensure_parent(out_path)
    clipped.to_file(out_path)

def clip_xiaoban_to_geometry_with_fields(
    src_vector: str,
    geom_gdf: gpd.GeoDataFrame,
    out_vector: str,
    xiaoban_id_field: str,
    allowed_ids: Optional[List[str]] = None,
    tree_count_field: Optional[str] = None,
    crown_field: Optional[str] = None,
    closure_field: Optional[str] = None,
    area_ha_field: Optional[str] = None,
    density_field: Optional[str] = None,
):
    dp_clip_xiaoban_to_geometry_with_fields(
        src_vector=src_vector,
        geom_gdf=geom_gdf,
        out_vector=out_vector,
        xiaoban_id_field=xiaoban_id_field,
        allowed_ids=allowed_ids,
        tree_count_field=tree_count_field,
        crown_field=crown_field,
        closure_field=closure_field,
        area_ha_field=area_ha_field,
        density_field=density_field,
    )

def crop_roi_terrain_bundle(
    roi_geom_gdf: gpd.GeoDataFrame,
    roi_dir: str,
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    landform_tif: Optional[str] = None,
    slope_position_tif: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    return dp_crop_roi_terrain_bundle(
        roi_geom_gdf=roi_geom_gdf,
        roi_dir=roi_dir,
        dem_tif=dem_tif,
        slope_tif=slope_tif,
        aspect_tif=aspect_tif,
        landform_tif=landform_tif,
        slope_position_tif=slope_position_tif,
    )


# =========================
# bad xiaoban 选择
# =========================

def _build_error_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["tree_count_error_abs"].fillna(0) * 1.0
        + df["mean_crown_width_error_abs"].fillna(0) * 50.0
        + df["closure_error_abs"].fillna(0) * 100.0
    )


def _normalize_aspect_label(v: Any) -> Optional[str]:
    s = safe_str(v, None)
    if not s:
        return None
    s = s.strip().lower()
    mapping = {
        "n": "north",
        "ne": "northeast",
        "e": "east",
        "se": "southeast",
        "s": "south",
        "sw": "southwest",
        "w": "west",
        "nw": "northwest",
    }
    return mapping.get(s, s)


def _terrain_complexity_score_from_row(row: pd.Series) -> float:
    terrain = detect_terrain_profile(row)

    score = 1.0
    if terrain["slope_class"] == "steep":
        score += 0.55
    elif terrain["slope_class"] == "moderate":
        score += 0.20

    if terrain["relief_class"] == "high_relief":
        score += 0.30
    elif terrain["relief_class"] == "mid_relief":
        score += 0.10

    landform = safe_str(terrain.get("landform_type"), "").lower()
    if landform in {"mountain_middle", "mountain_low", "hill_high"}:
        score += 0.30
    elif landform in {"hill_middle"}:
        score += 0.15

    slope_position = safe_str(terrain.get("slope_position_class"), "").lower()
    if slope_position in {"ridge", "valley"}:
        score += 0.25

    aspect = _normalize_aspect_label(terrain.get("aspect_class"))
    if aspect in {"north", "northeast", "northwest"}:
        score += 0.10

    return score


def select_bad_xiaoban_rows(
    details_csv: str,
    tree_count_err_thr: float = 80.0,
    crown_err_thr: float = 0.40,
    closure_err_thr: float = 0.15,
    top_k: int = 3,
) -> pd.DataFrame:
    df = pd.read_csv(details_csv)

    if df.empty:
        return df

    required = [
        "xiaoban_id",
        "tree_count_error_abs",
        "mean_crown_width_error_abs",
        "closure_error_abs",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"details.csv missing required column: {col}")

    df["xiaoban_id"] = df["xiaoban_id"].astype(str)
    df["terrain_complexity_score"] = df.apply(_terrain_complexity_score_from_row, axis=1)

    cond = (
        (df["tree_count_error_abs"].fillna(0) >= tree_count_err_thr)
        | (df["mean_crown_width_error_abs"].fillna(0) >= crown_err_thr)
        | (df["closure_error_abs"].fillna(0) >= closure_err_thr)
    )
    complex_cond = (
        (df["terrain_complexity_score"] >= 1.7)
        & (
            (df["tree_count_error_abs"].fillna(0) >= tree_count_err_thr * 0.65)
            | (df["mean_crown_width_error_abs"].fillna(0) >= crown_err_thr * 0.75)
            | (df["closure_error_abs"].fillna(0) >= closure_err_thr * 0.75)
        )
    )

    bad = df[cond | complex_cond].copy()

    if bad.empty:
        tmp = df.copy()
        tmp["error_score"] = _build_error_score(tmp)
        tmp["priority_score"] = tmp["error_score"] * tmp["terrain_complexity_score"]
        bad = tmp.sort_values("priority_score", ascending=False).head(top_k)
    else:
        bad["error_score"] = _build_error_score(bad)
        bad["priority_score"] = bad["error_score"] * bad["terrain_complexity_score"]
        bad = bad.sort_values("priority_score", ascending=False).head(top_k)

    return bad.reset_index(drop=True)


# =========================
# ROI
# =========================

def make_bad_roi_gdf(
    xiaoban_shp: str,
    xiaoban_id_field: str,
    bad_ids: List[str],
    buffer_m: float = 5.0,
) -> gpd.GeoDataFrame:
    return dp_make_bad_roi_gdf(
        xiaoban_shp=xiaoban_shp,
        xiaoban_id_field=xiaoban_id_field,
        bad_ids=bad_ids,
        buffer_m=buffer_m,
    )


# =========================
# config 构建
# =========================

def build_local_refine_config(
    base_config_path: str,
    out_config_path: str,
    local_input_image: str,
    local_output_dir: str,
    local_xiaoban_shp: str | None,
    params: Dict[str, Any],
    run_name: str,
    local_dem_tif: str | None = None,
    local_slope_tif: str | None = None,
    local_aspect_tif: str | None = None,
    local_landform_tif: str | None = None,
    local_slope_position_tif: str | None = None,
) -> Dict[str, Any]:
    cfg = load_yaml(base_config_path)

    cfg["run_name"] = run_name
    cfg["input_image"] = local_input_image
    cfg["output_dir"] = local_output_dir
    if local_xiaoban_shp:
        cfg["xiaoban_shp"] = local_xiaoban_shp
    else:
        cfg.pop("xiaoban_shp", None)
    cfg["_grouped_dispatch_active"] = True
    cfg["disable_mlflow"] = True
    cfg["keep_semantic_prior_artifacts"] = False

    if local_dem_tif:
        cfg["dem_tif"] = local_dem_tif
    if local_slope_tif:
        cfg["slope_tif"] = local_slope_tif
    if local_aspect_tif:
        cfg["aspect_tif"] = local_aspect_tif
    if local_landform_tif:
        cfg["landform_tif"] = local_landform_tif
    if local_slope_position_tif:
        cfg["slope_position_tif"] = local_slope_position_tif

    # terrain 规范字段
    cfg["terrain_landform_field"] = "landform_type"
    cfg["terrain_slope_class_field"] = "slope_class"
    cfg["terrain_aspect_class_field"] = "aspect_class"
    cfg["terrain_slope_position_field"] = "slope_position_class"

    # 规则阈值
    cfg["flat_slope_threshold_deg"] = cfg.get("flat_slope_threshold_deg", 5.0)
    cfg["plain_relief_threshold_m"] = cfg.get("plain_relief_threshold_m", 30.0)

    cfg["metrics_json"] = str(Path(local_output_dir) / "metrics.json")
    cfg["details_csv"] = str(Path(local_output_dir) / "details.csv")

    params = sanitize_params(params)
    for k, v in params.items():
        cfg[k] = v

    save_yaml(cfg, out_config_path)
    return cfg

# =========================
# merge
# =========================

def merge_global_and_local_instances(
    global_inst_shp: str,
    local_inst_shp: str,
    xiaoban_shp: str | None,
    xiaoban_id_field: str,
    bad_ids: List[str] | None,
    out_merged_shp: str,
    roi_extent_vector: str | None = None,
):
    global_gdf = gpd.read_file(global_inst_shp)
    local_gdf = gpd.read_file(local_inst_shp)

    if global_gdf.crs is None or local_gdf.crs is None:
        raise ValueError("One of shapefiles has no CRS.")

    local_gdf = local_gdf.to_crs(global_gdf.crs)
    region_gdf = None
    if roi_extent_vector and Path(roi_extent_vector).exists():
        region_gdf = gpd.read_file(roi_extent_vector)
        if region_gdf.crs is None:
            raise ValueError(f"ROI extent vector has no CRS: {roi_extent_vector}")
        region_gdf = region_gdf.to_crs(global_gdf.crs)
    elif xiaoban_shp and Path(xiaoban_shp).exists() and bad_ids:
        xgdf = gpd.read_file(xiaoban_shp)
        if xgdf.crs is None:
            raise ValueError("One of shapefiles has no CRS.")
        xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
        bad = xgdf[xgdf[xiaoban_id_field].isin([str(x) for x in bad_ids])].copy()
        if bad.empty:
            raise ValueError("No bad xiaoban found for merge.")
        region_gdf = bad.to_crs(global_gdf.crs)
    else:
        raise ValueError("Either roi_extent_vector or xiaoban_shp+bad_ids is required for merge.")

    region_union = unary_union(region_gdf.geometry.tolist())
    if xiaoban_shp and Path(str(xiaoban_shp)).exists() and bad_ids:
        xgdf = gpd.read_file(xiaoban_shp).to_crs(global_gdf.crs)
        xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
        local_filtered = filter_instances_to_ids_by_overlap(
            inst_gdf=local_gdf,
            polygon_gdf=xgdf,
            id_field=xiaoban_id_field,
            allowed_ids=bad_ids,
        )
    else:
        local_overlap_ratio = local_gdf.geometry.apply(lambda geom: overlap_share_with_geom(geom, region_union))
        local_centroid_in = local_gdf.geometry.centroid.within(region_union)
        local_filtered = local_gdf[(local_overlap_ratio > 0.20) | local_centroid_in].copy()
    local_filtered = suppress_small_boundary_fragments(
        local_filtered,
        region_gdf,
        boundary_band_m=1.5,
        min_area_m2=6.0,
    )
    local_filtered = merge_split_instances_by_proximity(
        local_filtered,
        boundary_gdf=region_gdf,
        boundary_band_m=1.5,
        merge_gap_m=0.9,
        centroid_distance_factor=1.4,
        max_centroid_distance_m=7.0,
    )

    global_overlap_ratio = global_gdf.geometry.apply(lambda geom: overlap_share_with_geom(geom, region_union))
    global_centroid_in_bad = global_gdf.geometry.centroid.within(region_union)
    global_keep = global_gdf[~((global_overlap_ratio >= 0.5) | global_centroid_in_bad)].copy()

    merged = pd.concat([global_keep, local_filtered], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=global_gdf.crs)
    merged = merge_split_instances_by_proximity(
        merged,
        boundary_gdf=region_gdf,
        boundary_band_m=1.5,
        merge_gap_m=0.9,
        centroid_distance_factor=1.4,
        max_centroid_distance_m=7.0,
    )
    merged = dedupe_instances_by_overlap(merged, overlap_ratio_thr=0.5)

    out_path = Path(out_merged_shp)
    ensure_parent(out_path)
    merged.to_file(out_path)


# =========================
# merged 评测 + 前后对比
# =========================
def evaluate_merged_result(
    base_cfg: Dict[str, Any],
    local_root: Path,
    merged_shp: str,
    bad_ids: List[str],
    group_plan: List[Dict[str, Any]],
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
) -> Dict[str, Any]:
    merged_metrics_json = str(local_root / "merged_metrics.json")
    merged_details_csv = str(local_root / "merged_details.csv")

    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_reference_quality",
        "--inst_shp", merged_shp,
        "--patch_raster", base_cfg["input_image"],
        "--xiaoban_shp", base_cfg["xiaoban_shp"],
        "--out_json", merged_metrics_json,
        "--out_csv", merged_details_csv,
        "--id_field", base_cfg["xiaoban_id_field"],
        "--tree_count_field", base_cfg["tree_count_field"],
        "--crown_field", base_cfg["crown_field"],
        "--closure_field", base_cfg["closure_field"],
        "--area_ha_field", base_cfg["area_ha_field"],
        "--flat_slope_threshold_deg", str(base_cfg.get("flat_slope_threshold_deg", 5.0)),
        "--plain_relief_threshold_m", str(base_cfg.get("plain_relief_threshold_m", 30.0)),
    ]

    if base_cfg.get("density_field"):
        cmd.extend(["--density_field", str(base_cfg["density_field"])])

    if dem_tif:
        cmd.extend(["--dem_tif", dem_tif])
    if slope_tif:
        cmd.extend(["--slope_tif", slope_tif])
    if aspect_tif:
        cmd.extend(["--aspect_tif", aspect_tif])

    res = run_streaming(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"merged evaluation failed:\n{res.stdout}")

    compare_json = str(local_root / "refine_compare_summary.json")
    with open(base_cfg["metrics_json"], "r", encoding="utf-8") as f:
        before_metrics = json.load(f)
    with open(merged_metrics_json, "r", encoding="utf-8") as f:
        after_metrics = json.load(f)

    before_details = pd.read_csv(base_cfg["details_csv"])
    after_details = pd.read_csv(merged_details_csv)

    if "xiaoban_id" in before_details.columns:
        before_details["xiaoban_id"] = before_details["xiaoban_id"].astype(str)
    if "xiaoban_id" in after_details.columns:
        after_details["xiaoban_id"] = after_details["xiaoban_id"].astype(str)

    metric_keys = [
        "tree_count_error_ratio",
        "mean_crown_width_error_ratio",
        "closure_error_abs",
        "density_error_abs",
    ]

    compare = {
        "base_metrics_json": base_cfg["metrics_json"],
        "merged_metrics_json": merged_metrics_json,
        "base_details_csv": base_cfg["details_csv"],
        "merged_details_csv": merged_details_csv,
        "bad_xiaoban_ids": [str(x) for x in bad_ids],
        "global_before_after": {},
        "bad_xiaoban_before_after": [],
        "group_plan": group_plan,
        "terrain_rule_config": {
            "flat_slope_threshold_deg": base_cfg.get("flat_slope_threshold_deg", 5.0),
            "plain_relief_threshold_m": base_cfg.get("plain_relief_threshold_m", 30.0),
        },
    }

    for key in metric_keys:
        before_v = before_metrics.get(key)
        after_v = after_metrics.get(key)
        compare["global_before_after"][key] = {
            "before": before_v,
            "after": after_v,
            "delta": (after_v - before_v) if (before_v is not None and after_v is not None) else None,
        }

    if (
        bad_ids
        and "xiaoban_id" in before_details.columns
        and "xiaoban_id" in after_details.columns
    ):
        merged_bad = before_details.merge(
            after_details,
            on="xiaoban_id",
            suffixes=("_before", "_after"),
        )
        merged_bad = merged_bad[merged_bad["xiaoban_id"].isin([str(x) for x in bad_ids])].copy()

        for _, row in merged_bad.iterrows():
            compare["bad_xiaoban_before_after"].append(
                {
                    "xiaoban_id": row["xiaoban_id"],
                    "tree_count_error_abs_before": row.get("tree_count_error_abs_before"),
                    "tree_count_error_abs_after": row.get("tree_count_error_abs_after"),
                    "mean_crown_width_error_abs_before": row.get("mean_crown_width_error_abs_before"),
                    "mean_crown_width_error_abs_after": row.get("mean_crown_width_error_abs_after"),
                    "closure_error_abs_before": row.get("closure_error_abs_before"),
                    "closure_error_abs_after": row.get("closure_error_abs_after"),
                }
            )

    Path(compare_json).write_text(json.dumps(compare, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "merged_metrics_json": merged_metrics_json,
        "merged_details_csv": merged_details_csv,
        "compare_json": compare_json,
        "bad_ids": bad_ids,
        "group_plan": group_plan,
        "terrain_rule_config": {
            "flat_slope_threshold_deg": base_cfg.get("flat_slope_threshold_deg", 5.0),
            "plain_relief_threshold_m": base_cfg.get("plain_relief_threshold_m", 30.0),
        },
    }

# =========================
# 局部参数选择策略增强版
# =========================

def detect_error_profile(row: pd.Series) -> Dict[str, Any]:
    tree_err = float(row.get("tree_count_error_abs", 0) or 0)
    crown_err = float(row.get("mean_crown_width_error_abs", 0) or 0)
    closure_err = float(row.get("closure_error_abs", 0) or 0)
    density_err = float(row.get("density_error_abs", 0) or 0)

    score_count = tree_err * 1.0
    score_crown = crown_err * 50.0
    score_closure = closure_err * 100.0
    score_density = density_err / 100.0

    dominant = max(
        [
            ("count", score_count),
            ("crown", score_crown),
            ("closure", score_closure),
            ("density", score_density),
        ],
        key=lambda x: x[1],
    )[0]

    pred_tree_count = row.get("pred_tree_count", None)
    expected_tree_count = row.get("expected_tree_count", None)
    pred_cover_ratio = row.get("pred_cover_ratio", None)
    expected_closure = row.get("expected_closure", None)
    pred_density = row.get("pred_density_trees_per_ha", None)
    expected_density = row.get("expected_density", None)

    count_direction = "unknown"
    if pd.notna(pred_tree_count) and pd.notna(expected_tree_count):
        count_direction = "under" if float(pred_tree_count) < float(expected_tree_count) else "over"

    cover_direction = "unknown"
    if pd.notna(pred_cover_ratio) and pd.notna(expected_closure):
        cover_direction = "low" if float(pred_cover_ratio) < float(expected_closure) else "high"

    density_direction = "unknown"
    if pd.notna(pred_density) and pd.notna(expected_density):
        density_direction = "low" if float(pred_density) < float(expected_density) else "high"

    return {
        "dominant_error": dominant,
        "count_direction": count_direction,
        "cover_direction": cover_direction,
        "density_direction": density_direction,
        "score_count": score_count,
        "score_crown": score_crown,
        "score_closure": score_closure,
        "score_density": score_density,
    }


def detect_terrain_profile(row: pd.Series) -> Dict[str, Any]:
    mean_slope = safe_float(row.get("mean_slope"), None)
    relief_elev = safe_float(row.get("relief_elev"), None)
    dominant_aspect = safe_str(row.get("dominant_aspect_class"), None)
    landform_type = safe_str(row.get("landform_type"), None)
    raw_slope_class = safe_str(row.get("slope_class"), None)
    aspect_class = safe_str(row.get("aspect_class"), dominant_aspect)
    slope_position_class = safe_str(row.get("slope_position_class"), None)

    if raw_slope_class:
        if raw_slope_class.startswith(("IV", "V", "VI")):
            slope_class = "steep"
        elif raw_slope_class.startswith("III"):
            slope_class = "moderate"
        elif raw_slope_class.startswith(("I", "II")):
            slope_class = "gentle"
        else:
            slope_class = "unknown"
    elif mean_slope is None:
        slope_class = "unknown"
    elif mean_slope >= 25:
        slope_class = "steep"
    elif mean_slope >= 12:
        slope_class = "moderate"
    else:
        slope_class = "gentle"

    if relief_elev is None:
        relief_class = "unknown"
    elif relief_elev >= 20:
        relief_class = "high_relief"
    elif relief_elev >= 8:
        relief_class = "mid_relief"
    else:
        relief_class = "low_relief"

    return {
        "mean_slope": mean_slope,
        "relief_elev": relief_elev,
        "dominant_aspect_class": dominant_aspect,
        "landform_type": landform_type,
        "slope_class": slope_class,
        "terrain_slope_class": raw_slope_class,
        "aspect_class": aspect_class,
        "slope_position_class": slope_position_class,
        "relief_class": relief_class,
    }


def apply_terrain_adjustments(
    *,
    params: Dict[str, Any],
    dominant_error: str,
    count_direction: str,
    cover_direction: str,
    terrain_profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    params = dict(params)
    notes: List[str] = []

    slope_class = terrain_profile.get("slope_class")
    relief_class = terrain_profile.get("relief_class")
    landform_type = safe_str(terrain_profile.get("landform_type"), "").lower()
    slope_position = safe_str(terrain_profile.get("slope_position_class"), "").lower()
    aspect_class = _normalize_aspect_label(terrain_profile.get("aspect_class"))

    is_complex_landform = landform_type in {"mountain_middle", "mountain_low", "hill_high", "hill_middle"}
    is_ridge_valley = slope_position in {"ridge", "valley"}
    is_shaded = aspect_class in {"north", "northeast", "northwest"}
    is_steep_complex = slope_class == "steep" or relief_class == "high_relief" or is_complex_landform

    if is_steep_complex:
        params["overlap"] = max(int(params.get("overlap", 128)), 192)
        params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40)
        notes.append("steep_or_high_relief_overlap_boost")

    if is_ridge_valley:
        params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40)
        params["overlap"] = max(int(params.get("overlap", 128)), 192)
        notes.append("ridge_valley_context_boost")

    if is_shaded and dominant_error in {"count", "closure", "crown"}:
        params["augment"] = True
        params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40)
        notes.append("shaded_slope_recall_boost")

    if dominant_error in {"count", "density"} and count_direction == "under":
        if is_ridge_valley or is_shaded:
            params["augment"] = True
            params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40)
            params["iou_merge_thr"] = max(float(params.get("iou_merge_thr", 0.28)), 0.35)
            notes.append("terrain_underseg_recall_boost")

    if dominant_error == "closure" and cover_direction == "low":
        if is_complex_landform or is_ridge_valley:
            params["augment"] = True
            params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40)
            notes.append("terrain_low_closure_recovery")

    if dominant_error in {"count", "density"} and count_direction == "over":
        if is_steep_complex:
            params["iou_merge_thr"] = max(float(params.get("iou_merge_thr", 0.24)), 0.40)
            params["tile_overlap"] = min(float(params.get("tile_overlap", 0.35)), 0.35)
            notes.append("steep_overseg_merge_bias")

    return sanitize_params(params), notes


def choose_local_params_for_one_xiaoban(
    row: pd.Series,
    base_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    规则驱动增强版：
    - 先看误差主导类型
    - 再用 terrain 做微调
    - 坡度优先用于决策；坡向当前主要用于解释和分组元数据保留
    """
    base = sanitize_params(base_params)
    err_profile = detect_error_profile(row)
    terrain_profile = detect_terrain_profile(row)

    strategy = "balanced"
    params = dict(base)

    dominant = err_profile["dominant_error"]
    count_direction = err_profile["count_direction"]
    cover_direction = err_profile["cover_direction"]
    density_direction = err_profile["density_direction"]

    slope_class = terrain_profile["slope_class"]
    landform_type = terrain_profile.get("landform_type")
    aspect_class = terrain_profile.get("aspect_class")
    slope_position_class = terrain_profile.get("slope_position_class")

    # 1) 数量不足：偏补漏检，但以 planner/base 参数为主，只做温和修正
    if dominant in ("count", "density") and (count_direction == "under" or density_direction == "low"):
        strategy = f"{slope_class}_count_under" if slope_class in {"steep", "moderate", "gentle"} else "count_under"
        params["augment"] = True
        params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40 if slope_class == "steep" else 0.35)
        params["overlap"] = max(int(params.get("overlap", 128)), 192)
        params["iou_merge_thr"] = max(float(params.get("iou_merge_thr", 0.35)), 0.35)

        if aspect_class in {"north", "northeast", "northwest", "flat_no_aspect"}:
            params["augment"] = True
        if slope_position_class in {"ridge", "valley"}:
            params["tile_overlap"] = max(float(params["tile_overlap"]), 0.40)

    # 2) 数量过多：偏抑制过分裂/过检，但不直接推回过保守模板
    elif dominant in ("count", "density") and (count_direction == "over" or density_direction == "high"):
        strategy = "steep_count_over" if slope_class == "steep" else "count_over"
        params["tile_overlap"] = min(float(params.get("tile_overlap", 0.35)), 0.35)
        params["iou_merge_thr"] = max(float(params.get("iou_merge_thr", 0.28)), 0.40 if slope_class == "steep" else 0.35)

    # 3) 冠幅主导：优先保留 planner 给出的多尺度窗口，只略增上下文
    elif dominant == "crown":
        strategy = "steep_crown_focus" if slope_class == "steep" else "crown_focus"
        params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.35 if slope_class != "steep" else 0.40)
        params["overlap"] = max(int(params.get("overlap", 128)), 192)
        params["iou_merge_thr"] = max(float(params.get("iou_merge_thr", 0.28)), 0.35)

    # 4) 覆盖/郁闭主导：优先覆盖恢复
    elif dominant == "closure":
        if cover_direction == "low":
            strategy = "steep_closure_low" if slope_class == "steep" else "closure_low"
            params["augment"] = True
            params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.40)
            params["overlap"] = max(int(params.get("overlap", 128)), 192)
        else:
            strategy = "closure_high"
            params["tile_overlap"] = min(float(params.get("tile_overlap", 0.35)), 0.35)
            params["iou_merge_thr"] = max(float(params.get("iou_merge_thr", 0.28)), 0.35)

    # 5) 默认折中
    else:
        strategy = "steep_balanced" if slope_class == "steep" else "balanced"
        params["augment"] = bool(params.get("augment", True))
        params["tile_overlap"] = max(float(params.get("tile_overlap", 0.35)), 0.35 if slope_class != "steep" else 0.40)

    if landform_type in {"mountain_middle", "mountain_low", "hill_high"}:
        params["iou_merge_thr"] = max(float(params["iou_merge_thr"]), 0.35)

    params, terrain_adjustments = apply_terrain_adjustments(
        params=params,
        dominant_error=dominant,
        count_direction=count_direction,
        cover_direction=cover_direction,
        terrain_profile=terrain_profile,
    )

    profile = {
        "error_profile": err_profile,
        "terrain_profile": terrain_profile,
        "terrain_complexity_score": _terrain_complexity_score_from_row(row),
        "terrain_adjustments": terrain_adjustments,
    }
    return strategy, params, profile


def build_group_plan(
    bad_df: pd.DataFrame,
    base_params: Dict[str, Any],
    strategy_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """
    strategy_mode:
    - single_params: 所有 bad xiaoban 共用一组 params
    - auto: 每个 xiaoban 先判型，再按 strategy+params 分组
    """
    base_params = sanitize_params(base_params)

    if bad_df.empty:
        return []

    if strategy_mode == "single_params":
        return [{
            "strategy": "single_params",
            "params": base_params,
            "xiaoban_ids": bad_df["xiaoban_id"].astype(str).tolist(),
            "members": bad_df[["xiaoban_id"]].to_dict(orient="records"),
        }]

    plan_map = {}
    for _, row in bad_df.iterrows():
        strategy, params, profile = choose_local_params_for_one_xiaoban(row, base_params)
        key = (
            strategy,
            params["diam_list"],
            params["tile"],
            params["overlap"],
            params["tile_overlap"],
            params["augment"],
            params["iou_merge_thr"],
            params["bsize"],
        )

        if key not in plan_map:
            plan_map[key] = {
                "strategy": strategy,
                "params": params,
                "xiaoban_ids": [],
                "members": [],
            }

        xid = str(row["xiaoban_id"])
        plan_map[key]["xiaoban_ids"].append(xid)

        member = row.to_dict()
        member["xiaoban_id"] = xid
        member["profile"] = profile
        plan_map[key]["members"].append(member)

    groups = list(plan_map.values())
    groups.sort(key=lambda g: len(g["xiaoban_ids"]), reverse=True)
    return groups


def _choose_local_params_for_signal_candidate(
    candidate: Dict[str, Any],
    base_params: Dict[str, Any],
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    params = sanitize_params(base_params)
    terrain_score = safe_float(candidate.get("terrain_score_mean"), 0.0) or 0.0
    texture_score = safe_float(candidate.get("texture_score_mean"), 0.0) or 0.0
    shadow_score = safe_float(candidate.get("shadow_score_mean"), 0.0) or 0.0
    boundary_score = safe_float(candidate.get("boundary_score_mean"), 0.0) or 0.0
    canopy_fraction = safe_float(candidate.get("canopy_fraction"), 0.0) or 0.0

    strategy = "balanced"
    if terrain_score >= 0.60 and max(texture_score, boundary_score) >= 0.50:
        strategy = "steep_crown_focus"
        params.update({
            "overlap": max(int(params.get("overlap", 128)), 192),
            "tile_overlap": max(float(params.get("tile_overlap", 0.35)), 0.40),
            "iou_merge_thr": max(float(params.get("iou_merge_thr", 0.28)), 0.35),
        })
    elif shadow_score >= 0.60:
        strategy = "closure_low"
        params.update({
            "augment": True,
            "overlap": max(int(params.get("overlap", 128)), 192),
            "tile_overlap": max(float(params.get("tile_overlap", 0.35)), 0.40),
        })
    elif max(texture_score, boundary_score) >= 0.55:
        strategy = "crown_focus"
        params.update({
            "overlap": max(int(params.get("overlap", 128)), 192),
            "tile_overlap": max(float(params.get("tile_overlap", 0.35)), 0.35),
            "iou_merge_thr": max(float(params.get("iou_merge_thr", 0.28)), 0.35),
        })
    elif canopy_fraction >= 0.75:
        strategy = "dense_balanced"
        params.update({
            "augment": True,
            "tile_overlap": max(float(params.get("tile_overlap", 0.35)), 0.35),
        })

    profile = {
        "roi_signal_profile": {
            "terrain_score_mean": terrain_score,
            "texture_score_mean": texture_score,
            "shadow_score_mean": shadow_score,
            "boundary_score_mean": boundary_score,
            "canopy_fraction": canopy_fraction,
        }
    }
    return strategy, sanitize_params(params), profile


def build_group_plan_from_roi_candidates(
    roi_candidates: List[Dict[str, Any]],
    base_params: Dict[str, Any],
    details_csv: str | None = None,
) -> List[Dict[str, Any]]:
    details_df = None
    if details_csv and Path(details_csv).exists():
        details_df = pd.read_csv(details_csv)
        if "xiaoban_id" in details_df.columns:
            details_df["xiaoban_id"] = details_df["xiaoban_id"].astype(str)

    def _roi_candidate_priority(candidate: Dict[str, Any]) -> float:
        return (
            float(candidate.get("score") or 0.0)
            + 0.08 * float(candidate.get("prior_overlap_ratio") or 0.0)
            + 0.05 * float(candidate.get("boundary_score_mean") or 0.0)
            + 0.03 * float(candidate.get("terrain_score_mean") or 0.0)
        )

    def _should_merge_same_xiaoban_groups(existing: Dict[str, Any], current: Dict[str, Any]) -> bool:
        existing_ids = sorted(set(existing.get("prior_xiaoban_ids") or existing.get("xiaoban_ids") or []))
        current_ids = sorted(set(current.get("prior_xiaoban_ids") or current.get("xiaoban_ids") or []))
        if not existing_ids or existing_ids != current_ids:
            return False

        existing_wkt = str(existing.get("roi_geometry_wkt") or "").strip()
        current_wkt = str(current.get("roi_geometry_wkt") or "").strip()
        if not existing_wkt or not current_wkt:
            return False
        if existing_wkt == current_wkt:
            return True

        try:
            existing_geom = shapely_wkt.loads(existing_wkt)
            current_geom = shapely_wkt.loads(current_wkt)
        except Exception:
            return False
        if existing_geom.is_empty or current_geom.is_empty:
            return False

        inter_area = float(existing_geom.intersection(current_geom).area)
        if inter_area <= 0:
            return False
        existing_overlap = inter_area / max(float(existing_geom.area), 1.0e-6)
        current_overlap = inter_area / max(float(current_geom.area), 1.0e-6)
        union_area = float(existing_geom.union(current_geom).area)
        iou = inter_area / max(union_area, 1.0e-6)
        return max(existing_overlap, current_overlap) >= 0.75 or iou >= 0.60

    groups: List[Dict[str, Any]] = []
    for candidate in roi_candidates:
        prior_ids = [str(x) for x in (candidate.get("prior_xiaoban_ids") or [])]
        strategy = None
        params = None
        profile = {}
        if details_df is not None and prior_ids and "xiaoban_id" in details_df.columns:
            matched = details_df[details_df["xiaoban_id"].isin(prior_ids)].copy()
            if not matched.empty:
                strategy, params, profile = choose_local_params_for_one_xiaoban(matched.iloc[0], base_params)
        if strategy is None or params is None:
            strategy, params, profile = _choose_local_params_for_signal_candidate(candidate, base_params)

        candidate_id = str(candidate.get("candidate_id") or f"signal_roi_{len(groups)+1:02d}")
        groups.append(
            {
                "strategy": strategy,
                "params": params,
                "xiaoban_ids": prior_ids or [candidate_id],
                "prior_xiaoban_ids": prior_ids,
                "roi_geometry_wkt": candidate.get("geometry_wkt"),
                "roi_geometry_crs": candidate.get("geometry_crs"),
                "roi_candidate": candidate,
                "members": [{"candidate_id": candidate_id, "profile": profile, **candidate}],
            }
        )

    merged_groups: List[Dict[str, Any]] = []
    for group in groups:
        existing_idx = next((idx for idx, item in enumerate(merged_groups) if _should_merge_same_xiaoban_groups(item, group)), None)
        if existing_idx is None:
            merged_groups.append(group)
            continue

        existing = merged_groups[existing_idx]
        existing_priority = _roi_candidate_priority(existing.get("roi_candidate") or {})
        current_priority = _roi_candidate_priority(group.get("roi_candidate") or {})
        if current_priority > existing_priority:
            preferred = group
            secondary = existing
        else:
            preferred = existing
            secondary = group

        merged_members = list(preferred.get("members") or [])
        seen_ids = {
            str(item.get("candidate_id") or "")
            for item in merged_members
            if isinstance(item, dict)
        }
        for item in secondary.get("members") or []:
            candidate_id = str((item or {}).get("candidate_id") or "")
            if candidate_id and candidate_id not in seen_ids:
                merged_members.append(item)
                seen_ids.add(candidate_id)

        merged_group = dict(preferred)
        merged_group["members"] = merged_members
        merged_group["xiaoban_ids"] = sorted(set(preferred.get("xiaoban_ids") or secondary.get("xiaoban_ids") or []))
        merged_group["prior_xiaoban_ids"] = sorted(set(preferred.get("prior_xiaoban_ids") or secondary.get("prior_xiaoban_ids") or []))
        merged_groups[existing_idx] = merged_group

    deduped_groups = list(merged_groups)
    deduped_groups.sort(key=lambda item: len(item.get("xiaoban_ids") or []), reverse=True)
    return deduped_groups


# =========================
# 单个 group 执行
# =========================

def run_one_group_refinement(
    base_config_path: str,
    base_cfg: Dict[str, Any],
    current_global_shp: str,
    group_idx: int,
    group: Dict[str, Any],
    xiaoban_id_field: str,
    buffer_m: float,
    local_root: Path,
    terrain_info: Dict[str, Any],
    preferred_child_model: str | None = None,
    child_plan_summary: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    xiaoban_ids = [str(x) for x in group["xiaoban_ids"]]
    prior_xiaoban_ids = [str(x) for x in (group.get("prior_xiaoban_ids") or [])]
    strategy = group["strategy"]
    params = sanitize_params(group["params"])

    group_name = f"group_{group_idx:02d}_{strategy}_{'_'.join(xiaoban_ids)}"
    group_root = local_root / group_name
    ensure_dir(group_root)

    roi_inputs = prepare_roi_refinement_inputs(
        base_cfg=base_cfg,
        xiaoban_ids=prior_xiaoban_ids or xiaoban_ids,
        buffer_m=buffer_m,
        group_name=group_name,
        terrain_info=terrain_info,
        roi_geometry_wkt=group.get("roi_geometry_wkt"),
        roi_geometry_crs=group.get("roi_geometry_crs"),
        roi_metadata=group.get("roi_candidate"),
    )

    local_image = str(roi_inputs["roi_image_tif"])
    local_xiaoban = str(roi_inputs["roi_xiaoban_gpkg"]) if roi_inputs.get("roi_xiaoban_gpkg") else None
    local_config = str(group_root / "planning_scheduler" / "runtime" / f"{group_name}.yaml")
    local_output_dir = str(group_root / "seg_output")

    local_cfg = build_local_refine_config(
        base_config_path=base_config_path,
        out_config_path=local_config,
        local_input_image=local_image,
        local_output_dir=local_output_dir,
        local_xiaoban_shp=local_xiaoban,
        params=params,
        run_name=group_name,
        local_dem_tif=roi_inputs.get("roi_dem_tif"),
        local_slope_tif=roi_inputs.get("roi_slope_tif"),
        local_aspect_tif=roi_inputs.get("roi_aspect_tif"),
        local_landform_tif=roi_inputs.get("roi_landform_tif"),
        local_slope_position_tif=roi_inputs.get("roi_slope_position_tif"),
    )

    try:
        semantic_prior_info = run_semantic_prior_task_via_worker(local_cfg)
        segmentation_info = execute_segmentation_model(
            cfg=local_cfg,
            m_sem_tif=semantic_prior_info["m_sem_tif"],
            phase="roi_child_inference",
            model_role="child_model",
            preferred_model=preferred_child_model,
            plan_summary=child_plan_summary or {},
        )
    except Exception as e:
        raise RuntimeError(f"Local refine failed [{group_name}]:\n{e}")

    local_inst_shp = segmentation_info["y_inst_shp"]
    if not Path(local_inst_shp).exists():
        raise FileNotFoundError(f"Local Y_inst.shp not found: {local_inst_shp}")

    merged_shp = str(group_root / "merged_after_group.shp")
    merge_global_and_local_instances(
        global_inst_shp=current_global_shp,
        local_inst_shp=local_inst_shp,
        xiaoban_shp=base_cfg.get("xiaoban_shp"),
        xiaoban_id_field=xiaoban_id_field,
        bad_ids=prior_xiaoban_ids or None,
        out_merged_shp=merged_shp,
        roi_extent_vector=roi_inputs.get("roi_extent_gpkg"),
    )

    group_summary = {
        "group_name": group_name,
        "group_root": str(group_root),
        "strategy": strategy,
        "params": params,
        "xiaoban_ids": xiaoban_ids,
        "prior_xiaoban_ids": prior_xiaoban_ids,
        "local_config": local_config,
        "local_output_dir": local_output_dir,
        "local_metrics_json": local_cfg["metrics_json"],
        "local_details_csv": local_cfg["details_csv"],
        "local_inst_shp": local_inst_shp,
        "merged_after_group_shp": merged_shp,
        "roi_image_tif": local_image,
        "roi_xiaoban_shp": local_xiaoban,
        "roi_dem_tif": roi_inputs.get("roi_dem_tif"),
        "roi_slope_tif": roi_inputs.get("roi_slope_tif"),
        "roi_aspect_tif": roi_inputs.get("roi_aspect_tif"),
        "roi_landform_tif": roi_inputs.get("roi_landform_tif"),
        "roi_slope_position_tif": roi_inputs.get("roi_slope_position_tif"),
        "roi_extraction_summary_json": roi_inputs.get("summary_json"),
        "roi_extent_gpkg": roi_inputs.get("roi_extent_gpkg"),
        "roi_metadata": roi_inputs.get("roi_metadata"),
        "roi_cache_root": roi_inputs.get("roi_cache_root"),
        "preferred_child_model": preferred_child_model,
        "child_plan_summary": child_plan_summary or {},
        "members": group.get("members", []),
    }

    save_json(group_summary, str(group_root / "group_summary.json"))
    return group_summary

# =========================
# 主流程
# =========================

def run_local_refinement(
    base_config_path: str,
    global_details_csv: str,
    global_inst_shp: str,
    best_params: Dict[str, Any],
    xiaoban_id_field: str = "XBH",
    top_k: int = 2,
    buffer_m: float = 5.0,
    strategy_mode: str = "auto",
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    local_refine_root: Optional[str] = None,
    preferred_child_model: str | None = None,
    child_plan_summary: dict[str, Any] | None = None,
    roi_candidates: Optional[List[Dict[str, Any]]] = None,
):
    base_cfg = load_yaml(base_config_path)
    base_params = sanitize_params(best_params)
    base_params, preferred_child_runtime_overrides = _merge_preferred_child_base_params(
        base_params=base_params,
        preferred_child_model=preferred_child_model,
        child_plan_summary=child_plan_summary,
    )

    if roi_candidates:
        bad_ids = [str(item.get("candidate_id") or f"signal_roi_{idx+1:02d}") for idx, item in enumerate(roi_candidates)]
        print(f"[local_refine] signal_roi_candidates = {bad_ids}")
        group_plan = build_group_plan_from_roi_candidates(
            roi_candidates=roi_candidates[:top_k],
            base_params=base_params,
            details_csv=global_details_csv,
        )
    else:
        bad_df = select_bad_xiaoban_rows(global_details_csv, top_k=top_k)
        if bad_df.empty:
            raise ValueError("No xiaoban rows selected from details.csv")
        bad_ids = bad_df["xiaoban_id"].astype(str).tolist()
        print(f"[local_refine] bad_xiaoban_ids = {bad_ids}")
        group_plan = build_group_plan(
            bad_df=bad_df,
            base_params=base_params,
            strategy_mode=strategy_mode,
        )

    print("[local_refine] group_plan:")
    for i, g in enumerate(group_plan, 1):
        print(f"  - group {i}: strategy={g['strategy']}, xiaoban_ids={g['xiaoban_ids']}, params={g['params']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "local_refine_signal_" if roi_candidates else "local_refine_"
    refine_name = prefix + "_".join([str(x) for x in bad_ids]) + f"_{stamp}"
    root_dir = local_refine_root or os.getenv(
        "FOREST_AGENT_LOCAL_REFINE_ROOT",
        "/home/xth/forest_agent_project/outputs/local_refine",
    )
    local_root = Path(root_dir) / refine_name
    ensure_dir(local_root)

    terrain_info = prepare_terrain_rasters(
        dem_tif=dem_tif,
        slope_tif=slope_tif,
        aspect_tif=aspect_tif,
        work_dir=str(local_root),
    )

    save_json(
        {
            "refine_name": refine_name,
            "base_config_path": base_config_path,
            "global_details_csv": global_details_csv,
            "global_inst_shp": global_inst_shp,
            "strategy_mode": strategy_mode,
            "base_params": base_params,
            "preferred_child_model": preferred_child_model,
            "preferred_child_runtime_overrides": preferred_child_runtime_overrides,
            "bad_xiaoban_ids": bad_ids,
            "roi_candidates": roi_candidates or [],
            "group_plan": group_plan,
            "terrain_inputs": terrain_info,
        },
        str(local_root / "refine_plan.json"),
    )

    current_global_shp = global_inst_shp
    group_summaries = []

    for idx, group in enumerate(group_plan, 1):
        group_summary = run_one_group_refinement(
            base_config_path=base_config_path,
            base_cfg=base_cfg,
            current_global_shp=current_global_shp,
            group_idx=idx,
            group=group,
            xiaoban_id_field=xiaoban_id_field,
            buffer_m=buffer_m,
            local_root=local_root,
            terrain_info=terrain_info,
            preferred_child_model=preferred_child_model,
            child_plan_summary=child_plan_summary,
        )
        group_summaries.append(group_summary)
        current_global_shp = group_summary["merged_after_group_shp"]

    final_merged_shp = str(local_root / "merged_global_local_Y_inst.shp")
    copy_vector_dataset(current_global_shp, final_merged_shp)

    merged_eval = evaluate_merged_result(
        base_cfg=base_cfg,
        local_root=local_root,
        merged_shp=final_merged_shp,
        bad_ids=[item for item in bad_ids if not str(item).startswith("signal_roi_")],
        group_plan=[
            {
                "strategy": gs["strategy"],
                "xiaoban_ids": gs["xiaoban_ids"],
                "prior_xiaoban_ids": gs.get("prior_xiaoban_ids") or [],
                "params": gs["params"],
                "group_name": gs["group_name"],
                "roi_extent_gpkg": gs.get("roi_extent_gpkg"),
                "roi_dem_tif": gs.get("roi_dem_tif"),
                "roi_slope_tif": gs.get("roi_slope_tif"),
                "roi_aspect_tif": gs.get("roi_aspect_tif"),
                "roi_landform_tif": gs.get("roi_landform_tif"),
                "roi_slope_position_tif": gs.get("roi_slope_position_tif"),
            }
            for gs in group_summaries
        ],
        dem_tif=terrain_info.get("dem_tif"),
        slope_tif=terrain_info.get("slope_tif"),
        aspect_tif=terrain_info.get("aspect_tif"),
    )

    summary = {
        "refine_name": refine_name,
        "strategy_mode": strategy_mode,
        "bad_xiaoban_ids": bad_ids,
        "roi_candidates": roi_candidates or [],
        "base_params": base_params,
        "preferred_child_model": preferred_child_model,
        "preferred_child_runtime_overrides": preferred_child_runtime_overrides,
        "terrain_inputs": terrain_info,
        "group_summaries": group_summaries,
        "merged_shp": final_merged_shp,
        "merged_metrics_json": merged_eval["merged_metrics_json"],
        "merged_details_csv": merged_eval["merged_details_csv"],
        "compare_json": merged_eval["compare_json"],
    }

    summary_path = local_root / "local_refine_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[local_refine] summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--global_details_csv", required=True)
    parser.add_argument("--global_inst_shp", required=True)
    parser.add_argument(
        "--best_params_json",
        required=False,
        default='{"diam_list":"96,192,320","tile":1536,"overlap":512,"tile_overlap":0.35,"augment":true,"iou_merge_thr":0.28,"bsize":256}',
        help='base params json, e.g. \'{"diam_list":"96,192,320","tile":1536,...}\'',
    )
    parser.add_argument("--xiaoban_id_field", default="XBH")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--buffer_m", type=float, default=5.0)
    parser.add_argument(
        "--strategy_mode",
        default="auto",
        choices=["auto", "single_params"],
        help="auto: 按局部误差类型 + terrain 自动分组选参; single_params: 所有 bad xiaoban 共用一组参数",
    )
    parser.add_argument("--dem_tif", default=None, help="Global DEM tif for terrain-aware ROI crop.")
    parser.add_argument("--slope_tif", default=None, help="Optional precomputed slope tif.")
    parser.add_argument("--aspect_tif", default=None, help="Optional precomputed aspect tif.")
    args = parser.parse_args()

    best_params = json.loads(args.best_params_json)

    run_local_refinement(
        base_config_path=args.base_config,
        global_details_csv=args.global_details_csv,
        global_inst_shp=args.global_inst_shp,
        best_params=best_params,
        xiaoban_id_field=args.xiaoban_id_field,
        top_k=args.top_k,
        buffer_m=args.buffer_m,
        strategy_mode=args.strategy_mode,
        dem_tif=args.dem_tif,
        slope_tif=args.slope_tif,
        aspect_tif=args.aspect_tif,
    )


if __name__ == "__main__":
    main()
