from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.data_processing.terrain.dem_pipeline import generate_terrain_products


EVAL_METRICS_FILENAME = "evaluation_metrics.json"
EVAL_DETAILS_FILENAME = "evaluation_details.csv"


def get_stage_output_paths(cfg: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(cfg["output_dir"])
    return {
        "m_sem_tif": str(output_dir / "M_sem.tif"),
        "m_sem_png": str(output_dir / "M_sem.png"),
        "y_inst_tif": str(output_dir / "Y_inst.tif"),
        "y_inst_shp": str(output_dir / "Y_inst.shp"),
        "y_inst_color_png": str(output_dir / "Y_inst_color.png"),
        "semantic_prior_tif": str(output_dir / "semantic_prior.tif"),
        "semantic_prior_png": str(output_dir / "semantic_prior.png"),
        "tree_crowns_shp": str(output_dir / "tree_crowns.shp"),
        "tree_points_shp": str(output_dir / "tree_points.shp"),
        "tree_crowns_preview_png": str(output_dir / "tree_crowns_preview.png"),
    }


def get_eval_output_paths(cfg: dict[str, Any]) -> dict[str, str]:
    parent = Path(cfg["metrics_json"]).resolve().parent
    return {
        "metrics_json": cfg["metrics_json"],
        "details_csv": cfg["details_csv"],
        "evaluation_metrics_json": str(parent / EVAL_METRICS_FILENAME),
        "evaluation_details_csv": str(parent / EVAL_DETAILS_FILENAME),
    }


def prepare_terrain_inputs_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    dem_tif = cfg.get("dem_tif")
    slope_tif = cfg.get("slope_tif")
    aspect_tif = cfg.get("aspect_tif")

    result = {
        "dem_tif": dem_tif,
        "slope_tif": slope_tif,
        "aspect_tif": aspect_tif,
        "landform_tif": cfg.get("landform_tif"),
        "slope_position_tif": cfg.get("slope_position_tif"),
        "terrain_generated": False,
    }
    if not dem_tif:
        return result

    metrics_json = cfg.get("metrics_json")
    if metrics_json:
        terrain_dir = Path(metrics_json).resolve().parent / "terrain_cache"
    else:
        terrain_dir = Path(cfg["output_dir"]).resolve() / "terrain_cache"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    auto_slope = terrain_dir / f"{Path(dem_tif).stem}_slope.tif"
    auto_aspect = terrain_dir / f"{Path(dem_tif).stem}_aspect.tif"
    auto_landform = terrain_dir / f"{Path(dem_tif).stem}_landform.tif"
    auto_slope_position = terrain_dir / f"{Path(dem_tif).stem}_slope_position.tif"

    if auto_slope.exists() and auto_aspect.exists() and auto_landform.exists() and auto_slope_position.exists():
        result["slope_tif"] = str(auto_slope)
        result["aspect_tif"] = str(auto_aspect)
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


def validate_runtime_cfg(cfg: dict[str, Any]) -> None:
    cfg["flat_slope_threshold_deg"] = cfg.get("flat_slope_threshold_deg", 5.0)
    cfg["plain_relief_threshold_m"] = cfg.get("plain_relief_threshold_m", 30.0)
    cfg["terrain_landform_field"] = cfg.get("terrain_landform_field", "landform_type")
    cfg["terrain_slope_class_field"] = cfg.get("terrain_slope_class_field", "slope_class")
    cfg["terrain_aspect_class_field"] = cfg.get("terrain_aspect_class_field", "aspect_class")
    cfg["terrain_slope_position_field"] = cfg.get("terrain_slope_position_field", "slope_position_class")

    required_keys = [
        "input_image",
        "output_dir",
        "metrics_json",
        "details_csv",
        "xiaoban_shp",
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "area_ha_field",
        "semantic_prior_script",
        "segmentation_script",
        "conda_sh",
        "conda_env",
        "work_dir",
        "diam_list",
        "tile",
        "overlap",
        "tile_overlap",
        "bsize",
        "augment",
        "iou_merge_thr",
    ]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

    if int(cfg["bsize"]) != 256:
        raise ValueError(f"Unsafe bsize={cfg['bsize']}. Must be fixed to 256.")


def collect_run_metadata(cfg: dict[str, Any], terrain_info: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "experiment_name",
        "run_name",
        "compartment_id",
        "patch_id",
        "forest_type",
        "agent_version",
        "input_image",
        "output_dir",
        "xiaoban_shp",
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "density_field",
        "area_ha_field",
        "semantic_prior_script",
        "segmentation_script",
        "diam_list",
        "tile",
        "overlap",
        "tile_overlap",
        "bsize",
        "augment",
        "iou_merge_thr",
        "semantic_prior_ckpt",
        "semantic_prior_extra_args",
        "segmentation_algorithm",
    ]
    meta = {k: cfg.get(k) for k in keys if k in cfg}
    meta["terrain_info"] = terrain_info
    meta["spatial_context_object_json"] = cfg.get("spatial_context_object_json")
    meta["terrain_constraint_fields"] = {
        "terrain_landform_field": cfg.get("terrain_landform_field", "landform_type"),
        "terrain_slope_class_field": cfg.get("terrain_slope_class_field", "slope_class"),
        "terrain_aspect_class_field": cfg.get("terrain_aspect_class_field", "aspect_class"),
        "terrain_slope_position_field": cfg.get("terrain_slope_position_field", "slope_position_class"),
    }
    return meta
