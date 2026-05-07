from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_layer.adapters import build_input_manifest
from ITD_agent.context_engine import build_online_scene_state


def _write_raster(path: Path, data: np.ndarray, resolution: float = 0.1) -> None:
    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(0, 0, resolution, resolution),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_build_online_scene_state_with_chm_and_public_dataset(tmp_path: Path) -> None:
    dom = tmp_path / "dom.tif"
    dem = tmp_path / "dem.tif"
    chm = tmp_path / "chm.tif"
    _write_raster(dom, np.full((32, 32), 128, dtype=np.uint8))
    _write_raster(dem, np.full((32, 32), 12, dtype=np.float32), resolution=12.5)
    chm_data = np.zeros((32, 32), dtype=np.float32)
    chm_data[8:12, 8:12] = 15
    chm_data[20:24, 20:24] = 18
    _write_raster(chm, chm_data)

    cfg = {
        "runtime": {"run_name": "scene_state_test"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": str(dom), "required": True}]},
            "terrain": {"dem": [{"id": "dem", "path": str(dem), "required": True}]},
            "canopy": {"chm": [{"id": "chm", "path": str(chm), "required": True}]},
            "public_datasets": {
                "datasets": [
                    {
                        "id": "dataset4",
                        "format": "coco",
                        "image_root": str(tmp_path / "images"),
                        "annotation_path": str(tmp_path / "ann.json"),
                        "forest_type": "subtropical_evergreen_broadleaf_forest",
                        "target_expert_families": ["boundary_calibration"],
                    }
                ]
            },
        },
        "chm_tif": str(chm),
        "dem_tif": str(dem),
    }
    manifest = build_input_manifest(cfg)
    data_processing_summary = {
        "image_profiles": [
            {
                "source_id": "dom",
                "resolution_x_m": 0.1,
                "width": 32,
                "height": 32,
                "area_ha": 0.01024,
                "texture_summary": {"contrast": 3.2, "entropy": 5.1, "energy": 0.14, "correlation": 0.65, "homogeneity": 0.42, "gradient_mean": 12.0},
                "quality_summary": {"quality_metrics": {"laplacian_variance": 200.0, "shadow_ratio_estimate": 0.12, "stripe_noise_score": 0.1, "color_cast_score": 0.02}},
            }
        ],
        "dem_profiles": [{"resolution_x_m": 12.5}],
        "metadata": {
            "input_manifest_summary": {
                "public_datasets": [
                    {
                        "forest_types": ["subtropical_evergreen_broadleaf_forest"],
                        "target_expert_families": ["boundary_calibration"],
                    }
                ]
            }
        },
    }
    terrain_info = {
        "global_terrain_background": {"landform_type": "hill_low"},
        "dom_terrain_context": {"landform_type": "hill_middle", "slope_class": "III_inclined"},
        "terrain_layer_policy": {"global_role": "weak_background_constraint", "dom_role": "primary_context"},
    }

    state = build_online_scene_state(
        runtime_cfg=cfg,
        input_manifest=manifest,
        terrain_info=terrain_info,
        data_processing_summary=data_processing_summary,
    )

    assert state["input_availability"]["has_dom"] is True
    assert state["input_availability"]["has_dem"] is True
    assert state["input_availability"]["has_chm"] is True
    assert state["chm_profile"]["available"] is True
    assert state["public_dataset_prior"]["available"] is True
    assert "boundary_calibration" in state["public_dataset_prior"]["recommended_model_families"]
