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
from ITD_agent.data_processing.height import build_height_raster_profiles
from ITD_agent.data_processing.imagery.priors import build_image_profiles


def _write_raster(path: Path, data: np.ndarray, resolution: float = 1.0) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(0, 10, resolution, resolution),
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_height_raster_profiles_crop_chm_to_dom_extent(tmp_path: Path) -> None:
    dom = tmp_path / "dom.tif"
    chm = tmp_path / "chm.tif"
    _write_raster(dom, np.full((10, 10), 120, dtype=np.uint8), resolution=1.0)
    _write_raster(chm, np.full((20, 20), 8, dtype=np.float32), resolution=1.0)
    cfg = {
        "runtime": {"run_name": "height_profile_test"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": str(dom), "required": True}]},
            "canopy": {"chm": [{"id": "chm", "path": str(chm), "required": True}]},
        },
    }
    manifest = build_input_manifest(cfg)
    image_profiles = build_image_profiles(manifest, {})
    storage_layout = {"raster_cache": str(tmp_path / "raster_cache")}

    profiles = build_height_raster_profiles(manifest, image_profiles, storage_layout)

    assert len(profiles) == 1
    profile = profiles[0].to_dict()
    assert profile["role"] == "chm"
    assert profile["normalization"]["status"] == "dom_extent_cropped"
    assert profile["height_summary"]["available"] is True
    assert Path(profile["normalization"]["normalized_path"]).exists()


def test_height_raster_profiles_filter_nodata_and_support_reproject_overlap(tmp_path: Path) -> None:
    dom = tmp_path / "dom3857.tif"
    chm = tmp_path / "chm4326.tif"
    _write_raster(dom, np.full((10, 10), 100, dtype=np.uint8), resolution=1.0)

    chm_data = np.full((10, 10), -9999.0, dtype=np.float32)
    chm_data[2:5, 2:5] = 9.0
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0.0001, 0.00001, 0.00001),
        "nodata": -9999.0,
    }
    with rasterio.open(chm, "w", **profile) as dst:
        dst.write(chm_data, 1)

    cfg = {
        "runtime": {"run_name": "height_profile_reproject_test"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": str(dom), "required": True}]},
            "canopy": {"chm": [{"id": "chm", "path": str(chm), "required": True}]},
        },
    }
    manifest = build_input_manifest(cfg)
    image_profiles = build_image_profiles(manifest, {})
    image_profiles[0].crs = "EPSG:3857"
    image_profiles[0].quality_summary = {"bounds": {"left": 0.0, "bottom": 0.0, "right": 10.0, "top": 10.0}}
    storage_layout = {"raster_cache": str(tmp_path / "raster_cache")}

    profiles = build_height_raster_profiles(manifest, image_profiles, storage_layout)
    summary = profiles[0].to_dict()["height_summary"]

    assert np.isfinite(summary["height_mean"])
    assert summary["height_max"] == 9.0
