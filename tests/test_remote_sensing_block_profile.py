from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.data_processing.remote_sensing.block_plan import generate_logical_block_plan
from ITD_agent.data_processing.remote_sensing.block_profile import build_processing_block_profiles


def _write_raster(path: Path, data: np.ndarray, *, nodata: int | float = 0) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(100.0, 31.0, 0.02, 0.02),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_mask(path: Path, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(100.0, 31.0, 0.02, 0.02),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_build_processing_block_profiles_reads_block_window_and_generates_metrics(tmp_path: Path) -> None:
    working_dom = tmp_path / "working_dom.tif"
    valid_mask = tmp_path / "valid_mask.tif"
    rgb = np.full((3, 400, 400), 120, dtype=np.uint8)
    rgb[:, 120:240, 120:240] = 40
    rgb[:, 250:330, 50:150] = 220
    mask = np.ones((400, 400), dtype=np.uint8)
    mask[:40, :] = 0

    _write_raster(working_dom, rgb)
    _write_mask(valid_mask, mask)

    dom_contract = {
        "dom_id": "dom_001",
        "working_dom_path": str(working_dom),
        "valid_mask_path": str(valid_mask),
        "band_mapping": {"red": 1, "green": 2, "blue": 3},
        "gsd_status": "acceptable",
        "transform": [0.02, 0.0, 100.0, 0.0, -0.02, 31.0],
        "width": 400,
        "height": 400,
        "processing_block_px": 256,
        "processing_block_stride_px": 224,
        "processing_block_overlap_px": 32,
        "processing_edge_absorb_px": 32,
        "processing_block_min_preferred_px": 224,
        "processing_block_max_preferred_px": 320,
        "tile_px": 128,
        "tile_stride_px": 96,
    }
    block_plan = generate_logical_block_plan(dom_contract)
    profiles = build_processing_block_profiles(dom_contract, block_plan, {})

    assert profiles
    first = profiles[0].to_dict()
    assert first["block_window"] == [0, 0, 256, 256]
    assert 0.0 < first["valid_pixel_ratio"] < 1.0
    assert first["brightness_mean"] is not None
    assert first["texture_complexity_score"] is not None
    assert first["block_heterogeneity_level"] in {"low", "medium", "high"}
    assert first["policy_template_name"] in {
        "default",
        "dense_small_crown",
        "large_sparse_crown",
        "shadow_weak_boundary",
        "high_heterogeneity",
    }


def test_build_processing_block_profiles_marks_low_valid_blocks_as_skip(tmp_path: Path) -> None:
    working_dom = tmp_path / "working_dom_low_valid.tif"
    valid_mask = tmp_path / "valid_mask_low_valid.tif"
    rgb = np.full((3, 256, 256), 100, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:8, :8] = 1

    _write_raster(working_dom, rgb)
    _write_mask(valid_mask, mask)

    dom_contract = {
        "dom_id": "dom_001",
        "working_dom_path": str(working_dom),
        "valid_mask_path": str(valid_mask),
        "band_mapping": {"red": 1, "green": 2, "blue": 3},
        "gsd_status": "acceptable",
        "transform": [0.02, 0.0, 100.0, 0.0, -0.02, 31.0],
        "width": 256,
        "height": 256,
        "processing_block_px": 256,
        "processing_block_stride_px": 224,
        "processing_block_overlap_px": 32,
        "processing_edge_absorb_px": 32,
        "processing_block_min_preferred_px": 224,
        "processing_block_max_preferred_px": 320,
        "tile_px": 128,
        "tile_stride_px": 96,
    }
    block_plan = generate_logical_block_plan(dom_contract)
    profiles = build_processing_block_profiles(dom_contract, block_plan, {})

    assert len(profiles) == 1
    profile = profiles[0].to_dict()
    assert profile["skip_block_candidate"] is True
    assert profile["status"] == "skip"
