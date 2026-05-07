from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.data_processing.contracts import ProcessingBlockProfile
from ITD_agent.data_processing.remote_sensing.tile_context import (
    build_tile_contexts_for_block,
    compute_tile_read_window,
)


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


def test_compute_tile_read_window_translates_local_to_global() -> None:
    read_window = compute_tile_read_window([3880, 3880, 320, 320], [96, 96, 128, 128], 128)
    assert read_window == [3976, 3976, 128, 128]


def test_build_tile_contexts_for_block_applies_padding_and_light_overrides(tmp_path: Path) -> None:
    working_dom = tmp_path / "working_dom_tile.tif"
    valid_mask = tmp_path / "valid_mask_tile.tif"
    rgb = np.full((3, 320, 320), 130, dtype=np.uint8)
    rgb[:, 100:180, 100:180] = 30
    mask = np.ones((320, 320), dtype=np.uint8)
    mask[250:, 250:] = 0

    _write_raster(working_dom, rgb)
    _write_mask(valid_mask, mask)

    dom_contract = {
        "working_dom_path": str(working_dom),
        "valid_mask_path": str(valid_mask),
        "band_mapping": {"red": 1, "green": 2, "blue": 3},
        "crs": "EPSG:4547",
        "gsd_x_m": 0.02,
        "gsd_status": "acceptable",
        "normalization_policy": "uint8_passthrough",
        "nodata_policy": "use_valid_mask",
        "tile_px": 128,
        "tile_overlap_px": 32,
        "tile_stride_px": 96,
        "allow_elastic_model_input": False,
        "pad_if_smaller_than_model_input": True,
        "discard_padding_output": True,
        "bsize": 256,
    }
    block_profile = ProcessingBlockProfile(
        block_id="dom_001_b_0001",
        dom_id="dom_001",
        block_index=1,
        block_window=[0, 0, 320, 320],
        width=320,
        height=320,
        risk_tags=["dense_texture"],
        quality_class="medium_risk",
        priority_score=0.72,
        block_heterogeneity_level="high",
        expected_failure_modes=["crown_merge"],
        diam_list="96,192,320",
        augment=False,
        iou_merge_thr=0.28,
        fusion_priority="normal",
        enable_tile_fast_check=True,
        shadow_ratio_estimate=0.10,
    )

    contexts = build_tile_contexts_for_block(dom_contract, block_profile)

    assert contexts
    edge_tiles = [item for item in contexts if item.edge_tile_flag]
    assert edge_tiles
    assert any(item.final_fusion_priority == "low" for item in edge_tiles)
    assert any(item.tile_delta_detected for item in contexts)
    assert all(item.metadata["tile_overlap_px"] == 32 for item in contexts)


def test_generate_tile_local_plan_rejects_stride_overlap_mismatch() -> None:
    block_profile = ProcessingBlockProfile(
        block_id="dom_001_b_0001",
        dom_id="dom_001",
        block_index=1,
        block_window=[0, 0, 320, 320],
        width=320,
        height=320,
    )
    bad_contract = {
        "tile_px": 128,
        "tile_overlap_px": 16,
        "tile_stride_px": 96,
        "snap_last_tile_to_edge": True,
    }

    from ITD_agent.data_processing.remote_sensing.tile_context import generate_tile_local_plan

    try:
        generate_tile_local_plan(block_profile, bad_contract)
    except ValueError as exc:
        assert "tile_stride_px" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched stride/overlap.")


def test_build_tile_run_context_rejects_padding_when_pad_disabled(tmp_path: Path) -> None:
    working_dom = tmp_path / "working_dom_pad_disabled.tif"
    valid_mask = tmp_path / "valid_mask_pad_disabled.tif"
    rgb = np.full((3, 320, 320), 130, dtype=np.uint8)
    mask = np.ones((320, 320), dtype=np.uint8)
    _write_raster(working_dom, rgb)
    _write_mask(valid_mask, mask)

    dom_contract = {
        "working_dom_path": str(working_dom),
        "valid_mask_path": str(valid_mask),
        "band_mapping": {"red": 1, "green": 2, "blue": 3},
        "crs": "EPSG:4547",
        "gsd_x_m": 0.02,
        "gsd_status": "acceptable",
        "normalization_policy": "uint8_passthrough",
        "nodata_policy": "use_valid_mask",
        "tile_px": 128,
        "tile_overlap_px": 32,
        "tile_stride_px": 96,
        "allow_elastic_model_input": False,
        "pad_if_smaller_than_model_input": False,
        "discard_padding_output": True,
        "bsize": 256,
    }
    block_profile = ProcessingBlockProfile(
        block_id="dom_001_b_0001",
        dom_id="dom_001",
        block_index=1,
        block_window=[0, 0, 320, 320],
        width=320,
        height=320,
    )

    from ITD_agent.data_processing.remote_sensing.tile_context import build_tile_contexts_for_block

    try:
        build_tile_contexts_for_block(dom_contract, block_profile)
    except ValueError as exc:
        assert "pad_if_smaller_than_model_input" in str(exc)
    else:
        raise AssertionError("Expected ValueError when padding is disabled for edge tiles.")
