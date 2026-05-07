from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.data_processing.remote_sensing.block_plan import generate_logical_block_plan


def _dom_contract(*, width: int, height: int) -> dict[str, object]:
    return {
        "dom_id": "dom_001",
        "width": width,
        "height": height,
        "transform": [0.02, 0.0, 100.0, 0.0, -0.02, 31.0],
        "processing_block_px": 5632,
        "processing_block_stride_px": 5120,
        "processing_block_overlap_px": 512,
        "processing_edge_absorb_px": 512,
        "processing_block_min_preferred_px": 5120,
        "processing_block_max_preferred_px": 6144,
        "tile_px": 1024,
        "tile_stride_px": 768,
    }


def test_generate_logical_block_plan_single_block_for_small_dom() -> None:
    plan = generate_logical_block_plan(_dom_contract(width=3072, height=2048))

    assert len(plan) == 1
    entry = plan[0].to_dict()
    assert entry["block_id"] == "dom_001_b_0001"
    assert entry["block_window"] == [0, 0, 3072, 2048]
    assert entry["edge_block_flag"] is True
    assert entry["expected_tile_count"] >= 1


def test_generate_logical_block_plan_respects_stride_and_overlap() -> None:
    plan = generate_logical_block_plan(_dom_contract(width=9000, height=9000))

    assert len(plan) == 4
    windows = [item.block_window for item in plan]
    assert windows[0] == [0, 0, 5632, 5632]
    assert windows[1][0] == 3880
    assert windows[2][1] == 3880
    assert all(item.overlap_with_neighbors_px == 512 for item in plan)


def test_generate_logical_block_plan_absorbs_small_edge_blocks() -> None:
    plan = generate_logical_block_plan(_dom_contract(width=10700, height=5632))

    assert len(plan) == 2
    assert plan[0].block_window == [0, 0, 5632, 5632]
    assert plan[1].block_window[2] >= 5120
    assert plan[1].block_window[0] + plan[1].block_window[2] == 10700
