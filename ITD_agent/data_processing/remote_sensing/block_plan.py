from __future__ import annotations

from math import ceil
from typing import Any

from rasterio.transform import array_bounds

from ITD_agent.data_processing.contracts import LogicalBlockPlanEntry


def _compute_axis_windows(
    size: int,
    block_px: int,
    stride_px: int,
    edge_absorb_px: int,
    min_preferred_px: int,
    max_preferred_px: int,
) -> list[tuple[int, int]]:
    if size <= block_px:
        return [(0, size)]

    windows: list[tuple[int, int]] = []
    start = 0
    while start < size:
        remaining = size - start
        if remaining <= block_px:
            final_start = start
            final_width = remaining
            if remaining < min_preferred_px:
                final_width = min(min_preferred_px, size)
                final_start = max(size - final_width, 0)
            final_width = min(final_width, max_preferred_px, size - final_start)

            if windows:
                prev_start, _ = windows[-1]
                if final_start <= prev_start + edge_absorb_px:
                    windows[-1] = (prev_start, min(size - prev_start, max_preferred_px))
                elif windows[-1] != (final_start, final_width):
                    windows.append((final_start, final_width))
            else:
                windows.append((final_start, final_width))
            break
        windows.append((start, block_px))
        start += stride_px

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in windows:
        normalized = (max(item[0], 0), min(item[1], size))
        if normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _block_geo_bounds(transform: Any, x: int, y: int, w: int, h: int) -> list[float]:
    min_y, min_x, max_y, max_x = array_bounds(h, w, transform * transform.translation(x, y))
    return [float(min_x), float(min_y), float(max_x), float(max_y)]


def _estimate_window_count(size: int, tile_size: int, stride: int) -> int:
    if size <= tile_size:
        return 1
    return int(ceil((size - tile_size) / float(stride))) + 1


def _estimate_tile_count_for_block(width: int, height: int, tile_px: int, tile_stride_px: int) -> int:
    return _estimate_window_count(width, tile_px, tile_stride_px) * _estimate_window_count(height, tile_px, tile_stride_px)


def generate_logical_block_plan(dom_contract: dict[str, Any]) -> list[LogicalBlockPlanEntry]:
    width = int(dom_contract["width"])
    height = int(dom_contract["height"])
    block_px = int(dom_contract["processing_block_px"])
    stride_px = int(dom_contract["processing_block_stride_px"])
    overlap_px = int(dom_contract["processing_block_overlap_px"])
    edge_absorb_px = int(dom_contract["processing_edge_absorb_px"])
    min_preferred_px = int(dom_contract["processing_block_min_preferred_px"])
    max_preferred_px = int(dom_contract["processing_block_max_preferred_px"])
    tile_px = int(dom_contract["tile_px"])
    tile_stride_px = int(dom_contract["tile_stride_px"])
    dom_id = str(dom_contract["dom_id"])

    transform_list = dom_contract.get("transform") or []
    if len(transform_list) != 6:
        raise ValueError("dom_input_contract.transform 缺失或非法，无法生成 block_geo_bounds。")

    from affine import Affine

    transform = Affine(*transform_list)
    x_windows = _compute_axis_windows(width, block_px, stride_px, edge_absorb_px, min_preferred_px, max_preferred_px)
    y_windows = _compute_axis_windows(height, block_px, stride_px, edge_absorb_px, min_preferred_px, max_preferred_px)

    entries: list[LogicalBlockPlanEntry] = []
    block_index = 0
    for y, h in y_windows:
        for x, w in x_windows:
            block_index += 1
            edge_flag = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
            entries.append(
                LogicalBlockPlanEntry(
                    block_id=f"{dom_id}_b_{block_index:04d}",
                    dom_id=dom_id,
                    block_index=block_index,
                    block_window=[int(x), int(y), int(w), int(h)],
                    block_geo_bounds=_block_geo_bounds(transform, x, y, w, h),
                    width=int(w),
                    height=int(h),
                    edge_block_flag=edge_flag,
                    overlap_with_neighbors_px=overlap_px,
                    expected_tile_count=_estimate_tile_count_for_block(w, h, tile_px, tile_stride_px),
                    status="ready",
                )
            )
    return entries
