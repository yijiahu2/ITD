from __future__ import annotations

from typing import Any

import numpy as np

from ITD_agent.data_processing.contracts import ProcessingBlockProfile, TileRunContext
from ITD_agent.data_processing.remote_sensing.common import (
    as_rgb_uint8,
    read_mask_window,
    read_raster_window,
    rgb_to_gray_uint8,
    valid_mask_from_data,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _axis_tile_windows(size: int, tile_px: int, stride_px: int, snap_last_tile_to_edge: bool) -> list[int]:
    if size <= tile_px:
        return [0]
    starts: list[int] = []
    current = 0
    while current < size:
        starts.append(current)
        current += stride_px
    final_start = size - tile_px
    if snap_last_tile_to_edge and (not starts or starts[-1] != final_start):
        starts.append(final_start)
    elif not snap_last_tile_to_edge and current < size:
        starts.append(current)
    return sorted(set(starts))


def generate_tile_local_plan(block_profile: ProcessingBlockProfile, dom_contract: dict[str, Any]) -> list[dict[str, Any]]:
    tile_px = int(dom_contract["tile_px"])
    stride_px = int(dom_contract["tile_stride_px"])
    tile_overlap_px = int(dom_contract.get("tile_overlap_px", tile_px - stride_px))
    expected_stride = tile_px - tile_overlap_px
    if stride_px != expected_stride:
        raise ValueError(f"tile_stride_px={stride_px} 必须等于 tile_px - tile_overlap_px={expected_stride}。")
    snap = bool(dom_contract.get("snap_last_tile_to_edge", True))
    x_starts = _axis_tile_windows(block_profile.width, tile_px, stride_px, snap)
    y_starts = _axis_tile_windows(block_profile.height, tile_px, stride_px, snap)
    plan: list[dict[str, Any]] = []
    tile_index = 0
    for y in y_starts:
        for x in x_starts:
            tile_index += 1
            plan.append(
                {
                    "tile_index": tile_index,
                    "tile_local_window": [int(x), int(y), tile_px, tile_px],
                }
            )
    return plan


def compute_tile_read_window(block_window: list[int], tile_local_window: list[int], tile_px: int) -> list[int]:
    bx, by, _, _ = [int(v) for v in block_window]
    tx, ty, tw, th = [int(v) for v in tile_local_window]
    return [bx + tx, by + ty, tile_px, tile_px]


def _compute_padding(block_window: list[int], tile_local_window: list[int], tile_px: int) -> tuple[int, int, int, int]:
    _, _, bw, bh = [int(v) for v in block_window]
    tx, ty, tw, th = [int(v) for v in tile_local_window]
    pad_left = 0
    pad_top = 0
    pad_right = max(tx + tw - bw, 0)
    pad_bottom = max(ty + th - bh, 0)
    return pad_left, pad_top, pad_right, pad_bottom


def compute_valid_write_window(block_window: list[int], tile_local_window: list[int], tile_px: int) -> list[int]:
    read_window = compute_tile_read_window(block_window, tile_local_window, tile_px)
    pad_left, pad_top, pad_right, pad_bottom = _compute_padding(block_window, tile_local_window, tile_px)
    valid_w = tile_px - pad_left - pad_right
    valid_h = tile_px - pad_top - pad_bottom
    return [read_window[0] + pad_left, read_window[1] + pad_top, valid_w, valid_h]


def read_tile_rgb_window(working_dom_path: str, read_window: list[int], band_mapping: dict[str, int], tile_px: int) -> tuple[np.ndarray, float | int | None]:
    indexes = [int(band_mapping[key]) for key in ("red", "green", "blue")]
    arr, _, _, nodata = read_raster_window(
        working_dom_path,
        indexes=indexes,
        window=read_window,
        boundless=True,
        fill_value=0,
    )
    return arr.astype(np.float32), nodata


def read_tile_valid_mask(valid_mask_path: str | None, read_window: list[int], tile_px: int) -> np.ndarray | None:
    return read_mask_window(valid_mask_path, window=read_window, boundless=True, fill_value=0)


def compute_tile_fast_features(rgb: np.ndarray, valid_mask: np.ndarray | None, nodata: float | int | None) -> dict[str, Any]:
    mask = valid_mask_from_data(rgb, nodata=nodata, valid_mask=valid_mask)
    valid_pixel_ratio = float(mask.mean()) if mask.size else 0.0
    empty_tile_flag = valid_pixel_ratio < 0.05
    rgb_u8 = as_rgb_uint8(rgb, mask)
    gray = rgb_to_gray_uint8(rgb_u8).astype(np.float32) / 255.0
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    brightness_proxy = float(np.mean(gray[mask])) if np.any(mask) else 0.0
    shadow_proxy = float(np.mean((1.0 - gray)[mask])) if np.any(mask) else 1.0
    gradient_proxy = float(np.mean(grad[mask])) if np.any(mask) else 0.0
    local_texture_proxy = float(np.std(gray[mask])) if np.any(mask) else 0.0
    return {
        "valid_pixel_ratio": valid_pixel_ratio,
        "empty_tile_flag": empty_tile_flag,
        "brightness_proxy": brightness_proxy,
        "shadow_proxy": shadow_proxy,
        "gradient_proxy": gradient_proxy,
        "local_texture_proxy": local_texture_proxy,
    }


def apply_tile_light_overrides(block_profile: ProcessingBlockProfile, tile_features: dict[str, Any], padding_ratio: float) -> dict[str, Any]:
    final_diam_list = block_profile.diam_list
    final_augment = block_profile.augment
    final_iou_merge_thr = block_profile.iou_merge_thr
    final_fusion_priority = block_profile.fusion_priority
    tile_delta_detected = False
    reasons: list[str] = []
    tile_risk_tags = list(block_profile.risk_tags)
    skip = False
    skip_reason = None

    if (_safe_float(tile_features.get("valid_pixel_ratio")) or 0.0) < 0.05:
        skip = True
        skip_reason = "low_valid_area"
        tile_delta_detected = True
        reasons.append("valid_pixel_ratio_below_0.05")

    block_shadow = _safe_float(block_profile.shadow_ratio_estimate) or 0.0
    shadow_proxy = _safe_float(tile_features.get("shadow_proxy")) or 0.0
    if shadow_proxy >= max(block_shadow + 0.10, 0.35):
        final_augment = True
        tile_delta_detected = True
        reasons.append("local_shadow")
        if "local_shadow" not in tile_risk_tags:
            tile_risk_tags.append("local_shadow")

    texture_proxy = _safe_float(tile_features.get("local_texture_proxy")) or 0.0
    gradient_proxy = _safe_float(tile_features.get("gradient_proxy")) or 0.0
    if texture_proxy >= 0.18 and gradient_proxy >= 0.12:
        final_diam_list = "64,96,160"
        tile_delta_detected = True
        reasons.append("local_small_crown_pattern")
    elif texture_proxy <= 0.08 and gradient_proxy <= 0.06:
        final_diam_list = "128,256,320"
        tile_delta_detected = True
        reasons.append("local_large_crown_pattern")

    if padding_ratio > 0.30:
        final_fusion_priority = "low"
        tile_delta_detected = True
        reasons.append("high_padding_ratio")

    return {
        "skip": skip,
        "skip_reason": skip_reason,
        "final_diam_list": final_diam_list,
        "final_augment": final_augment,
        "final_iou_merge_thr": final_iou_merge_thr,
        "final_fusion_priority": final_fusion_priority,
        "tile_delta_detected": tile_delta_detected,
        "tile_delta_reason": reasons,
        "tile_risk_tags": tile_risk_tags,
    }


def build_tile_run_context(dom_contract: dict[str, Any], block_profile: ProcessingBlockProfile, tile_plan_entry: dict[str, Any]) -> TileRunContext:
    tile_px = int(dom_contract["tile_px"])
    tile_overlap_px = int(dom_contract.get("tile_overlap_px", tile_px - int(dom_contract.get("tile_stride_px", tile_px))))
    tile_local_window = tile_plan_entry["tile_local_window"]
    allow_elastic = bool(dom_contract.get("allow_elastic_model_input", False))
    pad_small = bool(dom_contract.get("pad_if_smaller_than_model_input", True))
    if allow_elastic and pad_small:
        raise ValueError("allow_elastic_model_input 与 pad_if_smaller_than_model_input 不能同时启用。")
    read_window = compute_tile_read_window(block_profile.block_window, tile_local_window, tile_px)
    valid_write_window = compute_valid_write_window(block_profile.block_window, tile_local_window, tile_px)
    pad_left, pad_top, pad_right, pad_bottom = _compute_padding(block_profile.block_window, tile_local_window, tile_px)
    if not pad_small and (pad_left or pad_top or pad_right or pad_bottom):
        raise ValueError("pad_if_smaller_than_model_input=false 时，边缘 tile 不应产生 padding。")
    padding_ratio = float((pad_left + pad_top + pad_right + pad_bottom) * tile_px / float(tile_px * tile_px)) if tile_px > 0 else 0.0

    rgb, nodata = read_tile_rgb_window(dom_contract["working_dom_path"], read_window, dom_contract["band_mapping"], tile_px)
    valid_mask = read_tile_valid_mask(dom_contract.get("valid_mask_path"), read_window, tile_px)
    features = compute_tile_fast_features(rgb, valid_mask, nodata)
    overrides = apply_tile_light_overrides(block_profile, features, padding_ratio)

    return TileRunContext(
        tile_id=f"{block_profile.block_id}_t_{int(tile_plan_entry['tile_index']):04d}",
        dom_id=block_profile.dom_id,
        block_id=block_profile.block_id,
        tile_index=int(tile_plan_entry["tile_index"]),
        read_window=list(read_window),
        model_window=[0, 0, tile_px, tile_px] if not allow_elastic else [0, 0, valid_write_window[2], valid_write_window[3]],
        valid_write_window=list(valid_write_window),
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        padding_ratio=padding_ratio,
        edge_tile_flag=padding_ratio > 0.0,
        clip_to_valid_write_window=True,
        discard_padding_output=bool(dom_contract.get("discard_padding_output", True)),
        working_dom_path=str(dom_contract["working_dom_path"]),
        valid_mask_path=dom_contract.get("valid_mask_path"),
        crs=dom_contract.get("crs"),
        transform_ref="original_transform",
        gsd_m=_safe_float(dom_contract.get("gsd_x_m")) or _safe_float(dom_contract.get("gsd_y_m")),
        gsd_status=dom_contract.get("gsd_status"),
        band_mapping=dict(dom_contract.get("band_mapping") or {}),
        normalization_policy=dom_contract.get("normalization_policy"),
        nodata_policy=dom_contract.get("nodata_policy"),
        inherited_risk_tags=list(block_profile.risk_tags),
        inherited_quality_class=block_profile.quality_class,
        inherited_priority_score=block_profile.priority_score,
        inherited_block_heterogeneity_level=block_profile.block_heterogeneity_level,
        inherited_expected_failure_modes=list(block_profile.expected_failure_modes),
        inherited_diam_list=block_profile.diam_list,
        inherited_augment=block_profile.augment,
        inherited_iou_merge_thr=block_profile.iou_merge_thr,
        inherited_fusion_priority=block_profile.fusion_priority,
        enable_tile_fast_check=block_profile.enable_tile_fast_check,
        valid_pixel_ratio=_safe_float(features.get("valid_pixel_ratio")),
        empty_tile_flag=bool(features.get("empty_tile_flag")),
        brightness_proxy=_safe_float(features.get("brightness_proxy")),
        shadow_proxy=_safe_float(features.get("shadow_proxy")),
        gradient_proxy=_safe_float(features.get("gradient_proxy")),
        local_texture_proxy=_safe_float(features.get("local_texture_proxy")),
        tile_delta_detected=bool(overrides["tile_delta_detected"]),
        tile_delta_reason=list(overrides["tile_delta_reason"]),
        tile_risk_tags=list(overrides["tile_risk_tags"]),
        skip=bool(overrides["skip"]),
        skip_reason=overrides["skip_reason"],
        final_diam_list=overrides["final_diam_list"],
        final_augment=bool(overrides["final_augment"]),
        final_iou_merge_thr=_safe_float(overrides.get("final_iou_merge_thr")),
        final_bsize=int(dom_contract.get("bsize", 256)),
        final_fusion_priority=overrides["final_fusion_priority"],
        expert_model_name=None,
        export_sample_flag=False,
        memory_candidate_flag=False,
        finetune_candidate_flag=False,
        status="skipped" if overrides["skip"] else "ready",
        metadata={
            "allow_elastic_model_input": allow_elastic,
            "pad_if_smaller_than_model_input": pad_small,
            "tile_overlap_px": tile_overlap_px,
        },
    )


def build_tile_contexts_for_block(dom_contract: dict[str, Any], block_profile: ProcessingBlockProfile) -> list[TileRunContext]:
    plan = generate_tile_local_plan(block_profile, dom_contract)
    return [build_tile_run_context(dom_contract, block_profile, item) for item in plan]
