from __future__ import annotations

from typing import Any

import numpy as np

from ITD_agent.common.values import safe_float as _safe_float
from ITD_agent.data_processing.contracts import LogicalBlockPlanEntry, ProcessingBlockProfile
from ITD_agent.data_processing.remote_sensing.common import (
    as_rgb_uint8,
    coarse_grid_slices,
    read_mask_window,
    read_raster_window,
    rgb_to_gray_uint8,
    valid_mask_from_data,
)
from ITD_agent.data_processing.remote_sensing.policy_templates import select_policy_template
from ITD_agent.data_processing.remote_sensing.quality import estimate_quality_from_rgb
from ITD_agent.data_processing.remote_sensing.texture import estimate_texture


def read_block_rgb_window(working_dom_path: str, block_window: list[int], band_mapping: dict[str, int]) -> tuple[np.ndarray, float | int | None]:
    indexes = [int(band_mapping[key]) for key in ("red", "green", "blue")]
    arr, _, _, nodata = read_raster_window(working_dom_path, indexes=indexes, window=block_window)
    return arr.astype(np.float32), nodata


def read_block_valid_mask(valid_mask_path: str | None, block_window: list[int]) -> np.ndarray | None:
    return read_mask_window(valid_mask_path, window=block_window)


def compute_block_quality_metrics(rgb: np.ndarray, valid_mask: np.ndarray | None, nodata: float | int | None) -> dict[str, Any]:
    mask = valid_mask_from_data(rgb, nodata=nodata, valid_mask=valid_mask)
    rgb_u8 = as_rgb_uint8(rgb, mask)
    return estimate_quality_from_rgb(rgb_u8, mask)


def compute_block_texture_metrics(rgb: np.ndarray, valid_mask: np.ndarray | None, nodata: float | int | None) -> dict[str, Any]:
    mask = valid_mask_from_data(rgb, nodata=nodata, valid_mask=valid_mask)
    gray = rgb_to_gray_uint8(as_rgb_uint8(rgb, mask)).astype(np.float32)
    texture = estimate_texture(gray)
    contrast = _safe_float(texture.get("contrast")) or 0.0
    entropy = _safe_float(texture.get("entropy")) or 0.0
    gradient_mean = _safe_float(texture.get("gradient_mean")) or 0.0
    complexity = min(1.0, 0.30 * min(entropy / 6.0, 1.0) + 0.40 * min(contrast / 8.0, 1.0) + 0.30 * min(gradient_mean / 32.0, 1.0))
    texture["texture_complexity_score"] = complexity
    return texture


def compute_block_heterogeneity(rgb: np.ndarray, valid_mask: np.ndarray | None, nodata: float | int | None, coarse_grid: tuple[int, int] = (7, 7)) -> dict[str, Any]:
    mask = valid_mask_from_data(rgb, nodata=nodata, valid_mask=valid_mask)
    rgb_u8 = as_rgb_uint8(rgb, mask)
    gray = rgb_to_gray_uint8(rgb_u8).astype(np.float32)
    brightness_grid: list[float] = []
    shadow_grid: list[float] = []
    gradient_grid: list[float] = []
    valid_ratio_grid: list[float] = []
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    for y_slice, x_slice in coarse_grid_slices(gray.shape[0], gray.shape[1], coarse_grid[0], coarse_grid[1]):
        local_mask = mask[y_slice, x_slice]
        if local_mask.size == 0:
            continue
        valid_ratio_grid.append(float(local_mask.mean()))
        if not np.any(local_mask):
            brightness_grid.append(0.0)
            shadow_grid.append(1.0)
            gradient_grid.append(0.0)
            continue
        local_gray = gray[y_slice, x_slice]
        local_grad = grad[y_slice, x_slice]
        vals = local_gray[local_mask]
        brightness_grid.append(float(np.mean(vals)))
        shadow_grid.append(float(np.mean((local_gray <= np.percentile(vals, 20)).astype(np.float32)[local_mask])))
        gradient_grid.append(float(np.mean(local_grad[local_mask])))

    brightness_var = float(np.var(brightness_grid)) if brightness_grid else 0.0
    shadow_var = float(np.var(shadow_grid)) if shadow_grid else 0.0
    gradient_var = float(np.var(gradient_grid)) if gradient_grid else 0.0
    valid_ratio_var = float(np.var(valid_ratio_grid)) if valid_ratio_grid else 0.0
    heterogeneity_score = min(
        1.0,
        min(brightness_var / 800.0, 1.0) * 0.30
        + min(shadow_var / 0.08, 1.0) * 0.25
        + min(gradient_var / 120.0, 1.0) * 0.25
        + min(valid_ratio_var / 0.08, 1.0) * 0.20,
    )
    if heterogeneity_score >= 0.66:
        heterogeneity_level = "high"
    elif heterogeneity_score >= 0.33:
        heterogeneity_level = "medium"
    else:
        heterogeneity_level = "low"
    return {
        "heterogeneity_coarse_grid": [int(coarse_grid[0]), int(coarse_grid[1])],
        "brightness_variance_across_cells": brightness_var,
        "shadow_spatial_variance": shadow_var,
        "gradient_variance_across_cells": gradient_var,
        "valid_ratio_variance_across_cells": valid_ratio_var,
        "block_heterogeneity_score": heterogeneity_score,
        "block_heterogeneity_level": heterogeneity_level,
    }


def _expected_tile_count(block_entry: LogicalBlockPlanEntry) -> int:
    return int(block_entry.expected_tile_count)


def build_processing_block_profile(dom_contract: dict[str, Any], block_entry: LogicalBlockPlanEntry, runtime_cfg: dict[str, Any]) -> ProcessingBlockProfile:
    rgb, nodata = read_block_rgb_window(dom_contract["working_dom_path"], block_entry.block_window, dom_contract["band_mapping"])
    valid_mask = read_block_valid_mask(dom_contract.get("valid_mask_path"), block_entry.block_window)
    merged_mask = valid_mask_from_data(rgb, nodata=nodata, valid_mask=valid_mask)
    valid_pixel_ratio = float(merged_mask.mean()) if merged_mask.size else 0.0
    quality = compute_block_quality_metrics(rgb, valid_mask, nodata)
    texture = compute_block_texture_metrics(rgb, valid_mask, nodata)
    heterogeneity = compute_block_heterogeneity(rgb, valid_mask, nodata)
    block_features = {
        "valid_pixel_ratio": valid_pixel_ratio,
        "shadow_ratio_estimate": quality.get("shadow_ratio_estimate"),
        "blur_score": quality.get("blur_score"),
        "gradient_mean": texture.get("gradient_mean"),
        "texture_complexity_score": texture.get("texture_complexity_score"),
        "low_texture_flag": (texture.get("texture_complexity_score") or 0.0) < 0.30,
        "dense_texture_flag": (texture.get("texture_complexity_score") or 0.0) >= 0.60,
        "block_heterogeneity_level": heterogeneity.get("block_heterogeneity_level"),
    }
    policy = select_policy_template(block_features)
    expected_tile_count = _expected_tile_count(block_entry)
    empty_tile_estimate = int(round(expected_tile_count * (1.0 - valid_pixel_ratio)))
    high_risk_tile_estimate = int(round(expected_tile_count * (_safe_float(heterogeneity.get("block_heterogeneity_score")) or 0.0) * 0.5))
    return ProcessingBlockProfile(
        block_id=block_entry.block_id,
        dom_id=block_entry.dom_id,
        block_index=block_entry.block_index,
        block_window=list(block_entry.block_window),
        block_geo_bounds=list(block_entry.block_geo_bounds),
        width=block_entry.width,
        height=block_entry.height,
        edge_block_flag=block_entry.edge_block_flag,
        overlap_with_neighbors_px=block_entry.overlap_with_neighbors_px,
        valid_pixel_ratio=valid_pixel_ratio,
        skip_block_candidate=valid_pixel_ratio < 0.05,
        low_valid_area_flag=valid_pixel_ratio < 0.30,
        brightness_mean=_safe_float(quality.get("brightness_mean")),
        brightness_std=_safe_float(quality.get("brightness_std")),
        shadow_ratio_estimate=_safe_float(quality.get("shadow_ratio_estimate")),
        overexposed_ratio=_safe_float(quality.get("overexposed_ratio")),
        underexposed_ratio=_safe_float(quality.get("underexposed_ratio")),
        laplacian_variance=_safe_float(quality.get("laplacian_variance")),
        tenengrad=_safe_float(quality.get("tenengrad")),
        blur_score=_safe_float(quality.get("blur_score")),
        stripe_noise_score=_safe_float(quality.get("stripe_noise_score")),
        stripe_noise_direction=str(quality.get("stripe_noise_direction") or "none"),
        color_cast_score=_safe_float(quality.get("color_cast_score")),
        gradient_mean=_safe_float(texture.get("gradient_mean")),
        gradient_std=_safe_float(texture.get("gradient_std")),
        texture_entropy=_safe_float(texture.get("entropy")),
        texture_contrast=_safe_float(texture.get("contrast")),
        texture_homogeneity=_safe_float(texture.get("homogeneity")),
        texture_complexity_score=_safe_float(texture.get("texture_complexity_score")),
        low_texture_flag=bool(block_features["low_texture_flag"]),
        dense_texture_flag=bool(block_features["dense_texture_flag"]),
        heterogeneity_coarse_grid=list(heterogeneity["heterogeneity_coarse_grid"]),
        brightness_variance_across_cells=_safe_float(heterogeneity.get("brightness_variance_across_cells")),
        shadow_spatial_variance=_safe_float(heterogeneity.get("shadow_spatial_variance")),
        gradient_variance_across_cells=_safe_float(heterogeneity.get("gradient_variance_across_cells")),
        valid_ratio_variance_across_cells=_safe_float(heterogeneity.get("valid_ratio_variance_across_cells")),
        block_heterogeneity_score=_safe_float(heterogeneity.get("block_heterogeneity_score")),
        block_heterogeneity_level=str(heterogeneity.get("block_heterogeneity_level") or "low"),
        risk_tags=list(policy["risk_tags"]),
        localized_risk_tags=list(policy["localized_risk_tags"]),
        quality_class=str(policy["quality_class"]),
        priority_score=_safe_float(policy.get("priority_score")),
        expected_failure_modes=list(policy["expected_failure_modes"]),
        policy_template_name=str(policy["policy_template_name"]),
        diam_list=str(policy["diam_list"]),
        augment=bool(policy["augment"]),
        iou_merge_thr=_safe_float(policy.get("iou_merge_thr")),
        enable_tile_fast_check=bool(policy["enable_tile_fast_check"]),
        fusion_priority=str(policy["fusion_priority"]),
        expert_model_candidates=[],
        memory_candidate_policy="high_risk_only",
        finetune_candidate_policy="failed_or_corrected_only",
        expected_tile_count=expected_tile_count,
        empty_tile_estimate=empty_tile_estimate,
        high_risk_tile_estimate=high_risk_tile_estimate,
        status="skip" if valid_pixel_ratio < 0.05 else "ready",
        metadata={
            "working_dom_path": dom_contract["working_dom_path"],
            "valid_mask_path": dom_contract.get("valid_mask_path"),
            "gsd_status": dom_contract.get("gsd_status"),
        },
    )


def build_processing_block_profiles(dom_contract: dict[str, Any], block_plan: list[LogicalBlockPlanEntry], runtime_cfg: dict[str, Any]) -> list[ProcessingBlockProfile]:
    return [build_processing_block_profile(dom_contract, block_entry, runtime_cfg) for block_entry in block_plan]
