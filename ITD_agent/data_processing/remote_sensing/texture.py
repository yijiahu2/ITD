from __future__ import annotations

from typing import Any

import numpy as np

from ITD_agent.data_processing.remote_sensing.common import downsample


def _prepare_texture_sample(gray: np.ndarray, *, nodata: float | int | None = None, max_dim: int = 512) -> tuple[np.ndarray, np.ndarray]:
    sample = downsample(gray.astype(np.float32), max_dim=max_dim)
    if sample.size == 0:
        return sample, np.zeros(sample.shape, dtype=bool)
    valid_mask = np.isfinite(sample)
    if nodata is not None:
        valid_mask &= ~np.isclose(sample, float(nodata))
    return sample, valid_mask


def _estimate_basic_texture(sample: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    if sample.size == 0 or not np.any(valid_mask):
        return {}
    valid_values = sample[valid_mask]
    fill_value = float(np.median(valid_values))
    filled = np.where(valid_mask, sample, fill_value)
    gy, gx = np.gradient(filled)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    valid_grad = grad[valid_mask]
    return {
        "pixel_std": float(np.std(valid_values)),
        "pixel_mean": float(np.mean(valid_values)),
        "gradient_mean": float(np.mean(valid_grad)) if valid_grad.size else 0.0,
        "gradient_std": float(np.std(valid_grad)) if valid_grad.size else 0.0,
    }


def _quantize_texture(sample: np.ndarray, valid_mask: np.ndarray, *, levels: int = 16) -> np.ndarray:
    quantized = np.zeros(sample.shape, dtype=np.int32)
    if not np.any(valid_mask):
        return quantized
    valid_values = sample[valid_mask]
    value_min = float(np.min(valid_values))
    value_max = float(np.max(valid_values))
    if value_max <= value_min:
        quantized[valid_mask] = 0
        return quantized
    scaled = (valid_values - value_min) / (value_max - value_min)
    quantized[valid_mask] = np.clip(np.floor(scaled * (levels - 1)), 0, levels - 1).astype(np.int32)
    return quantized


def _pair_slices(length: int, delta: int) -> tuple[slice, slice]:
    if delta >= 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(-delta, length), slice(0, length + delta)


def _estimate_glcm_texture(
    sample: np.ndarray,
    valid_mask: np.ndarray,
    *,
    levels: int = 16,
    offsets: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (-1, 1)),
) -> dict[str, Any]:
    if sample.size == 0 or not np.any(valid_mask):
        return {}

    height, width = sample.shape
    if height < 2 or width < 2:
        return {}

    quantized = _quantize_texture(sample, valid_mask, levels=levels)
    cooc = np.zeros((levels, levels), dtype=np.float64)

    for dy, dx in offsets:
        if abs(dy) >= height or abs(dx) >= width:
            continue
        ys, yd = _pair_slices(height, dy)
        xs, xd = _pair_slices(width, dx)
        src_valid = valid_mask[ys, xs]
        dst_valid = valid_mask[yd, xd]
        pair_mask = src_valid & dst_valid
        if not np.any(pair_mask):
            continue

        src_vals = quantized[ys, xs][pair_mask]
        dst_vals = quantized[yd, xd][pair_mask]
        pair_ids = src_vals * levels + dst_vals
        counts = np.bincount(pair_ids, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
        cooc += counts
        cooc += counts.T

    total = float(cooc.sum())
    if total <= 0:
        return {}

    p = cooc / total
    i_idx = np.arange(levels, dtype=np.float64).reshape(-1, 1)
    j_idx = np.arange(levels, dtype=np.float64).reshape(1, -1)
    diff_sq = (i_idx - j_idx) ** 2

    contrast = float(np.sum(p * diff_sq))
    asm = float(np.sum(p ** 2))
    energy = float(np.sqrt(asm))
    homogeneity = float(np.sum(p / (1.0 + diff_sq)))
    nonzero = p[p > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    pi = np.sum(p, axis=1)
    pj = np.sum(p, axis=0)
    mu_i = float(np.sum(np.arange(levels, dtype=np.float64) * pi))
    mu_j = float(np.sum(np.arange(levels, dtype=np.float64) * pj))
    sigma_i = float(np.sqrt(np.sum(((np.arange(levels, dtype=np.float64) - mu_i) ** 2) * pi)))
    sigma_j = float(np.sqrt(np.sum(((np.arange(levels, dtype=np.float64) - mu_j) ** 2) * pj)))
    if sigma_i <= 1e-12 or sigma_j <= 1e-12:
        correlation = 1.0
    else:
        correlation = float(np.sum(p * ((i_idx - mu_i) * (j_idx - mu_j))) / (sigma_i * sigma_j))

    return {
        "contrast": contrast,
        "entropy": entropy,
        "asm": asm,
        "energy": energy,
        "correlation": correlation,
        "homogeneity": homogeneity,
        "idm": homogeneity,
        "glcm_levels": int(levels),
        "glcm_offsets": [list(item) for item in offsets],
    }


def estimate_texture(gray: np.ndarray, *, nodata: float | int | None = None) -> dict[str, Any]:
    sample, valid_mask = _prepare_texture_sample(gray, nodata=nodata)
    basic = _estimate_basic_texture(sample, valid_mask)
    glcm = _estimate_glcm_texture(sample, valid_mask)
    return {**basic, **glcm}
