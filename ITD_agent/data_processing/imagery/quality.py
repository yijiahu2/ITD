from __future__ import annotations

from typing import Any

import numpy as np
import rasterio

from ITD_agent.data_processing.imagery.common import normalize_to_uint8, rgb_to_gray_uint8


def prepare_rgb_preview(
    src: rasterio.io.DatasetReader,
    *,
    nodata: float | int | None = None,
    max_dim: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    out_h = min(int(src.height), max_dim)
    out_w = min(int(src.width), max_dim)
    band_count = min(int(src.count), 3)
    if band_count <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8), np.zeros((out_h, out_w), dtype=bool)
    band_indexes = list(range(1, band_count + 1))
    preview = src.read(band_indexes, out_shape=(band_count, out_h, out_w)).astype(np.float32)
    valid_mask = np.all(np.isfinite(preview), axis=0)
    if nodata is not None:
        valid_mask &= ~np.any(np.isclose(preview, float(nodata)), axis=0)
    if band_count == 1:
        band = normalize_to_uint8(preview[0], valid_mask)
        rgb = np.stack([band, band, band], axis=-1)
    else:
        channels = [normalize_to_uint8(preview[idx], valid_mask) for idx in range(band_count)]
        while len(channels) < 3:
            channels.append(channels[-1].copy())
        rgb = np.stack(channels[:3], axis=-1)
    return rgb, valid_mask


def _estimate_blur(gray_u8: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    if gray_u8.size == 0 or not np.any(valid_mask):
        return {}
    gray = gray_u8.astype(np.float32)
    fill_value = float(np.median(gray[valid_mask]))
    filled = np.where(valid_mask, gray, fill_value)
    gy, gx = np.gradient(filled)
    grad_mag_sq = gx ** 2 + gy ** 2

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        lap_var = 0.0
    else:
        center = filled[1:-1, 1:-1]
        lap = (
            filled[:-2, 1:-1]
            + filled[2:, 1:-1]
            + filled[1:-1, :-2]
            + filled[1:-1, 2:]
            - 4.0 * center
        )
        lap_valid = valid_mask[1:-1, 1:-1]
        lap_var = float(np.var(lap[lap_valid])) if np.any(lap_valid) else 0.0

    valid_grad = grad_mag_sq[valid_mask]
    tenengrad = float(np.mean(valid_grad)) if valid_grad.size else 0.0
    return {
        "laplacian_variance": lap_var,
        "tenengrad": tenengrad,
    }


def _estimate_exposure_and_shadow(rgb_u8: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    if rgb_u8.size == 0 or not np.any(valid_mask):
        return {}
    rgb = rgb_u8.astype(np.float32) / 255.0
    brightness = np.max(rgb, axis=2)
    valid_brightness = brightness[valid_mask]
    if valid_brightness.size == 0:
        return {}
    over_ratio = float(np.mean(valid_brightness >= 0.98))
    under_ratio = float(np.mean(valid_brightness <= 0.05))
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    threshold = min(0.35, float(np.quantile(valid_brightness, 0.20)) + 0.05)
    shadow_mask = valid_mask & (brightness <= threshold) & (chroma <= 0.35)
    shadow_ratio = float(np.mean(shadow_mask[valid_mask])) if np.any(valid_mask) else 0.0
    return {
        "brightness_mean": float(np.mean(valid_brightness)),
        "brightness_std": float(np.std(valid_brightness)),
        "overexposed_ratio": over_ratio,
        "underexposed_ratio": under_ratio,
        "shadow_ratio_estimate": shadow_ratio,
        "shadow_threshold": threshold,
    }


def _estimate_stripe_noise(gray_u8: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    if gray_u8.size == 0 or not np.any(valid_mask):
        return {}

    gray = gray_u8.astype(np.float32)

    def _directional_score(axis: int) -> float:
        if axis == 0:
            sums = np.sum(np.where(valid_mask, gray, 0.0), axis=1)
            counts = np.sum(valid_mask, axis=1)
        else:
            sums = np.sum(np.where(valid_mask, gray, 0.0), axis=0)
            counts = np.sum(valid_mask, axis=0)
        ok = counts > 0
        if np.count_nonzero(ok) < 8:
            return 0.0
        profile = sums[ok] / counts[ok]
        window = max(5, min(31, (len(profile) // 8) * 2 + 1))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smooth = np.convolve(profile, kernel, mode="same")
        residual = profile - smooth
        return float(np.std(residual) / (np.std(profile) + 1e-6))

    row_score = _directional_score(axis=0)
    col_score = _directional_score(axis=1)
    overall = max(row_score, col_score)
    direction = "row" if row_score > col_score * 1.1 else "column" if col_score > row_score * 1.1 else "mixed"
    return {
        "stripe_noise_score": float(overall),
        "stripe_noise_row_score": float(row_score),
        "stripe_noise_col_score": float(col_score),
        "stripe_noise_direction": direction,
    }


def _estimate_color_cast(rgb_u8: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    if rgb_u8.size == 0 or not np.any(valid_mask):
        return {}
    rgb = rgb_u8.astype(np.float32) / 255.0
    means = [float(np.mean(rgb[..., idx][valid_mask])) for idx in range(3)]
    overall_mean = float(np.mean(means)) if means else 0.0
    if overall_mean <= 1e-6:
        cast_score = 0.0
    else:
        cast_score = float(max(abs(ch - overall_mean) / overall_mean for ch in means))
    dominant_index = int(np.argmax(means)) if means else 0
    return {
        "color_cast_score": cast_score,
        "channel_means": {"r": means[0], "g": means[1], "b": means[2]},
        "dominant_channel": ["r", "g", "b"][dominant_index],
    }


def estimate_quality(rgb_u8: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    if rgb_u8.size == 0:
        return {}
    gray_u8 = rgb_to_gray_uint8(rgb_u8)
    quality: dict[str, Any] = {}
    quality.update(_estimate_blur(gray_u8, valid_mask))
    quality.update(_estimate_exposure_and_shadow(rgb_u8, valid_mask))
    quality.update(_estimate_stripe_noise(gray_u8, valid_mask))
    quality.update(_estimate_color_cast(rgb_u8, valid_mask))
    return quality
