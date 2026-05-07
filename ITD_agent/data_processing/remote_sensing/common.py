from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


def downsample(arr: np.ndarray, max_dim: int = 512) -> np.ndarray:
    if arr.ndim != 2:
        return arr
    h, w = arr.shape
    step_y = max(h // max_dim, 1)
    step_x = max(w // max_dim, 1)
    return arr[::step_y, ::step_x]


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def normalize_to_uint8(arr: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    data = arr.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data, dtype=np.uint8)
    valid_values = data[valid_mask]
    max_value = float(np.nanmax(valid_values))
    if max_value <= 255:
        scale = 255.0
    elif max_value <= 1023:
        scale = 1023.0
    elif max_value <= 4095:
        scale = 4095.0
    elif max_value <= 16383:
        scale = 16383.0
    else:
        scale = 65535.0
    normalized = np.clip(data / max(scale, 1.0), 0.0, 1.0) * 255.0
    return normalized.astype(np.uint8)


def rgb_to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def as_rgb_uint8(arr: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("Expected CHW raster array.")
    channels = [normalize_to_uint8(arr[idx], valid_mask) for idx in range(min(arr.shape[0], 3))]
    while len(channels) < 3:
        channels.append(channels[-1].copy())
    return np.stack(channels[:3], axis=-1)


def read_raster_window(
    path: str,
    *,
    indexes: list[int],
    window: list[int],
    boundless: bool = False,
    fill_value: float | int | None = None,
) -> tuple[np.ndarray, rasterio.Affine, Any, float | int | None]:
    x, y, width, height = [int(v) for v in window]
    with rasterio.open(path) as src:
        arr = src.read(
            indexes=indexes,
            window=Window(x, y, width, height),
            boundless=boundless,
            fill_value=fill_value,
        )
        transform = src.window_transform(Window(x, y, width, height))
        return arr, transform, src.crs, src.nodata


def read_mask_window(path: str | None, *, window: list[int], boundless: bool = False, fill_value: int = 0) -> np.ndarray | None:
    if not path:
        return None
    x, y, width, height = [int(v) for v in window]
    # valid_mask 是与 working_dom 同范围、同 transform、同宽高的全图单文件；
    # 后续 block/tile 统一按窗口逻辑读取，不再物理切成独立 mask 块。
    with rasterio.open(path) as src:
        return src.read(
            1,
            window=Window(x, y, width, height),
            boundless=boundless,
            fill_value=fill_value,
        )


def affine_from_list(values: list[float]) -> Affine:
    if len(values) != 6:
        raise ValueError("Transform list must contain 6 elements.")
    return Affine(*[float(v) for v in values])


def valid_mask_from_data(
    data: np.ndarray,
    *,
    nodata: float | int | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if data.ndim == 3:
        mask = np.all(np.isfinite(data), axis=0)
        if nodata is not None:
            mask &= ~np.any(np.isclose(data, float(nodata)), axis=0)
    else:
        mask = np.isfinite(data)
        if nodata is not None:
            mask &= ~np.isclose(data, float(nodata))
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)
    return mask


def coarse_grid_slices(height: int, width: int, rows: int, cols: int) -> list[tuple[slice, slice]]:
    row_edges = np.linspace(0, height, rows + 1, dtype=int)
    col_edges = np.linspace(0, width, cols + 1, dtype=int)
    slices: list[tuple[slice, slice]] = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            slices.append((slice(row_edges[row_idx], row_edges[row_idx + 1]), slice(col_edges[col_idx], col_edges[col_idx + 1])))
    return slices
