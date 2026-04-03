from __future__ import annotations

from typing import Any

import numpy as np


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
