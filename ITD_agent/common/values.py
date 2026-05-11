from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
    except Exception:
        return default
    return default if math.isnan(result) else result


def normalize_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)
