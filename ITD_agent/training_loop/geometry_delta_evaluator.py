from __future__ import annotations

from typing import Any


def compute_geometry_delta(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "tree_count_error_ratio",
        "mean_crown_width_error_ratio",
        "closure_error_abs",
        "density_error_abs",
    ]
    delta = {}
    for key in keys:
        before = _float_or_none(baseline_metrics.get(key))
        after = _float_or_none(candidate_metrics.get(key))
        delta[key + "_delta"] = None if before is None or after is None else after - before
    return {
        "status": "computed",
        "delta": delta,
        "lower_is_better": keys,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
