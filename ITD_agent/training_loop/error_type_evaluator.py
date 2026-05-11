from __future__ import annotations

from typing import Any


def compute_error_type_delta(
    *,
    baseline_errors: dict[str, Any],
    candidate_errors: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "under_segmentation",
        "over_segmentation",
        "false_positive",
        "false_negative",
        "small_crown_recall",
        "large_crown_split",
    ]
    delta = {}
    for key in keys:
        before = _to_float(baseline_errors.get(key))
        after = _to_float(candidate_errors.get(key))
        if before is None or after is None:
            delta[key + "_delta"] = None
        else:
            delta[key + "_delta"] = after - before
    return {
        "status": "computed",
        "delta": delta,
        "interpretation": _interpret(delta),
    }


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _interpret(delta: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key, value in delta.items():
        if value is None:
            out[key] = "not_available"
        elif value < 0:
            out[key] = "improved"
        elif value > 0:
            out[key] = "regressed"
        else:
            out[key] = "unchanged"
    return out
