from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from ITD_agent.segmentation.finetuning.io_utils import normalize_details_df


def _safe_float(v: Any) -> float | None:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def summarize_details_csv(
    details_csv_path: str,
    top_k: int = 3,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(details_csv_path)
    if not path.exists():
        return {"exists": False, "top_k_xiaoban": []}

    try:
        raw_df = pd.read_csv(path)
    except EmptyDataError:
        return {"exists": True, "num_units": 0, "top_k_xiaoban": []}
    if raw_df.empty:
        return {"exists": True, "num_units": 0, "top_k_xiaoban": []}

    if cfg is not None:
        df, _, _, _ = normalize_details_df(raw_df, cfg)
    else:
        df = raw_df.copy()

    for col in [
        "xiaoban_id",
        "XBH",
        "tree_count_error_abs",
        "mean_crown_width_error_abs",
        "closure_error_abs",
        "density_error_abs",
        "pred_tree_count",
        "pred_mean_crown_width",
        "pred_cover_ratio",
        "pred_density_trees_per_ha",
        "expected_tree_count",
        "expected_mean_crown_width",
        "expected_closure",
        "expected_density",
        "landform_type",
        "slope_class",
        "aspect_class",
        "slope_position_class",
        "mean_slope",
        "relief_elev",
    ]:
        if col not in df.columns:
            df[col] = None

    if "xiaoban_id" not in df.columns and "XBH" in df.columns:
        df["xiaoban_id"] = df["XBH"]

    def row_score(row: pd.Series) -> float:
        from ITD_agent.evaluation_analysis.reference_quality_engine import build_reference_score_breakdown

        pred_tree = _safe_float(row.get("pred_tree_count"))
        expected_tree = _safe_float(row.get("expected_tree_count"))
        pred_crown = _safe_float(row.get("pred_mean_crown_width"))
        expected_crown = _safe_float(row.get("expected_mean_crown_width"))
        metrics = {
            "tree_count_error_ratio": (
                abs(pred_tree - expected_tree) / max(abs(expected_tree), 1.0e-6)
                if pred_tree is not None and expected_tree not in (None, 0.0)
                else None
            ),
            "mean_crown_width_error_ratio": (
                abs(pred_crown - expected_crown) / max(abs(expected_crown), 1.0e-6)
                if pred_crown is not None and expected_crown not in (None, 0.0)
                else None
            ),
            "closure_error_abs": _safe_float(row.get("closure_error_abs")),
            "density_error_abs": _safe_float(row.get("density_error_abs")),
            "expected_density": _safe_float(row.get("expected_density")),
        }
        score = _safe_float(build_reference_score_breakdown(metrics, cfg=cfg).get("score"))
        return float(score or 0.0)

    df["error_score"] = df.apply(row_score, axis=1)
    ranked = df.sort_values("error_score", ascending=False)
    top_rows: list[dict[str, Any]] = []
    for _, row in ranked.head(max(int(top_k), 0)).iterrows():
        top_rows.append(
            {
                "xiaoban_id": str(row.get("xiaoban_id") or row.get("XBH")),
                "pred_tree_count": _safe_float(row.get("pred_tree_count")),
                "expected_tree_count": _safe_float(row.get("expected_tree_count")),
                "tree_count_error_abs": _safe_float(row.get("tree_count_error_abs")),
                "pred_mean_crown_width": _safe_float(row.get("pred_mean_crown_width")),
                "expected_mean_crown_width": _safe_float(row.get("expected_mean_crown_width")),
                "mean_crown_width_error_abs": _safe_float(row.get("mean_crown_width_error_abs")),
                "pred_cover_ratio": _safe_float(row.get("pred_cover_ratio")),
                "expected_closure": _safe_float(row.get("expected_closure")),
                "closure_error_abs": _safe_float(row.get("closure_error_abs")),
                "pred_density_trees_per_ha": _safe_float(row.get("pred_density_trees_per_ha")),
                "expected_density": _safe_float(row.get("expected_density")),
                "density_error_abs": _safe_float(row.get("density_error_abs")),
                "landform_type": row.get("landform_type"),
                "slope_class": row.get("slope_class"),
                "aspect_class": row.get("aspect_class"),
                "slope_position_class": row.get("slope_position_class"),
                "mean_slope": _safe_float(row.get("mean_slope")),
                "relief_elev": _safe_float(row.get("relief_elev")),
                "error_score": _safe_float(row.get("error_score")),
            }
        )

    return {
        "exists": True,
        "num_units": int(len(df)),
        "top_k_xiaoban": top_rows,
    }
