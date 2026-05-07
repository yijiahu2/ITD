from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .benchmark_engine import evaluate_benchmark_vector_result
from .contracts import FinetuneEffectAssessment
from .flow_decisions import build_finetune_effect_flow_decision


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dump_json(data: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _first_existing(paths: list[str | Path | None]) -> str | None:
    for item in paths:
        if not item:
            continue
        p = Path(str(item))
        if p.exists():
            return str(p.resolve())
    return None


def _resolve_tree_crowns_from_output_dir(output_dir: str | Path | None) -> str | None:
    if not output_dir:
        return None
    root = Path(str(output_dir))
    return _first_existing(
        [
            root / "final_outputs" / "tree_crowns.shp",
            root / "tree_crowns.shp",
            root / "Y_inst.shp",
        ]
    )


def _resolve_gt_shp_from_cfg(cfg: dict[str, Any]) -> str | None:
    benchmark_cfg = ((cfg.get("evaluation") or {}).get("final_report") or {}).get("benchmark") or {}
    return _first_existing(
        [
            benchmark_cfg.get("gt_tree_crowns_shp"),
            benchmark_cfg.get("ground_truth_shp"),
            benchmark_cfg.get("gt_shp"),
        ]
    )


def _resolve_from_config_and_csv(
    *,
    config_path: str | None,
    before_csv: str,
    after_csv: str,
) -> tuple[str | None, str | None, str | None]:
    cfg = _load_yaml(config_path) if config_path and Path(config_path).exists() else {}
    before_pred = _resolve_tree_crowns_from_output_dir(cfg.get("base_output_dir") or cfg.get("original_output_dir") or cfg.get("output_dir"))
    after_pred = _resolve_tree_crowns_from_output_dir(cfg.get("output_dir"))

    before_parent = Path(before_csv).resolve().parent
    after_parent = Path(after_csv).resolve().parent
    before_pred = before_pred or _first_existing(
        [
            before_parent / "final_outputs" / "tree_crowns.shp",
            before_parent / "tree_crowns.shp",
            before_parent / "Y_inst.shp",
        ]
    )
    after_pred = after_pred or _first_existing(
        [
            after_parent / "final_outputs" / "tree_crowns.shp",
            after_parent / "tree_crowns.shp",
            after_parent / "Y_inst.shp",
        ]
    )

    gt_shp = _resolve_gt_shp_from_cfg(cfg)
    base_cfg_path = cfg.get("base_config")
    if not gt_shp and base_cfg_path and Path(str(base_cfg_path)).exists():
        base_cfg = _load_yaml(base_cfg_path)
        gt_shp = _resolve_gt_shp_from_cfg(base_cfg)
        before_pred = before_pred or _resolve_tree_crowns_from_output_dir(base_cfg.get("output_dir"))
    return before_pred, after_pred, gt_shp


def _build_benchmark_gain(
    *,
    before_pred_shp: str,
    after_pred_shp: str,
    gt_shp: str,
    score_field: str | None = None,
) -> dict[str, Any]:
    before = evaluate_benchmark_vector_result(pred_shp=before_pred_shp, gt_shp=gt_shp, score_field=score_field)
    after = evaluate_benchmark_vector_result(pred_shp=after_pred_shp, gt_shp=gt_shp, score_field=score_field)
    metric_keys = ["precision", "recall", "ap50", "ap75", "mae", "rmse", "rmse_percent", "r2"]
    delta: dict[str, Any] = {}
    for key in metric_keys:
        before_v = before.get(key)
        after_v = after.get(key)
        if before_v is None or after_v is None:
            delta[key] = None
            continue
        raw_delta = float(after_v) - float(before_v)
        if key in {"mae", "rmse", "rmse_percent"}:
            raw_delta = float(before_v) - float(after_v)
        delta[key] = raw_delta
    return {
        "evaluation_mode": "benchmark",
        "ground_truth_file": gt_shp,
        "before_prediction_file": before_pred_shp,
        "after_prediction_file": after_pred_shp,
        "before": before,
        "after": after,
        "delta": delta,
    }


def _build_compare_flags(summary: dict[str, Any]) -> dict[str, Any]:
    gain_keys = ["mean_gain_tree_count", "mean_gain_crown", "mean_gain_closure", "mean_gain_density"]
    gains = [summary.get(key) for key in gain_keys if summary.get(key) is not None]
    positive_count = sum(1 for value in gains if float(value) > 0)
    negative_count = sum(1 for value in gains if float(value) < 0)
    benchmark_delta = ((summary.get("benchmark_gain") or {}).get("delta") or {})
    ap50_delta = benchmark_delta.get("ap50")
    regression_flag = bool(
        (ap50_delta is not None and float(ap50_delta) < -0.01)
        or (negative_count >= 2 and negative_count > positive_count)
    )
    accepted_improvement_flag = bool(
        not regression_flag
        and (
            (ap50_delta is not None and float(ap50_delta) > 0.01)
            or (positive_count >= 2 and positive_count >= negative_count)
        )
    )
    return {
        "accepted_improvement_flag": accepted_improvement_flag,
        "regression_flag": regression_flag,
    }


def build_finetune_recommendation(
    cfg: dict[str, Any],
    *,
    metrics: dict[str, Any],
    details_csv: str | None,
    roi_round_count: int,
) -> dict[str, Any]:
    from ITD_agent.finetune_pool.recommendation import build_finetune_recommendation as _impl

    return _impl(cfg, metrics=metrics, details_csv=details_csv, roi_round_count=roi_round_count)


def compare_finetune_effect(
    *,
    before_csv: str,
    after_csv: str,
    out_dir: str,
    join_col: str = "reference_unit_id",
    before_pred_shp: str | None = None,
    after_pred_shp: str | None = None,
    gt_shp: str | None = None,
    score_field: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    out_root = _ensure_dir(out_dir)
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)
    if join_col not in before.columns and join_col == "reference_unit_id" and "xiaoban_id" in before.columns:
        before = before.rename(columns={"xiaoban_id": "reference_unit_id"})
    if join_col not in after.columns and join_col == "reference_unit_id" and "xiaoban_id" in after.columns:
        after = after.rename(columns={"xiaoban_id": "reference_unit_id"})
    if join_col not in before.columns:
        raise RuntimeError(f"before_csv 缺少主键列 {join_col}，实际列为: {list(before.columns)}")
    if join_col not in after.columns:
        raise RuntimeError(f"after_csv 缺少主键列 {join_col}，实际列为: {list(after.columns)}")

    before_keep = [
        join_col,
        "tree_count_error_abs",
        "mean_crown_width_error_abs",
        "closure_error_abs",
        "density_error_abs",
        "mean_slope",
        "relief_elev",
        "dominant_aspect_class",
        "landform_type",
        "slope_class",
        "aspect_class",
        "slope_position_class",
    ]
    after_keep = [
        join_col,
        "tree_count_error_abs",
        "mean_crown_width_error_abs",
        "closure_error_abs",
        "density_error_abs",
    ]
    before = before[[c for c in before_keep if c in before.columns]].copy()
    after = after[[c for c in after_keep if c in after.columns]].copy()
    merged = before.merge(after, on=join_col, suffixes=("_before", "_after"))

    for before_col, after_col, out_col in [
        ("tree_count_error_abs_before", "tree_count_error_abs_after", "gain_tree_count"),
        ("mean_crown_width_error_abs_before", "mean_crown_width_error_abs_after", "gain_crown"),
        ("closure_error_abs_before", "closure_error_abs_after", "gain_closure"),
        ("density_error_abs_before", "density_error_abs_after", "gain_density"),
    ]:
        if before_col in merged.columns and after_col in merged.columns:
            merged[out_col] = merged[before_col] - merged[after_col]

    compare_csv = out_root / "finetune_compare.csv"
    merged.to_csv(compare_csv, index=False)

    terrain_group_cols = [c for c in ["landform_type", "slope_class", "aspect_class", "slope_position_class"] if c in merged.columns]
    stratified_gain: list[dict[str, Any]] = []
    if terrain_group_cols:
        grouped = merged.groupby(terrain_group_cols, dropna=False)
        for key, sub in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_dict = {terrain_group_cols[i]: key_tuple[i] for i in range(len(terrain_group_cols))}
            stratified_gain.append(
                {
                    **key_dict,
                    "num_samples": int(len(sub)),
                    "mean_gain_tree_count": float(sub["gain_tree_count"].mean()) if "gain_tree_count" in sub.columns else None,
                    "mean_gain_crown": float(sub["gain_crown"].mean()) if "gain_crown" in sub.columns else None,
                    "mean_gain_closure": float(sub["gain_closure"].mean()) if "gain_closure" in sub.columns else None,
                    "mean_gain_density": float(sub["gain_density"].mean()) if "gain_density" in sub.columns else None,
                }
            )

    summary = {
        "num_compared": int(len(merged)),
        "join_col": join_col,
        "mean_gain_tree_count": float(merged["gain_tree_count"].mean()) if "gain_tree_count" in merged.columns and len(merged) > 0 else None,
        "mean_gain_crown": float(merged["gain_crown"].mean()) if "gain_crown" in merged.columns and len(merged) > 0 else None,
        "mean_gain_closure": float(merged["gain_closure"].mean()) if "gain_closure" in merged.columns and len(merged) > 0 else None,
        "mean_gain_density": float(merged["gain_density"].mean()) if "gain_density" in merged.columns and len(merged) > 0 else None,
        "num_tree_improved": int((merged["gain_tree_count"] > 0).sum()) if "gain_tree_count" in merged.columns else None,
        "num_crown_improved": int((merged["gain_crown"] > 0).sum()) if "gain_crown" in merged.columns else None,
        "num_closure_improved": int((merged["gain_closure"] > 0).sum()) if "gain_closure" in merged.columns else None,
        "num_density_improved": int((merged["gain_density"] > 0).sum()) if "gain_density" in merged.columns else None,
        "terrain_group_cols": terrain_group_cols,
        "stratified_gain": stratified_gain,
    }

    resolved_before_pred = before_pred_shp
    resolved_after_pred = after_pred_shp
    resolved_gt_shp = gt_shp
    if not (resolved_before_pred and resolved_after_pred and resolved_gt_shp):
        auto_before, auto_after, auto_gt = _resolve_from_config_and_csv(
            config_path=config_path,
            before_csv=before_csv,
            after_csv=after_csv,
        )
        resolved_before_pred = resolved_before_pred or auto_before
        resolved_after_pred = resolved_after_pred or auto_after
        resolved_gt_shp = resolved_gt_shp or auto_gt

    if resolved_before_pred and resolved_after_pred and resolved_gt_shp:
        summary["benchmark_gain"] = _build_benchmark_gain(
            before_pred_shp=resolved_before_pred,
            after_pred_shp=resolved_after_pred,
            gt_shp=resolved_gt_shp,
            score_field=score_field,
        )
    else:
        summary["benchmark_gain"] = {
            "evaluation_mode": "benchmark_unavailable",
            "message": "缺少 before/after 预测树冠 shp 或 GT 树冠 shp，无法计算微调 benchmark 增益。",
            "before_prediction_file": resolved_before_pred,
            "after_prediction_file": resolved_after_pred,
            "ground_truth_file": resolved_gt_shp,
        }

    summary.update(_build_compare_flags(summary))
    summary_path = out_root / "finetune_gain_summary.json"
    flow_decision = build_finetune_effect_flow_decision(summary)
    summary["flow_decision"] = flow_decision
    _dump_json(summary, summary_path)
    result = FinetuneEffectAssessment(
        summary_json=str(summary_path),
        compare_csv=str(compare_csv),
        summary=summary,
    ).to_dict()
    result["flow_decision"] = flow_decision
    return result
