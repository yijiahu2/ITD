from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from ITD_agent.common.values import safe_float as _safe_float


def _detect_score_field(gdf: gpd.GeoDataFrame, configured: str | None = None) -> tuple[str | None, str]:
    if configured and configured in gdf.columns:
        return configured, "configured"
    for candidate in ["score", "confidence", "conf", "prob", "probability"]:
        if candidate in gdf.columns:
            return candidate, "detected"
    return None, "constant_one"


def _ensure_projected_common_crs(pred_gdf: gpd.GeoDataFrame, gt_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if pred_gdf.crs is None and gt_gdf.crs is None:
        raise ValueError("Predicted and ground-truth vectors both have no CRS.")
    if pred_gdf.crs is None:
        pred_gdf = pred_gdf.set_crs(gt_gdf.crs)
    if gt_gdf.crs is None:
        gt_gdf = gt_gdf.set_crs(pred_gdf.crs)
    if pred_gdf.crs != gt_gdf.crs:
        gt_gdf = gt_gdf.to_crs(pred_gdf.crs)
    if not pred_gdf.crs or not pred_gdf.crs.is_projected:
        target_crs = pred_gdf.estimate_utm_crs() or gt_gdf.estimate_utm_crs()
        if target_crs:
            pred_gdf = pred_gdf.to_crs(target_crs)
            gt_gdf = gt_gdf.to_crs(target_crs)
    return pred_gdf, gt_gdf


def _build_candidate_rows(
    pred_gdf: gpd.GeoDataFrame,
    gt_gdf: gpd.GeoDataFrame,
    score_field: str | None,
) -> list[dict[str, Any]]:
    gt_sindex = gt_gdf.sindex
    candidate_rows: list[dict[str, Any]] = []
    for pred_idx, pred_row in pred_gdf.iterrows():
        pred_geom = pred_row.geometry
        if pred_geom is None or pred_geom.is_empty:
            continue
        pred_area = float(pred_geom.area)
        score = float(pred_row[score_field]) if score_field and score_field in pred_row and _safe_float(pred_row[score_field]) is not None else 1.0
        bbox_hits = list(gt_sindex.intersection(pred_geom.bounds))
        matches: list[dict[str, Any]] = []
        for gt_idx in bbox_hits:
            gt_row = gt_gdf.iloc[int(gt_idx)]
            gt_geom = gt_row.geometry
            if gt_geom is None or gt_geom.is_empty:
                continue
            inter_area = float(pred_geom.intersection(gt_geom).area)
            if inter_area <= 0.0:
                continue
            union_area = float(pred_geom.union(gt_geom).area)
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou <= 0.0:
                continue
            matches.append(
                {
                    "gt_idx": int(gt_idx),
                    "iou": float(iou),
                    "gt_area": float(gt_geom.area),
                    "inter_area": inter_area,
                    "intersection_over_gt_area": float(inter_area / max(float(gt_geom.area), 1.0e-6)),
                }
            )
        matches.sort(key=lambda item: (-float(item["iou"]), int(item["gt_idx"])))
        candidate_rows.append(
            {
                "pred_idx": int(pred_idx),
                "score": float(score),
                "pred_area": pred_area,
                "matches": matches,
            }
        )
    candidate_rows.sort(key=lambda item: (-float(item["score"]), int(item["pred_idx"])))
    return candidate_rows


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    diff_idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[diff_idx + 1] - mrec[diff_idx]) * mpre[diff_idx + 1]))


def _compute_area_r2(tp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not tp_rows:
        return {
            "num_matched_crowns": 0,
            "mae": None,
            "rmse": None,
            "rmse_percent": None,
            "r2": None,
            "area_regression_unreliable_flag": True,
        }
    gt_area = np.array([float(row["matched_gt_area"]) for row in tp_rows], dtype=np.float64)
    pred_area = np.array([float(row["pred_area"]) for row in tp_rows], dtype=np.float64)
    diff = gt_area - pred_area
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    rmse_ratio = float(np.sqrt(np.mean(np.square(diff / np.clip(gt_area, 1e-6, None)))))
    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(gt_area - gt_area.mean())))
    r2 = None if len(gt_area) < 2 or ss_tot <= 1e-12 else float(1.0 - (ss_res / ss_tot))
    return {
        "num_matched_crowns": int(len(tp_rows)),
        "mae": mae,
        "rmse": rmse,
        "rmse_percent": float(rmse_ratio * 100.0),
        "r2": r2,
        "area_regression_unreliable_flag": bool(len(tp_rows) < 5),
    }


def _mean_iou_matched(tp_rows: list[dict[str, Any]]) -> float | None:
    if not tp_rows:
        return None
    return float(np.mean([float(row["best_iou"]) for row in tp_rows]))


def _f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return float(2.0 * precision * recall / denominator)


def _compute_pr_ap(candidate_rows: list[dict[str, Any]], total_gt: int, iou_thr: float) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for pred in candidate_rows:
        best_match = None
        for item in pred["matches"]:
            if item["gt_idx"] in matched_gt:
                continue
            best_match = item
            break
        is_tp = bool(best_match and float(best_match["iou"]) >= float(iou_thr))
        if is_tp:
            matched_gt.add(int(best_match["gt_idx"]))
        matches.append(
            {
                "pred_idx": int(pred["pred_idx"]),
                "score": float(pred["score"]),
                "pred_area": float(pred["pred_area"]),
                "best_iou": float(best_match["iou"]) if best_match else 0.0,
                "matched_gt_idx": int(best_match["gt_idx"]) if is_tp and best_match else None,
                "matched_gt_area": float(best_match["gt_area"]) if is_tp and best_match else None,
                "is_tp": bool(is_tp),
                "is_fp": not bool(is_tp),
            }
        )
    if not matches:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": int(total_gt),
            "crown_area": _compute_area_r2([]),
            "tp_rows": [],
            "mean_iou_matched": None,
        }
    tp = np.array([1 if row["is_tp"] else 0 for row in matches], dtype=np.int32)
    fp = np.array([1 if row["is_fp"] else 0 for row in matches], dtype=np.int32)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    final_tp = int(tp_cum[-1]) if tp_cum.size else 0
    final_fp = int(fp_cum[-1]) if fp_cum.size else 0
    final_fn = int(max(total_gt - final_tp, 0))
    tp_rows = [row for row in matches if row["is_tp"]]
    return {
        "ap": float(_ap_from_pr(recalls, precisions)),
        "precision": float(final_tp / max(final_tp + final_fp, 1)),
        "recall": float(final_tp / max(total_gt, 1)),
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
        "crown_area": _compute_area_r2(tp_rows),
        "tp_rows": tp_rows,
        "mean_iou_matched": _mean_iou_matched(tp_rows),
    }


def _compute_error_decomposition(
    *,
    candidate_rows: list[dict[str, Any]],
    pred_count: int,
    gt_count: int,
    iou_050: dict[str, Any],
    overlap_ratio_thr: float = 0.10,
) -> dict[str, Any]:
    pred_multi_gt = 0
    gt_multi_pred: dict[int, int] = {}
    for pred in candidate_rows:
        valid_matches = [item for item in pred["matches"] if float(item.get("intersection_over_gt_area") or 0.0) >= float(overlap_ratio_thr)]
        unique_gt = {int(item["gt_idx"]) for item in valid_matches}
        if len(unique_gt) > 1:
            pred_multi_gt += 1
        for item in valid_matches:
            gt_idx = int(item["gt_idx"])
            gt_multi_pred[gt_idx] = int(gt_multi_pred.get(gt_idx, 0)) + 1

    under_segmentation_score = 0.0 if pred_count <= 0 else float(pred_multi_gt / pred_count)
    over_segmentation_score = 0.0 if gt_count <= 0 else float(sum(1 for count in gt_multi_pred.values() if count > 1) / gt_count)
    miss_detection_score = 0.0 if gt_count <= 0 else float(iou_050["fn"] / gt_count)
    false_detection_score = 0.0 if pred_count <= 0 else float(iou_050["fp"] / pred_count)
    sorted_scores = sorted(
        [under_segmentation_score, over_segmentation_score, miss_detection_score, false_detection_score],
        reverse=True,
    )
    top = sorted_scores[0] if sorted_scores else 0.0
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    failure_severity = float(top)
    failure_pattern_confidence = float(min(max(top - second, 0.0), 1.0))
    return {
        "under_segmentation_score": under_segmentation_score,
        "over_segmentation_score": over_segmentation_score,
        "miss_detection_score": miss_detection_score,
        "false_detection_score": false_detection_score,
        "failure_severity": failure_severity,
        "failure_pattern_confidence": failure_pattern_confidence,
        "overlap_ratio_threshold": float(overlap_ratio_thr),
    }


def evaluate_benchmark_vector_result(
    *,
    pred_shp: str,
    gt_shp: str,
    score_field: str | None = None,
    error_overlap_ratio_thr: float = 0.10,
) -> dict[str, Any]:
    pred_gdf = gpd.read_file(pred_shp)
    gt_gdf = gpd.read_file(gt_shp)
    pred_gdf = pred_gdf[pred_gdf.geometry.notnull() & (~pred_gdf.geometry.is_empty)].copy()
    gt_gdf = gt_gdf[gt_gdf.geometry.notnull() & (~gt_gdf.geometry.is_empty)].copy()
    pred_gdf, gt_gdf = _ensure_projected_common_crs(pred_gdf, gt_gdf)
    resolved_score_field, score_source = _detect_score_field(pred_gdf, configured=score_field)
    candidate_rows = _build_candidate_rows(pred_gdf, gt_gdf, resolved_score_field)
    total_gt = int(len(gt_gdf))
    total_pred = int(len(pred_gdf))
    iou_050 = _compute_pr_ap(candidate_rows, total_gt=total_gt, iou_thr=0.50)
    iou_075 = _compute_pr_ap(candidate_rows, total_gt=total_gt, iou_thr=0.75)
    ap_thresholds = [round(value, 2) for value in np.arange(0.50, 1.00, 0.05)]
    ap_by_threshold = {f"{thr:.2f}": _compute_pr_ap(candidate_rows, total_gt=total_gt, iou_thr=thr)["ap"] for thr in ap_thresholds}
    error_decomposition = _compute_error_decomposition(
        candidate_rows=candidate_rows,
        pred_count=total_pred,
        gt_count=total_gt,
        iou_050=iou_050,
        overlap_ratio_thr=error_overlap_ratio_thr,
    )
    return {
        "evaluation_mode": "benchmark",
        "prediction_file": str(pred_shp),
        "ground_truth_file": str(gt_shp),
        "num_predictions": total_pred,
        "num_ground_truth": total_gt,
        "score_field": resolved_score_field,
        "score_source": score_source,
        "precision": iou_050["precision"],
        "recall": iou_050["recall"],
        "ap_50_95": float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0,
        "ap_by_threshold": ap_by_threshold,
        "ap50": iou_050["ap"],
        "ap75": iou_075["ap"],
        "f1_score50": _f1_score(iou_050["precision"], iou_050["recall"]),
        "mean_iou_matched": iou_050["mean_iou_matched"],
        "iou_0_75": {
            "precision": iou_075["precision"],
            "recall": iou_075["recall"],
        },
        "tp50": iou_050["tp"],
        "fp50": iou_050["fp"],
        "fn50": iou_050["fn"],
        "tp75": iou_075["tp"],
        "fp75": iou_075["fp"],
        "fn75": iou_075["fn"],
        "mae": iou_050["crown_area"]["mae"],
        "rmse": iou_050["crown_area"]["rmse"],
        "rmse_percent": iou_050["crown_area"]["rmse_percent"],
        "r2": iou_050["crown_area"]["r2"],
        "num_matched_crowns": iou_050["crown_area"]["num_matched_crowns"],
        "area_regression_unreliable_flag": iou_050["crown_area"]["area_regression_unreliable_flag"],
        "crown_area_iou_0_50": iou_050["crown_area"],
        "crown_area_iou_0_75": iou_075["crown_area"],
        "error_decomposition": error_decomposition,
    }
