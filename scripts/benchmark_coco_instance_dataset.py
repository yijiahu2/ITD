from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from pycocotools import mask as mask_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.segmentation.finetuning.io_utils import dump_csv, dump_json, ensure_dir, load_yaml
from ITD_agent.segmentation.coco_utils import (
    VALID_IMAGE_SUFFIXES,
    build_image_index as _build_image_index,
    collect_coco_jsons as _collect_coco_jsons,
    load_merged_coco as _load_merged_coco,
    normalize_split_mapping as _normalize_split_mapping,
    normalize_str_list as _normalize_str_list,
    resolve_image_path as _resolve_image_path,
    segmentation_to_rle as _segmentation_to_rle,
)
from tools.cached_stage_runners import predict_semantic_prior_cached, run_segmentation_cached


def _safe_name(text: str) -> str:
    chars: list[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "item"


def _normalize_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return [float(x) for x in items] if items else default
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return default


def _normalize_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _rle_area(rle: dict[str, Any]) -> float:
    return float(mask_utils.area(rle))


def _rle_bbox(rle: dict[str, Any]) -> list[float]:
    return [float(x) for x in mask_utils.toBbox(rle).tolist()]


def _prepare_gt_records(coco: dict[str, Any], image_dir: Path) -> list[dict[str, Any]]:
    by_name, by_stem = _build_image_index(image_dir)
    ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_image[int(ann["image_id"])].append(ann)

    records: list[dict[str, Any]] = []
    for image in coco.get("images", []):
        image_id = int(image["id"])
        image_path = _resolve_image_path(str(image["file_name"]), by_name, by_stem, image_dir)
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            with rasterio.open(image_path) as src:
                width = int(src.width)
                height = int(src.height)

        gt_instances: list[dict[str, Any]] = []
        for ann in ann_by_image.get(image_id, []):
            rle = _segmentation_to_rle(ann["segmentation"], height, width)
            area = _rle_area(rle)
            if area <= 0:
                continue
            gt_instances.append(
                {
                    "gt_id": int(ann["id"]),
                    "rle": rle,
                    "bbox": _rle_bbox(rle),
                    "area": area,
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        records.append(
            {
                "image_id": image_id,
                "image_name": image_path.name,
                "image_stem": image_path.stem,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "gt_instances": gt_instances,
            }
        )

    return records


def _write_semantic_prior_outputs(pred: dict[str, Any], output_dir: Path, save_prob_tif: bool) -> dict[str, str]:
    mod = pred["module"]
    mask = pred["mask"]
    prob = pred["probability"]

    out_tif = output_dir / "M_sem.tif"
    out_png = output_dir / "M_sem.png"
    out_shp = output_dir / "M_sem.shp"
    mod.write_tif(str(out_tif), mask, pred["profile"])
    mod.mask_to_shp(
        mask=mask,
        transform=pred["transform"],
        crs=pred["crs"],
        out_shp=str(out_shp),
        min_area_m2=mod.MIN_AREA_M2,
        simplify_tol=mod.SIMPLIFY_TOL,
    )
    mod.cv2.imwrite(str(out_png), (mask * 255).astype(mod.np.uint8))

    outputs = {
        "m_sem_tif": str(out_tif),
        "m_sem_png": str(out_png),
        "m_sem_shp": str(out_shp),
    }

    if save_prob_tif:
        prob_tif = output_dir / "M_sem_prob.tif"
        profile = pred["tiff_profile_uint8"].copy()
        profile.update(dtype=rasterio.float32, nodata=0.0)
        with rasterio.open(prob_tif, "w", **profile) as dst:
            dst.write(prob.astype(np.float32), 1)
        outputs["m_sem_prob_tif"] = str(prob_tif)

    return outputs


def _read_instance_labels(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int32)


def _score_from_mask(score_map: np.ndarray | None, mask: np.ndarray, mode: str) -> float:
    if score_map is None or mode == "constant_one":
        return 1.0

    vals = score_map[mask]
    if vals.size == 0:
        return 0.0

    if mode == "semantic_prior_max_prob":
        score = float(vals.max())
    elif mode == "semantic_prior_median_prob":
        score = float(np.median(vals))
    else:
        score = float(vals.mean())

    return float(np.clip(score, 0.0, 1.0))


def _instances_from_label_image(
    label_image: np.ndarray,
    image_id: int,
    score_map: np.ndarray | None,
    score_mode: str,
) -> list[dict[str, Any]]:
    pred_instances: list[dict[str, Any]] = []
    unique_ids = np.unique(label_image)
    unique_ids = unique_ids[unique_ids > 0]

    for inst_id in unique_ids.tolist():
        mask = label_image == int(inst_id)
        area = int(mask.sum())
        if area <= 0:
            continue
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        pred_instances.append(
            {
                "pred_id": int(inst_id),
                "image_id": image_id,
                "score": _score_from_mask(score_map, mask, score_mode),
                "area": float(area),
                "bbox": _rle_bbox(rle),
                "rle": rle,
            }
        )

    pred_instances.sort(key=lambda x: (-float(x["score"]), int(x["pred_id"])))
    return pred_instances


def _match_predictions(
    preds: list[dict[str, Any]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> tuple[list[dict[str, Any]], int]:
    gt_matched: dict[int, list[bool]] = {
        image_id: [False] * len(gts)
        for image_id, gts in gt_by_image.items()
    }
    total_gt = sum(len(gts) for gts in gt_by_image.values())

    match_rows: list[dict[str, Any]] = []
    ordered_preds = sorted(preds, key=lambda x: (-float(x["score"]), int(x["image_id"]), int(x["pred_id"])))
    for pred in ordered_preds:
        image_id = int(pred["image_id"])
        gts = gt_by_image.get(image_id, [])
        best_iou = 0.0
        best_idx = -1

        if gts:
            ious = mask_utils.iou(
                [pred["rle"]],
                [gt["rle"] for gt in gts],
                [int(gt.get("iscrowd", 0)) for gt in gts],
            )[0]
            for idx, iou_val in enumerate(ious.tolist()):
                if gt_matched[image_id][idx]:
                    continue
                if float(iou_val) > best_iou:
                    best_iou = float(iou_val)
                    best_idx = idx

        is_tp = best_idx >= 0 and best_iou >= float(iou_thr)
        if is_tp:
            gt_matched[image_id][best_idx] = True
            matched_gt_id = int(gts[best_idx]["gt_id"])
            matched_gt_area = float(gts[best_idx]["area"])
        else:
            matched_gt_id = None
            matched_gt_area = None

        match_rows.append(
            {
                "image_id": image_id,
                "pred_id": int(pred["pred_id"]),
                "score": float(pred["score"]),
                "pred_area": float(pred["area"]),
                "best_iou": float(best_iou),
                "matched_gt_id": matched_gt_id,
                "matched_gt_area": matched_gt_area,
                "is_tp": bool(is_tp),
                "is_fp": not bool(is_tp),
            }
        )

    return match_rows, total_gt


def _compute_crown_area_metrics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    tp_rows = [row for row in matches if row["is_tp"] and row["matched_gt_area"] is not None]
    if not tp_rows:
        return {
            "num_matched_crowns": 0,
            "mae": None,
            "rmse": None,
            "rmse_ratio": None,
            "rmse_percent": None,
            "r2": None,
        }

    gt_area = np.array([float(row["matched_gt_area"]) for row in tp_rows], dtype=np.float64)
    pred_area = np.array([float(row["pred_area"]) for row in tp_rows], dtype=np.float64)
    diff = gt_area - pred_area

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    safe_gt = np.clip(gt_area, 1e-6, None)
    rmse_ratio = float(np.sqrt(np.mean(np.square(diff / safe_gt))))
    rmse_percent = float(rmse_ratio * 100.0)

    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(gt_area - gt_area.mean())))
    if len(gt_area) < 2 or ss_tot <= 1e-12:
        r2 = None
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    return {
        "num_matched_crowns": int(len(tp_rows)),
        "mae": mae,
        "rmse": rmse,
        "rmse_ratio": rmse_ratio,
        "rmse_percent": rmse_percent,
        "r2": r2,
    }


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    diff_idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[diff_idx + 1] - mrec[diff_idx]) * mpre[diff_idx + 1]))


def _compute_pr_ap(
    preds: list[dict[str, Any]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> dict[str, Any]:
    matches, total_gt = _match_predictions(preds, gt_by_image, iou_thr=iou_thr)
    if not matches:
        return {
            "total_gt": int(total_gt),
            "total_predictions": 0,
            "tp": 0,
            "fp": 0,
            "fn": int(total_gt),
            "precision": 0.0,
            "recall": 0.0,
            "ap": 0.0,
            "crown_area_metrics": _compute_crown_area_metrics([]),
            "pr_curve": [],
            "matches": [],
        }

    tp = np.array([1 if row["is_tp"] else 0 for row in matches], dtype=np.int32)
    fp = np.array([1 if row["is_fp"] else 0 for row in matches], dtype=np.int32)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = _ap_from_pr(recalls, precisions)

    final_tp = int(tp_cum[-1]) if tp_cum.size else 0
    final_fp = int(fp_cum[-1]) if fp_cum.size else 0
    final_fn = int(max(total_gt - final_tp, 0))
    final_precision = float(final_tp / max(final_tp + final_fp, 1))
    final_recall = float(final_tp / max(total_gt, 1))

    pr_curve = []
    for idx, row in enumerate(matches):
        pr_curve.append(
            {
                "rank": idx + 1,
                "score": float(row["score"]),
                "precision": float(precisions[idx]),
                "recall": float(recalls[idx]),
                "best_iou": float(row["best_iou"]),
                "is_tp": bool(row["is_tp"]),
            }
        )

    return {
        "total_gt": int(total_gt),
        "total_predictions": int(len(matches)),
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
        "precision": final_precision,
        "recall": final_recall,
        "ap": float(ap),
        "crown_area_metrics": _compute_crown_area_metrics(matches),
        "pr_curve": pr_curve,
        "matches": matches,
    }


def _compute_pr_at_conf(
    preds: list[dict[str, Any]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    iou_thr: float,
    conf_thr: float,
) -> dict[str, Any]:
    kept = [pred for pred in preds if float(pred["score"]) >= float(conf_thr)]
    matches, total_gt = _match_predictions(kept, gt_by_image, iou_thr=iou_thr)
    tp = sum(1 for row in matches if row["is_tp"])
    fp = sum(1 for row in matches if row["is_fp"])
    fn = max(total_gt - tp, 0)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(total_gt, 1))
    return {
        "total_gt": int(total_gt),
        "kept_predictions": int(len(kept)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "confidence_threshold": float(conf_thr),
    }


def _evaluate_split(
    split_name: str,
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    split_out_dir: Path,
    iou_thresholds: list[float],
    report_conf_threshold: float,
    score_mode: str,
    save_prob_tif: bool,
) -> dict[str, Any]:
    ensure_dir(split_out_dir)
    gt_by_image = {int(rec["image_id"]): rec["gt_instances"] for rec in records}
    all_preds: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for idx, rec in enumerate(records, start=1):
        image_name = rec["image_name"]
        print(f"[{split_name}] ({idx}/{len(records)}) {image_name}")
        image_out_dir = split_out_dir / f"{idx:04d}_{_safe_name(rec['image_stem'])}"
        ensure_dir(image_out_dir)

        infer_cfg = dict(cfg)
        infer_cfg["grouped_inference_enabled"] = False
        infer_cfg["_grouped_dispatch_active"] = False
        infer_cfg["input_image"] = rec["image_path"]
        infer_cfg["output_dir"] = str(image_out_dir)

        semantic_prior_pred = predict_semantic_prior_cached(infer_cfg)
        semantic_prior_outputs = _write_semantic_prior_outputs(semantic_prior_pred, image_out_dir, save_prob_tif=save_prob_tif)
        segmentation_info = run_segmentation_cached(infer_cfg, semantic_prior_outputs["m_sem_tif"])

        label_image = _read_instance_labels(segmentation_info["y_inst_tif"])
        pred_instances = _instances_from_label_image(
            label_image=label_image,
            image_id=int(rec["image_id"]),
            score_map=semantic_prior_pred["probability"],
            score_mode=score_mode,
        )
        all_preds.extend(pred_instances)

        pred_count = len(pred_instances)
        gt_count = len(rec["gt_instances"])
        image_rows.append(
            {
                "split": split_name,
                "image_id": int(rec["image_id"]),
                "image_name": image_name,
                "image_path": rec["image_path"],
                "width": int(rec["width"]),
                "height": int(rec["height"]),
                "num_gt_instances": int(gt_count),
                "num_pred_instances": int(pred_count),
                "output_dir": str(image_out_dir),
                "y_inst_tif": segmentation_info["y_inst_tif"],
                "y_inst_shp": segmentation_info["y_inst_shp"],
                "score_mode": score_mode,
            }
        )

        for pred in pred_instances:
            pred_rows.append(
                {
                    "split": split_name,
                    "image_id": int(rec["image_id"]),
                    "image_name": image_name,
                    "pred_id": int(pred["pred_id"]),
                    "score": float(pred["score"]),
                    "area": float(pred["area"]),
                    "bbox_x": float(pred["bbox"][0]),
                    "bbox_y": float(pred["bbox"][1]),
                    "bbox_w": float(pred["bbox"][2]),
                    "bbox_h": float(pred["bbox"][3]),
                }
            )

    split_metrics: dict[str, Any] = {
        "split": split_name,
        "num_images": int(len(records)),
        "num_gt_instances": int(sum(len(rec["gt_instances"]) for rec in records)),
        "num_pred_instances": int(len(all_preds)),
        "iou_metrics": {},
    }

    for iou_thr in iou_thresholds:
        pr_ap = _compute_pr_ap(all_preds, gt_by_image, iou_thr=iou_thr)
        pr_at_conf = _compute_pr_at_conf(all_preds, gt_by_image, iou_thr=iou_thr, conf_thr=report_conf_threshold)
        iou_key = f"{iou_thr:.2f}"
        split_metrics["iou_metrics"][iou_key] = {
            "iou_threshold": float(iou_thr),
            "ap": float(pr_ap["ap"]),
            "precision_at_all_predictions": float(pr_ap["precision"]),
            "recall_at_all_predictions": float(pr_ap["recall"]),
            "precision_at_confidence_threshold": float(pr_at_conf["precision"]),
            "recall_at_confidence_threshold": float(pr_at_conf["recall"]),
            "precision_percent_at_confidence_threshold": float(pr_at_conf["precision"] * 100.0),
            "recall_percent_at_confidence_threshold": float(pr_at_conf["recall"] * 100.0),
            "tp_at_confidence_threshold": int(pr_at_conf["tp"]),
            "fp_at_confidence_threshold": int(pr_at_conf["fp"]),
            "fn_at_confidence_threshold": int(pr_at_conf["fn"]),
            "confidence_threshold": float(report_conf_threshold),
            "crown_area": pr_ap["crown_area_metrics"],
        }

        pr_curve_path = split_out_dir / f"pr_curve_iou_{iou_key.replace('.', '_')}.csv"
        dump_csv(pd.DataFrame(pr_ap["pr_curve"]), pr_curve_path)

        matches_path = split_out_dir / f"prediction_matches_iou_{iou_key.replace('.', '_')}.csv"
        dump_csv(pd.DataFrame(pr_ap["matches"]), matches_path)

    image_df = pd.DataFrame(image_rows)
    pred_df = pd.DataFrame(pred_rows)
    dump_csv(image_df, split_out_dir / "per_image_manifest.csv")
    dump_csv(pred_df, split_out_dir / "prediction_instances.csv")

    if "0.50" in split_metrics["iou_metrics"]:
        split_metrics["ap50"] = float(split_metrics["iou_metrics"]["0.50"]["ap"])
        split_metrics["crown_area_iou_0_50"] = split_metrics["iou_metrics"]["0.50"]["crown_area"]
    if "0.75" in split_metrics["iou_metrics"]:
        split_metrics["ap75"] = float(split_metrics["iou_metrics"]["0.75"]["ap"])
        split_metrics["crown_area_iou_0_75"] = split_metrics["iou_metrics"]["0.75"]["crown_area"]

    dump_json(split_metrics, split_out_dir / "split_summary.json")
    return split_metrics


def _run_inference_only_split(
    split_name: str,
    image_paths: list[Path],
    cfg: dict[str, Any],
    split_out_dir: Path,
    save_prob_tif: bool,
) -> dict[str, Any]:
    ensure_dir(split_out_dir)
    rows: list[dict[str, Any]] = []
    total_pred_instances = 0

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{split_name}] ({idx}/{len(image_paths)}) {image_path.name}")
        image_out_dir = split_out_dir / f"{idx:04d}_{_safe_name(image_path.stem)}"
        ensure_dir(image_out_dir)

        infer_cfg = dict(cfg)
        infer_cfg["grouped_inference_enabled"] = False
        infer_cfg["_grouped_dispatch_active"] = False
        infer_cfg["input_image"] = str(image_path)
        infer_cfg["output_dir"] = str(image_out_dir)

        semantic_prior_pred = predict_semantic_prior_cached(infer_cfg)
        semantic_prior_outputs = _write_semantic_prior_outputs(semantic_prior_pred, image_out_dir, save_prob_tif=save_prob_tif)
        segmentation_info = run_segmentation_cached(infer_cfg, semantic_prior_outputs["m_sem_tif"])
        label_image = _read_instance_labels(segmentation_info["y_inst_tif"])
        pred_count = int((np.unique(label_image) > 0).sum())
        total_pred_instances += pred_count

        rows.append(
            {
                "split": split_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "output_dir": str(image_out_dir),
                "m_sem_tif": semantic_prior_outputs["m_sem_tif"],
                "y_inst_tif": segmentation_info["y_inst_tif"],
                "y_inst_shp": segmentation_info["y_inst_shp"],
                "num_pred_instances": pred_count,
            }
        )

    dump_csv(pd.DataFrame(rows), split_out_dir / "inference_manifest.csv")
    summary = {
        "split": split_name,
        "annotation_available": False,
        "num_images": int(len(image_paths)),
        "num_pred_instances": int(total_pred_instances),
        "message": "该 split 未提供 COCO 标注，因此仅执行推理，不计算 P/R/AP。",
    }
    dump_json(summary, split_out_dir / "split_summary.json")
    return summary


def _merge_base_config(base_cfg: dict[str, Any], overrides: Any) -> dict[str, Any]:
    merged = dict(base_cfg)
    if isinstance(overrides, dict):
        merged.update(overrides)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="benchmark yaml config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_cfg_path = cfg.get("base_config")
    if not base_cfg_path:
        raise ValueError("benchmark config 缺少 base_config")

    base_cfg = load_yaml(base_cfg_path)
    base_cfg = _merge_base_config(base_cfg, cfg.get("base_config_overrides"))
    dataset_root = Path(cfg["dataset_root"]).expanduser().resolve()
    output_root = ensure_dir(cfg.get("output_root", str(PROJECT_ROOT / "outputs" / "public_coco_benchmark")))

    split_mapping = _normalize_split_mapping(cfg.get("split_mapping"))
    requested_splits = _normalize_str_list(cfg.get("benchmark_splits"), ["validation"])
    image_dirname = str(cfg.get("image_dirname", "image"))
    annotation_dirname = str(cfg.get("annotation_dirname", "annotation"))
    iou_thresholds = _normalize_float_list(cfg.get("eval_iou_thresholds"), [0.5, 0.75])
    report_conf_threshold = float(cfg.get("report_confidence_threshold", 0.5))
    score_mode = str(cfg.get("prediction_score_mode", "semantic_prior_mean_prob")).strip() or "semantic_prior_mean_prob"
    save_prob_tif = bool(cfg.get("save_semantic_prior_probability_tif", False))
    max_images_per_split = _normalize_int(cfg.get("max_images_per_split"))

    benchmark_summary: dict[str, Any] = {
        "config": {
            "dataset_root": str(dataset_root),
            "base_config": str(base_cfg_path),
            "output_root": str(output_root),
            "benchmark_splits": requested_splits,
            "split_mapping": split_mapping,
            "image_dirname": image_dirname,
            "annotation_dirname": annotation_dirname,
            "eval_iou_thresholds": iou_thresholds,
            "report_confidence_threshold": report_conf_threshold,
            "prediction_score_mode": score_mode,
            "max_images_per_split": max_images_per_split,
        },
        "splits": {},
    }

    split_overview_rows: list[dict[str, Any]] = []
    for split_alias in requested_splits:
        split_dirname = split_mapping.get(split_alias, split_alias)
        split_dir = dataset_root / split_dirname
        image_dir = split_dir / image_dirname
        ann_path = split_dir / annotation_dirname

        if not image_dir.exists():
            raise FileNotFoundError(f"split 图像目录不存在: {image_dir}")

        print(f"[LOAD] split={split_alias} dir={split_dir}")
        has_annotations = ann_path.exists() and bool(_collect_coco_jsons(ann_path))
        split_out_dir = Path(output_root) / split_alias

        if has_annotations:
            coco = _load_merged_coco(ann_path)
            records = _prepare_gt_records(coco, image_dir=image_dir)
            if max_images_per_split is not None:
                records = records[:max_images_per_split]
            split_summary = _evaluate_split(
                split_name=split_alias,
                records=records,
                cfg=base_cfg,
                split_out_dir=split_out_dir,
                iou_thresholds=iou_thresholds,
                report_conf_threshold=report_conf_threshold,
                score_mode=score_mode,
                save_prob_tif=save_prob_tif,
            )
            split_summary["annotation_available"] = True
        else:
            image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES)
            if max_images_per_split is not None:
                image_paths = image_paths[:max_images_per_split]
            split_summary = _run_inference_only_split(
                split_name=split_alias,
                image_paths=image_paths,
                cfg=base_cfg,
                split_out_dir=split_out_dir,
                save_prob_tif=save_prob_tif,
            )

        benchmark_summary["splits"][split_alias] = split_summary
        overview_row = {
            "split": split_alias,
            "annotation_available": bool(split_summary.get("annotation_available", False)),
            "num_images": int(split_summary.get("num_images", 0)),
            "num_gt_instances": int(split_summary.get("num_gt_instances", 0)) if split_summary.get("annotation_available", False) else None,
            "num_pred_instances": int(split_summary.get("num_pred_instances", 0)) if split_summary.get("annotation_available", False) else None,
        }
        for iou_key, metric in split_summary.get("iou_metrics", {}).items():
            overview_row[f"ap_iou_{iou_key}"] = float(metric["ap"])
            overview_row[f"precision_iou_{iou_key}_conf"] = float(metric["precision_at_confidence_threshold"])
            overview_row[f"recall_iou_{iou_key}_conf"] = float(metric["recall_at_confidence_threshold"])
            crown_area = metric.get("crown_area", {})
            overview_row[f"mae_area_iou_{iou_key}"] = crown_area.get("mae")
            overview_row[f"rmse_area_iou_{iou_key}"] = crown_area.get("rmse")
            overview_row[f"rmse_percent_area_iou_{iou_key}"] = crown_area.get("rmse_percent")
            overview_row[f"r2_area_iou_{iou_key}"] = crown_area.get("r2")
            overview_row[f"matched_crowns_iou_{iou_key}"] = crown_area.get("num_matched_crowns")
        split_overview_rows.append(overview_row)

    dump_json(benchmark_summary, Path(output_root) / "benchmark_summary.json")
    dump_csv(pd.DataFrame(split_overview_rows), Path(output_root) / "benchmark_overview.csv")
    for split_alias, split_summary in benchmark_summary["splits"].items():
        if not split_summary.get("annotation_available", False):
            print(f"[RESULT] split={split_alias} annotation_available=false")
            continue
        ap50 = split_summary.get("ap50")
        ap75 = split_summary.get("ap75")
        line = f"[RESULT] split={split_alias} "
        line += f"AP50={ap50:.4f} " if ap50 is not None else "AP50=NA "
        line += f"AP75={ap75:.4f}" if ap75 is not None else "AP75=NA"
        print(line)
    print(f"[OK] benchmark finished: {output_root}")


if __name__ == "__main__":
    main()
