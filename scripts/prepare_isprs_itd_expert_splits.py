from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.segmentation.finetuning.io_utils import dump_json, ensure_dir


SPECIALIST_FAMILIES = (
    "dense_adhesion",
    "shadow_topography",
    "large_crown_over_split",
    "boundary_calibration",
)
ALL_FAMILIES = SPECIALIST_FAMILIES + ("cross_domain_generalist",)


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    source_role: str
    dataset_id: str
    metrics: dict[str, float]
    family_scores: dict[str, float]
    primary_family: str
    confidence_gap: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-holdout-frac", type=float, default=0.10)
    parser.add_argument("--val-holdout-frac", type=float, default=0.10)
    parser.add_argument("--min-specialist-train-images", type=int, default=400)
    parser.add_argument("--min-specialist-val-images", type=int, default=60)
    parser.add_argument("--max-pixel-samples-per-dataset", type=int, default=64)
    return parser.parse_args()


def _ann_irregularity(ann: dict[str, Any]) -> float:
    bbox = ann.get("bbox") or [0, 0, 0, 0]
    width = float(bbox[2]) if len(bbox) > 2 else 0.0
    height = float(bbox[3]) if len(bbox) > 3 else 0.0
    area = max(width * height, float(ann.get("area") or 0.0))
    if area <= 1e-6 or width <= 1e-6 or height <= 1e-6:
        return 1.0
    perimeter = max(0.0, 2.0 * (width + height))
    return float((perimeter * perimeter) / max(4.0 * math.pi * area, 1e-6))


def _image_stats(image_path: Path) -> tuple[float, float]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3:
        arr = np.repeat(arr[..., None], 3, axis=2)
    sampled = arr[::4, ::4, :]
    gray = 0.299 * sampled[:, :, 0] + 0.587 * sampled[:, :, 1] + 0.114 * sampled[:, :, 2]
    brightness = float(gray.mean() / 255.0)
    shadow_ratio = float((gray < 60.0).mean())
    return brightness, shadow_ratio


def _estimate_dataset_pixel_stats(
    *,
    dataset_root: Path,
    payloads: list[dict[str, Any]],
    max_samples_per_dataset: int,
) -> dict[str, tuple[float, float]]:
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for payload in payloads:
        for image in payload.get("images", []):
            dataset_id = str(image.get("_dataset_id") or "unknown")
            if len(grouped_paths[dataset_id]) >= max_samples_per_dataset:
                continue
            grouped_paths[dataset_id].append(dataset_root / str(image["file_name"]))

    stats: dict[str, tuple[float, float]] = {}
    for dataset_id, paths in grouped_paths.items():
        brightness_values: list[float] = []
        shadow_values: list[float] = []
        for path in paths:
            brightness, shadow_ratio = _image_stats(path)
            brightness_values.append(brightness)
            shadow_values.append(shadow_ratio)
        if brightness_values:
            stats[dataset_id] = (
                float(sum(brightness_values) / len(brightness_values)),
                float(sum(shadow_values) / len(shadow_values)),
            )
        else:
            stats[dataset_id] = (0.5, 0.1)
    return stats


def _load_role_payloads(dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_payloads: list[dict[str, Any]] = []
    for ann_path in sorted(dataset_root.glob("Training_set/Dataset_*_train/annotation/*.json")):
        dataset_name = ann_path.stem
        dataset_rel_root = ann_path.parent.parent
        with open(ann_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for image in payload.get("images", []):
            image["file_name"] = str((dataset_rel_root / "images" / Path(str(image["file_name"])).name).as_posix())
            image["_dataset_id"] = dataset_name
            image["_source_role"] = "train"
        train_payloads.append(payload)

    val_path = dataset_root / "Validation_set" / "annotation" / "validation_gt.json"
    with open(val_path, "r", encoding="utf-8") as f:
        val_payload = json.load(f)
    for image in val_payload.get("images", []):
        image["file_name"] = str((Path("Validation_set") / "images" / Path(str(image["file_name"])).name).as_posix())
        image["_dataset_id"] = "Validation_set"
        image["_source_role"] = "val"
    return train_payloads, [val_payload]


def _merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1
    for payload in payloads:
        local_to_global: dict[int, int] = {}
        for image in payload.get("images", []):
            copied = dict(image)
            old_id = int(copied["id"])
            copied["id"] = next_image_id
            local_to_global[old_id] = next_image_id
            merged_images.append(copied)
            next_image_id += 1
        for ann in payload.get("annotations", []):
            copied = dict(ann)
            copied["id"] = next_ann_id
            copied["image_id"] = local_to_global[int(copied["image_id"])]
            merged_annotations.append(copied)
            next_ann_id += 1
        if not categories and payload.get("categories"):
            categories = payload["categories"]
    if not categories:
        categories = [{"id": 1, "name": "crown", "supercategory": "crown"}]
    return {"images": merged_images, "annotations": merged_annotations, "categories": categories}


def _zscore_map(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    series = list(values.values())
    mean = statistics.fmean(series)
    std = statistics.pstdev(series)
    if std <= 1e-8:
        return {key: 0.0 for key in values}
    return {key: float((value - mean) / std) for key, value in values.items()}


def _score_families(z: dict[str, float]) -> tuple[dict[str, float], str, float]:
    dense = 1.4 * z["instance_count"] + 1.1 * z["coverage_ratio"] - 1.0 * z["mean_area"] + 0.2 * z["shadow_ratio"]
    shadow = 1.5 * z["shadow_ratio"] - 1.0 * z["brightness"] + 0.35 * z["coverage_ratio"] + 0.25 * z["irregularity"]
    large = 1.4 * z["mean_area"] - 1.0 * z["instance_count"] + 0.35 * z["brightness"] + 0.3 * z["irregularity"]
    boundary = 1.55 * z["irregularity"] + 0.35 * z["shadow_ratio"] + 0.2 * z["instance_count"]
    scores = {
        "dense_adhesion": dense,
        "shadow_topography": shadow,
        "large_crown_over_split": large,
        "boundary_calibration": boundary,
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_family, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else top_score
    gap = float(top_score - second_score)
    if gap < 0.35:
        return {**scores, "cross_domain_generalist": 0.5 - gap}, "cross_domain_generalist", gap
    return {**scores, "cross_domain_generalist": 0.0}, top_family, gap


def _subset_payload(payload: dict[str, Any], selected_ids: set[int]) -> dict[str, Any]:
    images = [dict(img) for img in payload.get("images", []) if int(img["id"]) in selected_ids]
    anns = [dict(ann) for ann in payload.get("annotations", []) if int(ann["image_id"]) in selected_ids]
    image_id_map: dict[int, int] = {}
    next_image_id = 1
    next_ann_id = 1
    remapped_images: list[dict[str, Any]] = []
    remapped_anns: list[dict[str, Any]] = []
    for image in images:
        old_id = int(image["id"])
        image["id"] = next_image_id
        image_id_map[old_id] = next_image_id
        remapped_images.append(image)
        next_image_id += 1
    for ann in anns:
        ann["id"] = next_ann_id
        ann["image_id"] = image_id_map[int(ann["image_id"])]
        remapped_anns.append(ann)
        next_ann_id += 1
    return {
        "images": remapped_images,
        "annotations": remapped_anns,
        "categories": payload.get("categories") or [{"id": 1, "name": "crown", "supercategory": "crown"}],
    }


def _deterministic_holdout(image_ids: list[int], fraction: float) -> tuple[set[int], set[int]]:
    ordered = sorted(image_ids)
    if not ordered:
        return set(), set()
    holdout_count = min(len(ordered), max(1, int(math.ceil(len(ordered) * fraction))))
    holdout_ids = set(ordered[:holdout_count])
    remain_ids = set(ordered[holdout_count:])
    return remain_ids, holdout_ids


def _write_family_payloads(
    *,
    output_dir: Path,
    family: str,
    merged_train: dict[str, Any],
    merged_val: dict[str, Any],
    train_ids: set[int],
    val_ids: set[int],
    train_holdout_frac: float,
    val_holdout_frac: float,
) -> dict[str, Any]:
    remain_train_ids, holdout_train_ids = _deterministic_holdout(list(train_ids), train_holdout_frac)
    remain_val_ids, holdout_val_ids = _deterministic_holdout(list(val_ids), val_holdout_frac)

    train_payload = _subset_payload(merged_train, remain_train_ids)
    val_payload = _subset_payload(merged_val, remain_val_ids)
    holdout_from_train = _subset_payload(merged_train, holdout_train_ids)
    holdout_from_val = _subset_payload(merged_val, holdout_val_ids)
    test_payload = _merge_payloads([holdout_from_train, holdout_from_val])

    family_dir = ensure_dir(output_dir / family / "annotations")
    dump_json(train_payload, family_dir / "instances_train.json")
    dump_json(val_payload, family_dir / "instances_val.json")
    dump_json(test_payload, family_dir / "instances_test.json")
    return {
        "family": family,
        "train_images": len(train_payload["images"]),
        "train_annotations": len(train_payload["annotations"]),
        "val_images": len(val_payload["images"]),
        "val_annotations": len(val_payload["annotations"]),
        "test_images": len(test_payload["images"]),
        "test_annotations": len(test_payload["annotations"]),
        "holdout_from_train": len(holdout_train_ids),
        "holdout_from_val": len(holdout_val_ids),
        "annotation_files": {
            "train": str((family_dir / "instances_train.json").resolve()),
            "val": str((family_dir / "instances_val.json").resolve()),
            "test": str((family_dir / "instances_test.json").resolve()),
        },
    }


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    train_payloads, val_payloads = _load_role_payloads(dataset_root)
    dataset_pixel_stats = _estimate_dataset_pixel_stats(
        dataset_root=dataset_root,
        payloads=[*train_payloads, *val_payloads],
        max_samples_per_dataset=args.max_pixel_samples_per_dataset,
    )
    merged_train = _merge_payloads(train_payloads)
    merged_val = _merge_payloads(val_payloads)

    train_ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in merged_train.get("annotations", []):
        train_ann_by_image[int(ann["image_id"])].append(ann)
    val_ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in merged_val.get("annotations", []):
        val_ann_by_image[int(ann["image_id"])].append(ann)

    raw_metrics: dict[int, dict[str, float]] = {}
    image_meta: dict[int, dict[str, Any]] = {}
    for role, payload, ann_index in (
        ("train", merged_train, train_ann_by_image),
        ("val", merged_val, val_ann_by_image),
    ):
        for image in payload.get("images", []):
            image_id = int(image["id"])
            anns = ann_index.get(image_id, [])
            counts = len(anns)
            areas = [float(ann.get("area") or 0.0) for ann in anns if float(ann.get("area") or 0.0) > 0.0]
            coverage = float(sum(areas) / max(1, int(image["width"]) * int(image["height"])))
            irregularities = [_ann_irregularity(ann) for ann in anns] or [1.0]
            dataset_id = str(image.get("_dataset_id") or role)
            brightness, shadow_ratio = dataset_pixel_stats.get(dataset_id, (0.5, 0.1))
            raw_metrics[image_id] = {
                "instance_count": float(counts),
                "mean_area": float(sum(areas) / len(areas)) if areas else 0.0,
                "coverage_ratio": coverage,
                "brightness": brightness,
                "shadow_ratio": shadow_ratio,
                "irregularity": float(sum(irregularities) / len(irregularities)),
            }
            image_meta[image_id] = {
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "source_role": role,
                "dataset_id": dataset_id,
            }

    z_metrics = {
        key: _zscore_map({image_id: values[key] for image_id, values in raw_metrics.items()})
        for key in next(iter(raw_metrics.values())).keys()
    }

    records: dict[int, ImageRecord] = {}
    train_selected: dict[str, set[int]] = {family: set() for family in SPECIALIST_FAMILIES}
    val_selected: dict[str, set[int]] = {family: set() for family in SPECIALIST_FAMILIES}
    specialist_rankings: dict[str, list[tuple[int, float, str]]] = {family: [] for family in SPECIALIST_FAMILIES}
    all_train_ids = {int(img["id"]) for img in merged_train["images"]}
    all_val_ids = {int(img["id"]) for img in merged_val["images"]}

    for image_id, metrics in raw_metrics.items():
        z = {metric_name: z_metrics[metric_name][image_id] for metric_name in z_metrics}
        family_scores, primary_family, gap = _score_families(z)
        meta = image_meta[image_id]
        record = ImageRecord(
            image_id=image_id,
            file_name=meta["file_name"],
            width=meta["width"],
            height=meta["height"],
            source_role=meta["source_role"],
            dataset_id=meta["dataset_id"],
            metrics=metrics,
            family_scores=family_scores,
            primary_family=primary_family,
            confidence_gap=gap,
        )
        records[image_id] = record
        if primary_family in SPECIALIST_FAMILIES:
            target = train_selected if meta["source_role"] == "train" else val_selected
            target[primary_family].add(image_id)
        for family in SPECIALIST_FAMILIES:
            specialist_rankings[family].append((image_id, family_scores[family], meta["source_role"]))

    for family in SPECIALIST_FAMILIES:
        specialist_rankings[family].sort(key=lambda item: item[1], reverse=True)
        target_train = train_selected[family]
        target_val = val_selected[family]
        if len(target_train) < args.min_specialist_train_images:
            for image_id, _, role in specialist_rankings[family]:
                if role != "train":
                    continue
                target_train.add(image_id)
                if len(target_train) >= args.min_specialist_train_images:
                    break
        if len(target_val) < args.min_specialist_val_images:
            for image_id, _, role in specialist_rankings[family]:
                if role != "val":
                    continue
                target_val.add(image_id)
                if len(target_val) >= args.min_specialist_val_images:
                    break

    family_summaries: list[dict[str, Any]] = []
    for family in SPECIALIST_FAMILIES:
        family_summaries.append(
            _write_family_payloads(
                output_dir=output_dir,
                family=family,
                merged_train=merged_train,
                merged_val=merged_val,
                train_ids=train_selected[family],
                val_ids=val_selected[family],
                train_holdout_frac=args.train_holdout_frac,
                val_holdout_frac=args.val_holdout_frac,
            )
        )

    family_summaries.append(
        _write_family_payloads(
            output_dir=output_dir,
            family="cross_domain_generalist",
            merged_train=merged_train,
            merged_val=merged_val,
            train_ids=all_train_ids,
            val_ids=all_val_ids,
            train_holdout_frac=args.train_holdout_frac,
            val_holdout_frac=args.val_holdout_frac,
        )
    )

    dump_json(
        {
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "families": family_summaries,
            "dataset_pixel_stats": {
                dataset_id: {
                    "brightness_mean": brightness,
                    "shadow_ratio_mean": shadow_ratio,
                }
                for dataset_id, (brightness, shadow_ratio) in dataset_pixel_stats.items()
            },
            "image_records": {
                str(image_id): {
                    "file_name": record.file_name,
                    "source_role": record.source_role,
                    "dataset_id": record.dataset_id,
                    "primary_family": record.primary_family,
                    "confidence_gap": record.confidence_gap,
                    "metrics": record.metrics,
                    "family_scores": record.family_scores,
                }
                for image_id, record in records.items()
            },
        },
        output_dir / "expert_split_summary.json",
    )
    print(f"[OK] expert splits prepared: {output_dir}")


if __name__ == "__main__":
    main()
