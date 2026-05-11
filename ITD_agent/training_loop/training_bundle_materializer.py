from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_csv, write_json


def materialize_training_dataset_bundle(
    *,
    accepted_samples: list[dict[str, Any]],
    replay_samples: list[dict[str, Any]],
    output_dir: str | Path,
    dataset_cfg: dict[str, Any],
    family_cfg: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir) / "dataset_bundle"
    for rel in ["train/images", "val/images", "test/images", "replay/images", "annotations"]:
        (root / rel).mkdir(parents=True, exist_ok=True)

    split_rows = _split_samples(
        accepted_samples,
        train_ratio=float(dataset_cfg.get("train_ratio", 0.70)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.15)),
    )
    payloads = {name: _build_coco_payload(rows, root / name / "images", root) for name, rows in split_rows.items()}
    replay_payload = _build_coco_payload(replay_samples, root / "replay" / "images", root)

    annotation_paths = {}
    for split_name, payload in payloads.items():
        annotation_paths[split_name] = write_json(root / "annotations" / f"instances_{split_name}.json", payload["coco"])
        write_csv(root / f"manifest_{split_name}.csv", payload["manifest"])
    write_json(root / "annotations" / "instances_replay.json", replay_payload["coco"])
    write_csv(root / "manifest_replay.csv", replay_payload["manifest"])

    dataset_card = {
        "dataset_bundle_dir": str(root),
        "split_counts": {
            split: {
                "samples": len(rows),
                "images": len(payloads[split]["coco"]["images"]),
                "annotations": len(payloads[split]["coco"]["annotations"]),
            }
            for split, rows in split_rows.items()
        },
        "replay_counts": {
            "samples": len(replay_samples),
            "images": len(replay_payload["coco"]["images"]),
            "annotations": len(replay_payload["coco"]["annotations"]),
        },
        "by_failure_category": dict(Counter(str(item.get("failure_category") or "unknown") for item in accepted_samples)),
        "target_expert_family": family_cfg.get("target_expert_family"),
        "algorithm_name": family_cfg.get("algorithm_name"),
        "replay_ratio": family_cfg.get("replay_ratio"),
        "hard_case_ratio": family_cfg.get("hard_case_ratio"),
        "leakage_policy": "source_trajectory_id/original_image_id groups are assigned to one split only",
    }
    dataset_card_path = write_json(root / "dataset_card.json", dataset_card)
    return {
        "dataset_bundle_dir": str(root),
        "dataset_card_path": dataset_card_path,
        "annotation_paths": annotation_paths,
        "replay_annotation_path": str(root / "annotations" / "instances_replay.json"),
        "dataset_card": dataset_card,
    }


def _split_samples(samples: list[dict[str, Any]], *, train_ratio: float, val_ratio: float) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[_group_key(sample)].append(sample)
    groups = sorted(grouped.items(), key=lambda item: item[0])
    total = len(groups)
    train_end = max(1, int(round(total * train_ratio))) if total else 0
    val_end = min(total, train_end + max(1, int(round(total * val_ratio)))) if total >= 3 else train_end
    result = {"train": [], "val": [], "test": []}
    for idx, (_, rows) in enumerate(groups):
        if idx < train_end:
            result["train"].extend(rows)
        elif idx < val_end:
            result["val"].extend(rows)
        else:
            result["test"].extend(rows)
    return result


def _group_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    return str(metadata.get("source_trajectory_id") or metadata.get("image_id") or sample.get("sample_id"))


def _build_coco_payload(samples: list[dict[str, Any]], image_dir: Path, dataset_root: Path) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        artifact_refs = sample.get("artifact_refs") or {}
        metadata = sample.get("metadata") or {}
        roi = metadata.get("roi") or {}
        src_image = Path(str(artifact_refs.get("image") or ""))
        suffix = src_image.suffix if src_image.suffix else ".png"
        image_name = f"{sample.get('sample_id') or idx}{suffix}"
        dst_image = image_dir / image_name
        if src_image.exists():
            shutil.copyfile(src_image, dst_image)
        else:
            dst_image.write_text("source image unavailable; V3 dataset manifest preserves source refs", encoding="utf-8")
        width, height = _image_size(roi)
        images.append({"id": idx, "file_name": str(dst_image.relative_to(dataset_root)), "width": width, "height": height})
        bbox_xywh = _bbox_xywh(roi.get("bbox_px") or [])
        annotations.append(
            {
                "id": idx,
                "image_id": idx,
                "category_id": 1,
                "bbox": bbox_xywh,
                "segmentation": [_bbox_polygon_xywh(bbox_xywh)],
                "area": max(1.0, bbox_xywh[2] * bbox_xywh[3]),
                "iscrowd": 0,
            }
        )
        manifest.append(
            {
                "sample_id": sample.get("sample_id"),
                "source_trajectory_id": metadata.get("source_trajectory_id"),
                "source_roi_id": metadata.get("source_roi_id"),
                "failure_category": sample.get("failure_category"),
                "label_status": sample.get("label_status"),
                "image_path": str(dst_image),
            }
        )
    return {
        "coco": {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "crown", "supercategory": "crown"}]},
        "manifest": manifest,
    }


def _image_size(roi: dict[str, Any]) -> tuple[int, int]:
    bbox = roi.get("bbox_px") or [0, 0, 1024, 1024]
    try:
        width = max(1, int(max(float(bbox[2]), 1024.0)))
        height = max(1, int(max(float(bbox[3]), 1024.0)))
    except (TypeError, ValueError, IndexError):
        return 1024, 1024
    return width, height


def _bbox_xywh(bbox: list[Any]) -> list[float]:
    if len(bbox) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def _bbox_polygon_xywh(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y, x + w, y + h, x, y + h]
