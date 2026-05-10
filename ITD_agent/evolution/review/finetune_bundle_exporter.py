from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_csv, write_json


def export_finetune_bundle(*, review_output_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    src = Path(review_output_dir) / "finetune_pool"
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    samples = _load_jsonl(src / "samples.jsonl")
    rows: list[dict[str, Any]] = []
    images_dir = dst / "coco_export" / "images"
    anns_dir = dst / "coco_export" / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    anns_dir.mkdir(parents=True, exist_ok=True)
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id"))
        image_path = Path(str(sample.get("image_crop_path") or ""))
        copied_image = images_dir / f"{sample_id}.png"
        if image_path.exists():
            shutil.copyfile(image_path, copied_image)
        rows.append({k: sample.get(k) for k in ["sample_id", "source_trajectory_id", "source_roi_id", "sample_type", "target_error_type", "quality_score", "metadata_path"]})
        coco_images.append({"id": idx, "file_name": copied_image.name, "width": 1024, "height": 1024})
        roi = sample.get("roi") or {}
        bbox = roi.get("bbox_px") or [0, 0, 1, 1]
        coco_annotations.append({"id": idx, "image_id": idx, "category_id": 1, "bbox": _xyxy_to_xywh(bbox), "area": max(1.0, float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1])), "iscrowd": 0})
    write_csv(dst / "manifest.csv", rows)
    write_json(dst / "manifest.json", rows)
    coco_payload = {"images": coco_images, "annotations": coco_annotations, "categories": [{"id": 1, "name": "tree"}]}
    write_json(anns_dir / "instances_itd_v2_candidates.json", coco_payload)
    return {"out_dir": str(dst), "sample_count": len(samples), "manifest_csv": str(dst / "manifest.csv"), "coco_annotations": str(anns_dir / "instances_itd_v2_candidates.json")}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _xyxy_to_xywh(bbox: list[Any]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
