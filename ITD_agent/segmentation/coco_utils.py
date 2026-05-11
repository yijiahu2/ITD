from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_utils

from ITD_agent.common.values import normalize_str_list

VALID_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_SPLIT_MAPPING = {
    "train": "Training_set",
    "val": "Validation_set",
    "validation": "Validation_set",
    "test": "Testing_set",
}


def normalize_split_mapping(raw: Any) -> dict[str, str]:
    mapping = dict(DEFAULT_SPLIT_MAPPING)
    if isinstance(raw, dict):
        for key, value in raw.items():
            mapping[str(key)] = str(value)
    return mapping


def collect_coco_jsons(annotation_path: str | Path) -> list[Path]:
    path = Path(annotation_path)
    if path.is_file() and path.suffix.lower() == ".json":
        return [path]
    if not path.exists():
        return []
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".json")
    return []


def load_merged_coco(annotation_path: str | Path) -> dict[str, Any]:
    json_paths = collect_coco_jsons(annotation_path)
    if not json_paths:
        raise FileNotFoundError(f"未找到 COCO 标注 json: {annotation_path}")

    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        local_to_global: dict[int, int] = {}
        for image in payload.get("images", []):
            image_copy = dict(image)
            old_id = int(image_copy["id"])
            image_copy["id"] = next_image_id
            local_to_global[old_id] = next_image_id
            merged_images.append(image_copy)
            next_image_id += 1

        for ann in payload.get("annotations", []):
            ann_copy = dict(ann)
            ann_copy["id"] = next_ann_id
            ann_copy["image_id"] = local_to_global[int(ann_copy["image_id"])]
            merged_annotations.append(ann_copy)
            next_ann_id += 1

        if not categories and payload.get("categories"):
            categories = payload["categories"]

    return {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }


def build_image_index(image_dir: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in sorted(Path(image_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        by_name[path.name] = path
        by_stem.setdefault(path.stem, path)
    return by_name, by_stem


def resolve_image_path(file_name: str, by_name: dict[str, Path], by_stem: dict[str, Path], image_dir: str | Path) -> Path:
    candidate = Path(image_dir) / file_name
    if candidate.exists():
        return candidate

    basename = Path(file_name).name
    if basename in by_name:
        return by_name[basename]

    stem = Path(file_name).stem
    if stem in by_stem:
        return by_stem[stem]

    raise FileNotFoundError(f"找不到与 COCO file_name 对应的图像: {file_name}")


def segmentation_to_rle(segmentation: Any, height: int, width: int) -> dict[str, Any]:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.merge(rles)

    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            return mask_utils.frPyObjects(segmentation, height, width)
        return segmentation

    raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")


__all__ = [
    "VALID_IMAGE_SUFFIXES",
    "build_image_index",
    "collect_coco_jsons",
    "load_merged_coco",
    "normalize_split_mapping",
    "normalize_str_list",
    "resolve_image_path",
    "segmentation_to_rle",
]
