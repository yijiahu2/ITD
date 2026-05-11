from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from ITD_agent.segmentation.finetuning.io_utils import dump_json, load_yaml
from ITD_agent.segmentation.coco_utils import (
    build_image_index as _build_image_index,
    load_merged_coco as _load_merged_coco,
    normalize_split_mapping as _normalize_split_mapping,
    normalize_str_list as _normalize_str_list,
    resolve_image_path as _resolve_image_path,
    segmentation_to_rle as _segmentation_to_rle,
)
from ITD_agent.segmentation.finetuning.prepare_data_processing_external_dataset import (
    _binary_mask_to_coco_annotation,
    _build_coco_split,
    _iter_tile_samples,
    _read_image_rgb_uint8,
    _save_mask_png,
)


def _decode_union_mask(annotations: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    union_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        rle = _segmentation_to_rle(ann["segmentation"], height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded > 0, axis=2)
        union_mask |= (decoded > 0).astype(np.uint8)
    return union_mask


def _normalize_split_name(split_name: str) -> str:
    name = str(split_name).strip().lower()
    if name == "validation":
        return "val"
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["public_dataset_root"]).expanduser().resolve()
    split_mapping = _normalize_split_mapping(cfg.get("public_dataset_split_mapping"))
    train_splits = [_normalize_split_name(x) for x in _normalize_str_list(cfg.get("public_dataset_train_splits"), ["train"])]
    val_splits = [_normalize_split_name(x) for x in _normalize_str_list(cfg.get("public_dataset_val_splits"), ["val", "validation"])]
    test_splits = [_normalize_split_name(x) for x in _normalize_str_list(cfg.get("public_dataset_test_splits"), ["test"])]
    split_to_role = {name: "train" for name in train_splits}
    split_to_role.update({name: "val" for name in val_splits})
    split_to_role.update({name: "test" for name in test_splits})

    tile_size = int(cfg.get("external_dataset_tile_size", 1024))
    tile_overlap = int(cfg.get("external_dataset_tile_overlap", 256))
    image_dirname = str(cfg.get("public_dataset_image_dirname", "images"))
    annotation_dirname = str(cfg.get("public_dataset_annotation_dirname", "annotation"))
    max_images_per_split = cfg.get("public_dataset_max_images_per_split")
    max_images_per_split = int(max_images_per_split) if max_images_per_split not in {None, ""} else None

    out_dir = Path(cfg["output_dir"]) / cfg.get("external_dataset_dirname", "external_data_processing_dataset")
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    summary_splits: list[dict[str, Any]] = []
    image_id = 1
    visited_keys: set[tuple[str, str]] = set()

    for split_alias, split_rel_path in split_mapping.items():
        normalized_split = _normalize_split_name(split_alias)
        split_role = split_to_role.get(normalized_split)
        if split_role is None:
            continue
        visit_key = (split_role, str(split_rel_path))
        if visit_key in visited_keys:
            continue
        visited_keys.add(visit_key)

        split_dir = dataset_root / split_rel_path
        image_dir = split_dir / image_dirname
        annotation_path = split_dir / annotation_dirname
        if not image_dir.exists():
            raise FileNotFoundError(f"公开数据集 split 图像目录不存在: {image_dir}")
        if not annotation_path.exists():
            if split_role == "test":
                summary_splits.append(
                    {
                        "split_alias": split_alias,
                        "split_rel_path": split_rel_path,
                        "split_role": split_role,
                        "num_images": 0,
                        "num_tiles": 0,
                        "status": "skipped_missing_annotation",
                    }
                )
                continue
            raise FileNotFoundError(f"公开数据集 split 标注目录不存在: {annotation_path}")

        coco = _load_merged_coco(annotation_path)
        ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            ann_by_image[int(ann["image_id"])].append(ann)

        by_name, by_stem = _build_image_index(image_dir)
        images = coco.get("images", [])
        if max_images_per_split is not None:
            images = images[:max_images_per_split]

        split_tile_count = 0
        split_image_count = 0
        for image in images:
            image_path = _resolve_image_path(str(image["file_name"]), by_name, by_stem, image_dir)
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            image_rgb = _read_image_rgb_uint8(image_path)
            if width <= 0 or height <= 0:
                height, width = image_rgb.shape[:2]

            mask_bin = _decode_union_mask(ann_by_image.get(int(image["id"]), []), height=height, width=width)
            tile_samples = _iter_tile_samples(
                image_rgb=image_rgb,
                mask_bin=mask_bin,
                tile_size=tile_size,
                overlap=tile_overlap,
            )

            for tile_suffix, image_tile, mask_tile in tile_samples:
                tile_roi_id = f"{image_path.stem}_{tile_suffix}"
                image_dst = images_dir / f"{tile_roi_id}.png"
                mask_dst = masks_dir / f"{tile_roi_id}.png"

                Image.fromarray(image_tile, mode="RGB").save(image_dst)
                _save_mask_png(mask_tile, mask_dst)

                tile_h, tile_w = mask_tile.shape
                annotation = _binary_mask_to_coco_annotation(
                    mask_bin=mask_tile,
                    image_id=image_id,
                    ann_id=image_id,
                    category_id=1,
                )

                rec = {
                    "id": image_id,
                    "roi_id": tile_roi_id,
                    "source_image_id": int(image["id"]),
                    "source_image_name": image_path.name,
                    "split": split_role,
                    "image_src": str(image_path),
                    "image_dst": str(image_dst),
                    "mask_dst": str(mask_dst),
                    "image_relpath": image_dst.name,
                    "mask_relpath": mask_dst.name,
                    "width": tile_w,
                    "height": tile_h,
                    "annotation": annotation,
                    "foreground_pixels": int(mask_tile.sum()),
                }
                split_records[split_role].append(rec)
                image_id += 1
                split_tile_count += 1

            split_image_count += 1

        summary_splits.append(
            {
                "split_alias": split_alias,
                "split_rel_path": split_rel_path,
                "split_role": split_role,
                "num_images": int(split_image_count),
                "num_tiles": int(split_tile_count),
            }
        )

    train_json = _build_coco_split(split_records["train"], "train")
    val_json = _build_coco_split(split_records["val"], "val")
    test_json = _build_coco_split(split_records["test"], "test")

    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)
    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_json, f, ensure_ascii=False, indent=2)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)

    summary = {
        "status": "prepared",
        "dataset_root": str(out_dir),
        "source_dataset_root": str(dataset_root),
        "source_type": "public_coco_instance_to_semantic",
        "num_train": len(split_records["train"]),
        "num_val": len(split_records["val"]),
        "num_test": len(split_records["test"]),
        "outputs": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "train_json": str(out_dir / "train.json"),
            "val_json": str(out_dir / "val.json"),
            "test_json": str(out_dir / "test.json"),
        },
        "tiling": {
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
        },
        "splits": summary_splits,
    }
    dump_json(summary, out_dir / "external_dataset_summary.json")
    print(
        f"[OK] public COCO data processing dataset prepared: {out_dir}, "
        f"train={len(split_records['train'])}, val={len(split_records['val'])}, test={len(split_records['test'])}"
    )


if __name__ == "__main__":
    main()
