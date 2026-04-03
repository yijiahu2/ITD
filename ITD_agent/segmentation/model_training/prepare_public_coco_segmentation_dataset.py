from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, ensure_dir, load_yaml


VALID_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DATASET_ID_PATTERN = re.compile(r"Dataset_(\d+)_")


def _normalize_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return default


def _normalize_split_mapping(raw: Any) -> dict[str, str]:
    mapping = {
        "train": "Training_set",
        "val": "Validation_set",
        "validation": "Validation_set",
        "test": "Testing_set",
    }
    if isinstance(raw, dict):
        for key, value in raw.items():
            mapping[str(key)] = str(value)
    return mapping


def _normalize_split_name(split_name: str) -> str:
    return str(split_name).strip().lower()


def _normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return [int(x) for x in items]
    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value]
    return [int(value)]


def _parse_dataset_id(file_name: str) -> int | None:
    match = DATASET_ID_PATTERN.search(str(file_name))
    if not match:
        return None
    return int(match.group(1))


def _collect_coco_jsons(annotation_path: Path) -> list[Path]:
    if annotation_path.is_file() and annotation_path.suffix.lower() == ".json":
        return [annotation_path]
    if not annotation_path.exists():
        return []
    if annotation_path.is_dir():
        return sorted(p for p in annotation_path.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    return []


def _load_merged_coco(annotation_path: Path) -> dict[str, Any]:
    json_paths = _collect_coco_jsons(annotation_path)
    if not json_paths:
        raise FileNotFoundError(f"未找到 COCO 标注 json: {annotation_path}")

    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

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


def _build_image_index(image_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        by_name[path.name] = path
        by_stem.setdefault(path.stem, path)
    return by_name, by_stem


def _resolve_image_path(
    file_name: str,
    by_name: dict[str, Path],
    by_stem: dict[str, Path],
    image_dir: Path,
) -> Path:
    candidate = image_dir / file_name
    if candidate.exists():
        return candidate

    basename = Path(file_name).name
    if basename in by_name:
        return by_name[basename]

    stem = Path(file_name).stem
    if stem in by_stem:
        return by_stem[stem]

    raise FileNotFoundError(f"找不到与 COCO file_name 对应的图像: {file_name}")


def _remap_split_coco(
    *,
    dataset_root: Path,
    split_dir: Path,
    image_dirname: str,
    annotation_dirname: str,
    max_images: int | None,
    include_dataset_ids: set[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_dir = split_dir / image_dirname
    annotation_path = split_dir / annotation_dirname
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"标注目录不存在: {annotation_path}")

    coco = _load_merged_coco(annotation_path)
    images = coco.get("images", [])
    if include_dataset_ids:
        images = [
            image
            for image in images
            if _parse_dataset_id(str(image.get("file_name", ""))) in include_dataset_ids
        ]
    if max_images is not None:
        images = images[:max_images]

    allowed_image_ids = {int(img["id"]) for img in images}
    annotations = [
        dict(ann)
        for ann in coco.get("annotations", [])
        if int(ann["image_id"]) in allowed_image_ids
    ]

    by_name, by_stem = _build_image_index(image_dir)
    next_image_id = 1
    next_ann_id = 1
    image_id_mapping: dict[int, int] = {}
    remapped_images: list[dict[str, Any]] = []
    remapped_annotations: list[dict[str, Any]] = []

    for image in images:
        image_copy = dict(image)
        src_image_path = _resolve_image_path(str(image_copy["file_name"]), by_name, by_stem, image_dir)
        image_copy["file_name"] = str(src_image_path.resolve().relative_to(dataset_root))
        old_id = int(image_copy["id"])
        image_copy["id"] = next_image_id
        image_id_mapping[old_id] = next_image_id
        remapped_images.append(image_copy)
        next_image_id += 1

    for ann in annotations:
        ann_copy = dict(ann)
        ann_copy["id"] = next_ann_id
        ann_copy["image_id"] = image_id_mapping[int(ann_copy["image_id"])]
        remapped_annotations.append(ann_copy)
        next_ann_id += 1

    categories = coco.get("categories") or [{"id": 1, "name": "crown", "supercategory": "crown"}]
    split_summary = {
        "split_dir": str(split_dir),
        "image_dir": str(image_dir),
        "annotation_path": str(annotation_path),
        "include_dataset_ids": sorted(include_dataset_ids) if include_dataset_ids else [],
        "num_images": len(remapped_images),
        "num_annotations": len(remapped_annotations),
    }
    return {
        "images": remapped_images,
        "annotations": remapped_annotations,
        "categories": categories,
    }, split_summary


def _merge_role_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1

    for payload in payloads:
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

    if not categories:
        categories = [{"id": 1, "name": "crown", "supercategory": "crown"}]

    return {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }


def _reindex_payload(payload: dict[str, Any]) -> dict[str, Any]:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories") or [{"id": 1, "name": "crown", "supercategory": "crown"}]

    next_image_id = 1
    next_ann_id = 1
    image_id_mapping: dict[int, int] = {}
    remapped_images: list[dict[str, Any]] = []
    remapped_annotations: list[dict[str, Any]] = []

    for image in images:
        image_copy = dict(image)
        old_id = int(image_copy["id"])
        image_copy["id"] = next_image_id
        image_id_mapping[old_id] = next_image_id
        remapped_images.append(image_copy)
        next_image_id += 1

    for ann in annotations:
        ann_copy = dict(ann)
        ann_copy["id"] = next_ann_id
        ann_copy["image_id"] = image_id_mapping[int(ann_copy["image_id"])]
        remapped_annotations.append(ann_copy)
        next_ann_id += 1

    return {
        "images": remapped_images,
        "annotations": remapped_annotations,
        "categories": categories,
    }


def _payload_subset_by_image_ids(payload: dict[str, Any], selected_image_ids: set[int]) -> dict[str, Any]:
    images = [dict(image) for image in payload.get("images", []) if int(image["id"]) in selected_image_ids]
    annotations = [
        dict(ann)
        for ann in payload.get("annotations", [])
        if int(ann["image_id"]) in selected_image_ids
    ]
    return _reindex_payload(
        {
            "images": images,
            "annotations": annotations,
            "categories": payload.get("categories"),
        }
    )


def _split_payload_holdout(
    payload: dict[str, Any],
    *,
    fraction: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    images = list(payload.get("images", []))
    if not images or fraction <= 0:
        empty = {
            "images": [],
            "annotations": [],
            "categories": payload.get("categories") or [{"id": 1, "name": "crown", "supercategory": "crown"}],
        }
        return _reindex_payload(payload), empty, {"selected_images": 0, "remaining_images": len(images)}

    indexed = sorted(images, key=lambda x: str(x.get("file_name", "")))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    holdout_count = min(len(indexed), max(1, int(math.ceil(len(indexed) * fraction))))
    holdout_ids = {int(image["id"]) for image in indexed[:holdout_count]}
    remain_ids = {int(image["id"]) for image in indexed[holdout_count:]}

    remain_payload = _payload_subset_by_image_ids(payload, remain_ids)
    holdout_payload = _payload_subset_by_image_ids(payload, holdout_ids)
    split_summary = {
        "fraction": fraction,
        "seed": seed,
        "selected_images": len(holdout_payload["images"]),
        "selected_annotations": len(holdout_payload["annotations"]),
        "remaining_images": len(remain_payload["images"]),
        "remaining_annotations": len(remain_payload["annotations"]),
    }
    return remain_payload, holdout_payload, split_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_root = Path(cfg["public_dataset_root"]).expanduser().resolve()
    split_mapping = _normalize_split_mapping(cfg.get("public_dataset_split_mapping"))
    train_splits = _normalize_str_list(cfg.get("public_dataset_train_splits"), ["train"])
    val_splits = _normalize_str_list(cfg.get("public_dataset_val_splits"), ["val", "validation"])
    test_splits = _normalize_str_list(cfg.get("public_dataset_test_splits"), ["test"])
    image_dirname = str(cfg.get("public_dataset_image_dirname", "images"))
    annotation_dirname = str(cfg.get("public_dataset_annotation_dirname", "annotation"))
    max_images_per_split = cfg.get("public_dataset_max_images_per_split")
    max_images_per_split = int(max_images_per_split) if max_images_per_split not in {None, ""} else None
    include_dataset_ids_cfg = cfg.get("public_dataset_include_dataset_ids_by_role") or {}
    holdout_test_fraction = float(cfg.get("public_dataset_holdout_test_fraction") or 0.0)
    holdout_test_seed = int(cfg.get("public_dataset_holdout_test_seed") or 42)
    holdout_test_source_roles = _normalize_str_list(
        cfg.get("public_dataset_holdout_test_source_roles"),
        [],
    )

    out_dir = Path(cfg["output_dir"]) / cfg.get("segmentation_dataset_dirname", "external_segmentation_dataset")
    ann_dir = ensure_dir(out_dir / "annotations")

    role_to_aliases = {
        "train": train_splits,
        "val": val_splits,
        "test": test_splits,
    }
    merged_by_role: dict[str, dict[str, Any]] = {}
    summary_roles: dict[str, Any] = {}

    for role, aliases in role_to_aliases.items():
        role_payloads: list[dict[str, Any]] = []
        role_summaries: list[dict[str, Any]] = []
        include_dataset_ids = set(_normalize_int_list(include_dataset_ids_cfg.get(role))) if isinstance(include_dataset_ids_cfg, dict) else set()

        for alias in aliases:
            split_rel = split_mapping.get(alias)
            if not split_rel:
                raise KeyError(f"public_dataset_split_mapping 缺少 split={alias}")
            split_dir = dataset_root / split_rel
            if not split_dir.exists():
                raise FileNotFoundError(f"公开数据集 split 目录不存在: {split_dir}")

            annotation_path = split_dir / annotation_dirname
            if not annotation_path.exists():
                if role == "test":
                    role_summaries.append(
                        {
                            "split_alias": alias,
                            "split_rel_path": split_rel,
                            "status": "skipped_missing_annotation",
                        }
                    )
                    continue
                raise FileNotFoundError(f"公开数据集 split 标注目录不存在: {annotation_path}")

            payload, split_summary = _remap_split_coco(
                dataset_root=dataset_root,
                split_dir=split_dir,
                image_dirname=image_dirname,
                annotation_dirname=annotation_dirname,
                max_images=max_images_per_split,
                include_dataset_ids=include_dataset_ids or None,
            )
            split_summary["split_alias"] = alias
            split_summary["split_rel_path"] = split_rel
            role_payloads.append(payload)
            role_summaries.append(split_summary)

        merged_payload = _merge_role_payloads(role_payloads)
        merged_by_role[role] = merged_payload
        summary_roles[role] = role_summaries

        out_json = ann_dir / f"instances_{role}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(merged_payload, f, ensure_ascii=False, indent=2)

    holdout_summary: dict[str, Any] = {}
    if holdout_test_fraction > 0 and holdout_test_source_roles:
        holdout_payloads: list[dict[str, Any]] = []
        for offset, role in enumerate(holdout_test_source_roles):
            normalized_role = str(role).strip().lower()
            if normalized_role not in merged_by_role:
                raise KeyError(f"holdout test source role 无效: {normalized_role}")
            remain_payload, holdout_payload, role_summary = _split_payload_holdout(
                merged_by_role[normalized_role],
                fraction=holdout_test_fraction,
                seed=holdout_test_seed + offset,
            )
            merged_by_role[normalized_role] = remain_payload
            holdout_payloads.append(holdout_payload)
            holdout_summary[normalized_role] = role_summary

        merged_by_role["test"] = _merge_role_payloads(holdout_payloads)
        for role in {"train", "val", "test"}:
            out_json = ann_dir / f"instances_{role}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(merged_by_role[role], f, ensure_ascii=False, indent=2)

    summary = {
        "status": "prepared",
        "source_type": "public_coco_instance",
        "dataset_root": str(out_dir),
        "source_dataset_root": str(dataset_root),
        "annotation_files": {
            "train": str(ann_dir / "instances_train.json"),
            "val": str(ann_dir / "instances_val.json"),
            "test": str(ann_dir / "instances_test.json"),
        },
        "counts": {
            "train_images": len(merged_by_role["train"]["images"]),
            "train_annotations": len(merged_by_role["train"]["annotations"]),
            "val_images": len(merged_by_role["val"]["images"]),
            "val_annotations": len(merged_by_role["val"]["annotations"]),
            "test_images": len(merged_by_role["test"]["images"]),
            "test_annotations": len(merged_by_role["test"]["annotations"]),
        },
        "roles": summary_roles,
        "holdout_test": {
            "enabled": bool(holdout_test_fraction > 0 and holdout_test_source_roles),
            "fraction": holdout_test_fraction,
            "seed": holdout_test_seed,
            "source_roles": holdout_test_source_roles,
            "splits": holdout_summary,
        },
    }
    dump_json(summary, out_dir / "prepare_summary.json")
    print(f"[OK] segmentation public coco dataset prepared: {out_dir}")


if __name__ == "__main__":
    main()
