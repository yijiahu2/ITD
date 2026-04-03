from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def read_rgb_image_with_profile(input_image: str) -> tuple[np.ndarray, dict[str, Any], Any, Any]:
    with rasterio.open(input_image) as src:
        arr = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    if arr.ndim != 3:
        raise ValueError(f"Expected raster with CHW layout, got shape={arr.shape}")

    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    elif arr.shape[0] == 1:
        rgb = np.repeat(np.transpose(arr[:1], (1, 2, 0)), 3, axis=2)
    else:
        raise ValueError(f"Unsupported band count: {arr.shape[0]}")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, profile, transform, crs


def export_rgb_png_from_raster(input_image: str, out_png: str) -> str:
    rgb, _, _, _ = read_rgb_image_with_profile(input_image)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_png)
    return str(out_png)


def read_binary_mask(mask_tif: str) -> np.ndarray:
    with rasterio.open(mask_tif) as ds:
        return (ds.read(1) > 0).astype(np.uint8)


def write_label_tif(out_path: str, label_img: np.ndarray, profile: dict[str, Any]) -> None:
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=rasterio.int32,
        compress="LZW",
        nodata=0,
        photometric="MINISBLACK",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(label_img.astype(np.int32), 1)


def write_instance_color_png(out_path: str, label_img: np.ndarray) -> None:
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_img)
    ids = ids[ids > 0]
    rng = np.random.default_rng(12345)
    color_map: dict[int, np.ndarray] = {}
    for inst_id in ids:
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        color_map[int(inst_id)] = color
        canvas[label_img == inst_id] = color
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def write_instance_scores_json(out_path: str, records: list[dict[str, Any]]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_prediction_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        masks = data["masks"]
        scores = data["scores"]
    masks = masks.astype(bool)
    scores = scores.astype(np.float32)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks, scores


def export_instances_to_shp(
    label_img: np.ndarray,
    transform: Any,
    crs: Any,
    out_shp: str,
    *,
    score_by_id: dict[int, float] | None = None,
    min_area_px: int = 1,
) -> None:
    import fiona

    score_by_id = score_by_id or {}
    mask = label_img > 0
    geom_groups: dict[int, list[Any]] = {}
    area_by_id: dict[int, float] = {}

    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area <= 0:
        pixel_area = 1.0

    for geom, val in features.shapes(label_img.astype(np.int32), mask=mask, transform=transform):
        inst_id = int(val)
        if inst_id <= 0:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        area_px = float(poly.area / pixel_area)
        if area_px < float(min_area_px):
            continue
        geom_groups.setdefault(inst_id, []).append(poly)
        area_by_id[inst_id] = float(area_by_id.get(inst_id, 0.0)) + area_px

    Path(out_shp).parent.mkdir(parents=True, exist_ok=True)
    crs_wkt = crs.to_wkt() if crs is not None and hasattr(crs, "to_wkt") else None
    schema = {
        "geometry": "Polygon",
        "properties": {"id": "int", "score": "float", "area_px": "float"},
    }
    with fiona.open(
        out_shp,
        "w",
        driver="ESRI Shapefile",
        crs_wkt=crs_wkt,
        schema=schema,
        encoding="UTF-8",
    ) as sink:
        for inst_id in sorted(geom_groups.keys()):
            geom = unary_union(geom_groups[inst_id])
            if geom.is_empty:
                continue
            sink.write(
                {
                    "geometry": mapping(geom),
                    "properties": {
                        "id": int(inst_id),
                        "score": float(score_by_id.get(inst_id, 0.0)),
                        "area_px": float(area_by_id.get(inst_id, 0.0)),
                    },
                }
            )


def ensure_vector_from_label_tif(
    label_tif: str,
    out_shp: str,
    *,
    score_json: str | None = None,
    min_area_px: int = 1,
) -> None:
    score_by_id: dict[int, float] = {}
    if score_json and Path(score_json).exists():
        try:
            records = json.loads(Path(score_json).read_text(encoding="utf-8"))
            for item in records:
                inst_id = item.get("instance_id")
                score = item.get("score")
                if inst_id is not None and score is not None:
                    score_by_id[int(inst_id)] = float(score)
        except Exception:
            score_by_id = {}

    with rasterio.open(label_tif) as ds:
        label_img = ds.read(1).astype(np.int32)
        transform = ds.transform
        crs = ds.crs

    export_instances_to_shp(
        label_img,
        transform,
        crs,
        out_shp,
        score_by_id=score_by_id,
        min_area_px=min_area_px,
    )


def materialize_segmentation_outputs_from_prediction_npz(
    *,
    input_image: str,
    m_sem_tif: str,
    pred_npz: str,
    outputs: dict[str, str],
    score_thr: float,
    min_area_px: int,
    min_sem_overlap_ratio: float,
    clip_to_msem: bool,
    max_instances: int | None = None,
) -> None:
    rgb, profile, transform, crs = read_rgb_image_with_profile(input_image)
    _ = rgb
    msem_mask = read_binary_mask(m_sem_tif)
    masks, scores = load_prediction_npz(pred_npz)
    label_img, records = build_label_image_from_masks(
        masks,
        scores,
        msem_mask,
        score_thr=score_thr,
        min_area_px=min_area_px,
        min_sem_overlap_ratio=min_sem_overlap_ratio,
        clip_to_msem=clip_to_msem,
        max_instances=max_instances,
    )
    score_by_id = {int(x["instance_id"]): float(x["score"]) for x in records}

    write_label_tif(outputs["y_inst_tif"], label_img, profile)
    export_instances_to_shp(
        label_img,
        transform,
        crs,
        outputs["y_inst_shp"],
        score_by_id=score_by_id,
        min_area_px=min_area_px,
    )
    write_instance_color_png(outputs["y_inst_color_png"], label_img)
    instance_scores_json = str(Path(outputs["y_inst_tif"]).with_name("instance_scores.json"))
    prediction_summary_json = str(Path(outputs["y_inst_tif"]).with_name("prediction_summary.json"))
    write_instance_scores_json(instance_scores_json, records)
    write_instance_scores_json(
        prediction_summary_json,
        [
            {
                "raw_prediction_count": int(len(scores)),
                "kept_instance_count": int(len(records)),
                "score_thr": score_thr,
                "min_area_px": min_area_px,
                "min_sem_overlap_ratio": min_sem_overlap_ratio,
                "clip_to_msem": clip_to_msem,
            }
        ],
    )


def build_label_image_from_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    msem_mask: np.ndarray,
    *,
    score_thr: float,
    min_area_px: int,
    min_sem_overlap_ratio: float,
    clip_to_msem: bool,
    max_instances: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h, w = msem_mask.shape
    label_img = np.zeros((h, w), dtype=np.int32)
    records: list[dict[str, Any]] = []

    if masks.size == 0:
        return label_img, records

    order = np.argsort(-scores)
    next_id = 1
    for rank, idx in enumerate(order):
        if max_instances is not None and next_id > max_instances:
            break
        score = float(scores[idx])
        if score < float(score_thr):
            continue

        mask = masks[idx].astype(bool)
        if mask.shape != (h, w):
            raise ValueError(f"Prediction mask shape mismatch: expected {(h, w)}, got {mask.shape}")

        raw_area = int(mask.sum())
        if raw_area < int(min_area_px):
            continue

        sem_overlap = int(np.logical_and(mask, msem_mask > 0).sum())
        sem_overlap_ratio = float(sem_overlap) / float(max(raw_area, 1))
        if sem_overlap_ratio < float(min_sem_overlap_ratio):
            continue

        if clip_to_msem:
            mask = np.logical_and(mask, msem_mask > 0)

        area_after_clip = int(mask.sum())
        if area_after_clip < int(min_area_px):
            continue

        assignable = np.logical_and(mask, label_img == 0)
        final_area = int(assignable.sum())
        if final_area < int(min_area_px):
            continue

        inst_id = next_id
        label_img[assignable] = inst_id
        records.append(
            {
                "instance_id": inst_id,
                "rank": int(rank),
                "score": score,
                "raw_area_px": raw_area,
                "sem_overlap_px": sem_overlap,
                "sem_overlap_ratio": sem_overlap_ratio,
                "final_area_px": final_area,
            }
        )
        next_id += 1

    return label_img, records
