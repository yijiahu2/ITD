from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pycocotools import mask as mask_utils


def read_instance_labels(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int32)


def score_from_mask(score_map: np.ndarray | None, mask: np.ndarray, mode: str) -> float:
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


def rle_bbox(rle: dict[str, Any]) -> list[float]:
    return [float(x) for x in mask_utils.toBbox(rle).tolist()]


def instances_from_label_image(
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
                "score": score_from_mask(score_map, mask, score_mode),
                "area": float(area),
                "bbox": rle_bbox(rle),
                "rle": rle,
            }
        )

    pred_instances.sort(key=lambda x: (-float(x["score"]), int(x["pred_id"])))
    return pred_instances


def instances_from_label_raster(
    *,
    image_id: int,
    y_inst_tif: str | Path,
    score_map: np.ndarray | None,
    score_mode: str,
) -> list[dict[str, Any]]:
    return instances_from_label_image(
        label_image=read_instance_labels(y_inst_tif),
        image_id=image_id,
        score_map=score_map,
        score_mode=score_mode,
    )
