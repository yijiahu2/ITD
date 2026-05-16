from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _load_predictions(
    *,
    repo_root: str,
    config_file: str,
    checkpoint: str,
    device: str,
    score_thr: float,
    image_bgr: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    tile_batch_size: int,
    merge_iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from mmdet.apis import inference_detector, init_detector

    model = init_detector(config_file, checkpoint, device=device)
    height, width = image_bgr.shape[:2]

    if tile_size <= 0 or max(height, width) <= tile_size:
        result = inference_detector(model, image_bgr)
    else:
        step = max(tile_size - tile_overlap, 1)
        tiles = []
        offsets = []
        for y in range(0, height, step):
            for x in range(0, width, step):
                y1 = min(y + tile_size, height)
                x1 = min(x + tile_size, width)
                tiles.append(image_bgr[y:y1, x:x1])
                offsets.append((x, y))

        candidate_masks = []
        candidate_scores = []
        for start in range(0, len(tiles), max(tile_batch_size, 1)):
            batch = tiles[start:start + max(tile_batch_size, 1)]
            batch_results = inference_detector(model, batch)
            if isinstance(batch_results, list):
                result_list = batch_results
            else:
                result_list = [batch_results]

            for local_idx, result in enumerate(result_list):
                tile_index = start + local_idx
                offset_x, offset_y = offsets[tile_index]
                pred_instances = getattr(result, "pred_instances", None)
                if pred_instances is None:
                    continue
                scores = _to_numpy(getattr(pred_instances, "scores", None))
                masks = _to_numpy(getattr(pred_instances, "masks", None))
                if scores is None or masks is None:
                    continue
                if masks.ndim == 2:
                    masks = masks[None, ...]
                masks = masks.astype(bool)
                scores = scores.astype(np.float32)
                keep = scores >= float(score_thr)
                if not np.any(keep):
                    continue
                masks = masks[keep]
                scores = scores[keep]

                for mask, score in zip(masks, scores):
                    full_mask = np.zeros((height, width), dtype=bool)
                    h, w = mask.shape
                    full_mask[offset_y:offset_y + h, offset_x:offset_x + w] = mask
                    candidate_masks.append(full_mask)
                    candidate_scores.append(float(score))

        if not candidate_masks:
            return np.zeros((0, height, width), dtype=bool), np.zeros((0,), dtype=np.float32)

        masks = np.stack(candidate_masks, axis=0)
        scores = np.asarray(candidate_scores, dtype=np.float32)
        return _mask_nms(masks, scores, float(merge_iou_thr))

    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is None:
        return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool), np.zeros((0,), dtype=np.float32)

    scores = _to_numpy(getattr(pred_instances, "scores", None))
    masks = _to_numpy(getattr(pred_instances, "masks", None))
    if scores is None or masks is None:
        return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool), np.zeros((0,), dtype=np.float32)

    masks = masks.astype(bool)
    scores = scores.astype(np.float32)
    if masks.ndim == 2:
        masks = masks[None, ...]
    keep = scores >= float(score_thr)
    if not np.any(keep):
        return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool), np.zeros((0,), dtype=np.float32)
    masks = masks[keep]
    scores = scores[keep]
    return masks, scores


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_iou(
    a: np.ndarray,
    b: np.ndarray,
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
    area_a: int,
    area_b: int,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    a_crop = a[iy1:iy2, ix1:ix2]
    b_crop = b[iy1:iy2, ix1:ix2]
    inter = np.logical_and(a_crop, b_crop).sum()
    if inter == 0:
        return 0.0

    union = area_a + area_b - int(inter)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _mask_nms(masks: np.ndarray, scores: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    keep_masks = []
    keep_bboxes = []
    keep_areas = []
    keep_scores = []
    bboxes = [_mask_bbox(mask) for mask in masks]
    areas = [int(mask.sum()) for mask in masks]
    for idx in order:
        mask = masks[idx]
        bbox = bboxes[idx]
        area = areas[idx]
        score = float(scores[idx])
        suppressed = False
        for kept, kept_bbox, kept_area in zip(keep_masks, keep_bboxes, keep_areas):
            if _mask_iou(mask, kept, bbox, kept_bbox, area, kept_area) >= iou_thr:
                suppressed = True
                break
        if suppressed:
            continue
        keep_masks.append(mask)
        keep_bboxes.append(bbox)
        keep_areas.append(area)
        keep_scores.append(score)

    if not keep_masks:
        return np.zeros((0,) + masks.shape[1:], dtype=bool), np.zeros((0,), dtype=np.float32)
    return np.stack(keep_masks, axis=0), np.asarray(keep_scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--input_png", required=True)
    parser.add_argument("--pred_npz", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--msem_tif", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--algorithm_name", required=True)
    parser.add_argument("--y_inst_tif", required=True)
    parser.add_argument("--y_inst_shp", required=True)
    parser.add_argument("--y_inst_color_png", required=True)
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_root = str(cfg.get("repo_root") or "/home/xth/mmdetection331")
    config_file = str(cfg["config_file"])
    checkpoint = str(cfg["checkpoint"])
    device = str(cfg.get("device") or "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device}, but CUDA is not available in the current process. "
            "Fix the shell/driver environment instead of falling back to CPU."
        )
    score_thr = float(cfg.get("score_thr", 0.2))
    tile_size = int(cfg.get("tile_size", 1536))
    tile_overlap = int(cfg.get("tile_overlap", 256))
    tile_batch_size = int(cfg.get("tile_batch_size", 1))
    merge_iou_thr = float(cfg.get("merge_iou_thr", 0.4))
    image_bgr = cv2.imread(args.input_png, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read input_png: {args.input_png}")

    masks, scores = _load_predictions(
        repo_root=repo_root,
        config_file=config_file,
        checkpoint=checkpoint,
        device=device,
        score_thr=score_thr,
        image_bgr=image_bgr,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_batch_size=tile_batch_size,
        merge_iou_thr=merge_iou_thr,
    )
    np.savez_compressed(args.pred_npz, masks=masks.astype(np.uint8), scores=scores.astype(np.float32))


if __name__ == "__main__":
    main()
