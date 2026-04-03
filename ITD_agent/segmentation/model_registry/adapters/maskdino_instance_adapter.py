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
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.DEVICE = device

    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0

    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image_bgr)
    instances = outputs.get("instances")
    if instances is None:
        return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool), np.zeros((0,), dtype=np.float32)

    instances = instances.to("cpu")
    scores = _to_numpy(getattr(instances, "scores", None))
    masks = _to_numpy(getattr(instances, "pred_masks", None))
    if scores is None or masks is None:
        return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool), np.zeros((0,), dtype=np.float32)

    masks = masks.astype(bool)
    scores = scores.astype(np.float32)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks, scores


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

    repo_root = str(cfg.get("repo_root") or "/home/xth/MaskDINO")
    config_file = str(cfg["config_file"])
    checkpoint = str(cfg["checkpoint"])
    device = str(cfg.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA requested but unavailable, falling back to cpu: {device}")
        device = "cpu"
    image_bgr = cv2.imread(args.input_png, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read input_png: {args.input_png}")

    masks, scores = _load_predictions(
        repo_root=repo_root,
        config_file=config_file,
        checkpoint=checkpoint,
        device=device,
        image_bgr=image_bgr,
    )
    np.savez_compressed(args.pred_npz, masks=masks.astype(np.uint8), scores=scores.astype(np.float32))


if __name__ == "__main__":
    main()
