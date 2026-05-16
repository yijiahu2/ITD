from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_SEGFORMER_MODEL_PATH = "/home/xth/tcd/models/tcd-segformer-mit-b5"
_SEGFORMER_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
_CELLPOSE_CACHE: dict[tuple[bool, str], Any] = {}


def _write_json(path: str | Path, payload: Any) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _load_rgb(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def _assert_runtime_device_available(device: str) -> None:
    import torch

    selected = str(device or "").strip()
    if not selected.startswith("cuda"):
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        f"Requested device {selected}, but CUDA is not available in the current process. "
        "Fix the shell/driver environment instead of falling back to CPU."
    )


def _load_segformer(model_path: str, device: str):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    model_dir = Path(model_path).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"SegFormer model path not found: {model_dir}")
    cache_key = (str(model_dir), str(device))
    cached = _SEGFORMER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    print(f"[main_model] load SegFormer model={model_dir} device={device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    cached = (processor, model, torch)
    _SEGFORMER_CACHE[cache_key] = cached
    return cached


def _load_cellpose_model(device: str):
    from cellpose import models

    use_gpu = str(device).startswith("cuda")
    cache_key = (use_gpu, "cpsam")
    cached = _CELLPOSE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    print(f"[main_model] load Cellpose-SAM gpu={use_gpu}", flush=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    _CELLPOSE_CACHE[cache_key] = model
    return model


def _predict_semantic_mask(
    rgb: np.ndarray,
    *,
    model_path: str,
    device: str,
    threshold: float,
    tile_size: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    processor, model, torch = _load_segformer(model_path, device)
    height, width = rgb.shape[:2]
    tile_size = max(int(tile_size), 1)
    overlap = max(min(int(overlap), tile_size - 1), 0)
    step = max(tile_size - overlap, 1)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, height, step):
            for x in range(0, width, step):
                y1 = min(y + tile_size, height)
                x1 = min(x + tile_size, width)
                tile = rgb[y:y1, x:x1]
                inputs = processor(images=tile, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=tile.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0]
                probs = torch.softmax(logits, dim=0)
                if probs.shape[0] > 1:
                    canopy = probs[1]
                else:
                    canopy = probs[0]
                prob = canopy.detach().cpu().numpy().astype(np.float32)
                prob_sum[y:y1, x:x1] += prob
                weight[y:y1, x:x1] += 1.0

    probability = prob_sum / np.maximum(weight, 1e-6)
    semantic_mask = (probability >= float(threshold)).astype(np.uint8)
    return semantic_mask, probability


def _run_cellpose_sam(
    rgb: np.ndarray,
    semantic_mask: np.ndarray,
    *,
    device: str,
    diam_list: list[float],
    use_rgb: bool,
    bsize: int,
    tile_overlap: float,
    augment: bool,
    niter: int,
) -> np.ndarray:
    cp = _load_cellpose_model(device)
    device_rgb = rgb.copy()
    if np.any(semantic_mask > 0):
        median = np.median(device_rgb[semantic_mask > 0], axis=0).astype(np.uint8)
        device_rgb[semantic_mask == 0] = median
    else:
        device_rgb[:] = 0
    channels = [0, 0] if use_rgb else [0, 0]

    label = np.zeros(semantic_mask.shape, dtype=np.int32)
    next_id = 1
    for diameter in diam_list:
        print(f"[main_model] Cellpose-SAM start diameter={float(diameter):.1f}", flush=True)
        try:
            masks, _flows, _styles = cp.eval(
                device_rgb,
                diameter=float(diameter),
                channels=channels,
                flow_threshold=1.0,
                cellprob_threshold=0.0,
                bsize=int(bsize),
                tile_overlap=float(tile_overlap),
                augment=bool(augment),
                niter=int(niter),
            )
        except TypeError:
            masks, _flows, _styles = cp.eval(
                device_rgb,
                diameter=float(diameter),
                channels=channels,
                flow_threshold=1.0,
                cellprob_threshold=0.0,
            )
        masks = np.asarray(masks, dtype=np.int32)
        masks[semantic_mask == 0] = 0
        before_id = next_id
        for local_id in np.unique(masks):
            if int(local_id) <= 0:
                continue
            candidate = masks == int(local_id)
            assignable = np.logical_and(candidate, label == 0)
            if int(assignable.sum()) <= 0:
                continue
            label[assignable] = next_id
            next_id += 1
        print(
            f"[main_model] Cellpose-SAM done diameter={float(diameter):.1f} "
            f"new_instances={next_id - before_id}",
            flush=True,
        )
    return label


def _label_to_color(label_img: np.ndarray) -> np.ndarray:
    max_id = int(label_img.max())
    canvas = np.zeros(label_img.shape + (3,), dtype=np.uint8)
    if max_id <= 0:
        return canvas
    rng = np.random.default_rng(12345)
    colors = rng.integers(40, 255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[label_img]


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _rle_from_mask(mask: np.ndarray) -> dict[str, Any] | None:
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": rle["counts"]}


def _instances_from_label(
    label_img: np.ndarray,
    probability: np.ndarray,
    *,
    image_id: int | str,
    min_area_px: int,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for inst_id in sorted(int(value) for value in np.unique(label_img) if int(value) > 0):
        mask = label_img == inst_id
        area = int(mask.sum())
        if area < int(min_area_px):
            continue
        bbox = _bbox_from_mask(mask)
        record: dict[str, Any] = {
            "image_id": int(image_id) if str(image_id).isdigit() else image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": float(area),
            "score": float(probability[mask].mean()) if area else 0.0,
            "instance_id": inst_id,
        }
        rle = _rle_from_mask(mask)
        if rle:
            record["segmentation"] = rle
        instances.append(record)
    return instances


def infer_one_image(
    *,
    image_path: str | Path,
    output_dir: str | Path,
    image_id: int | str,
    model_cfg: dict[str, Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    cfg = dict(model_cfg or {})
    runtime_cfg = dict(cfg.get("runtime") or {})
    stage1_cfg = dict(cfg.get("stage1_semantic_prior") or {})
    stage2_cfg = dict(cfg.get("stage2_instance") or {})
    postprocess_cfg = dict(cfg.get("postprocess") or {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_device = str(device or runtime_cfg.get("device") or "cuda:0")
    _assert_runtime_device_available(selected_device)
    print(f"[main_model] image_id={image_id} load image={image_path}", flush=True)
    rgb = _load_rgb(image_path)
    print(f"[main_model] image_id={image_id} semantic prior start", flush=True)
    semantic_mask, probability = _predict_semantic_mask(
        rgb,
        model_path=str(stage1_cfg.get("semantic_prior_model_path") or stage1_cfg.get("model_path") or DEFAULT_SEGFORMER_MODEL_PATH),
        device=selected_device,
        threshold=float(stage1_cfg.get("semantic_thr", 0.5)),
        tile_size=int(stage1_cfg.get("semantic_tile_size", 1024)),
        overlap=int(stage1_cfg.get("semantic_overlap", 0)),
    )
    print(
        f"[main_model] image_id={image_id} semantic prior done "
        f"semantic_fraction={float(semantic_mask.mean()):.6f}",
        flush=True,
    )
    diam_raw = stage2_cfg.get("diam_list", "96,192,320")
    diam_list = [float(item) for item in str(diam_raw).split(",") if str(item).strip()]
    print(f"[main_model] image_id={image_id} instance segmentation start", flush=True)
    instance_mask = _run_cellpose_sam(
        rgb,
        semantic_mask,
        device=selected_device,
        diam_list=diam_list or [96.0, 192.0, 320.0],
        use_rgb=bool(stage2_cfg.get("use_rgb", True)),
        bsize=int(stage2_cfg.get("bsize", 256)),
        tile_overlap=float(stage2_cfg.get("tile_overlap", 0.35)),
        augment=bool(stage2_cfg.get("augment", True)),
        niter=int(stage2_cfg.get("niter", 0)),
    )
    print(
        f"[main_model] image_id={image_id} instance segmentation done "
        f"raw_instances={int(instance_mask.max())}",
        flush=True,
    )

    semantic_png = out_dir / "semantic_mask.png"
    semantic_npy = out_dir / "semantic_mask.npy"
    instance_png = out_dir / "instance_mask.png"
    instance_npy = out_dir / "instance_mask.npy"
    Image.fromarray((semantic_mask * 255).astype(np.uint8)).save(semantic_png)
    np.save(semantic_npy, semantic_mask.astype(np.uint8))
    Image.fromarray(_label_to_color(instance_mask)).save(instance_png)
    np.save(instance_npy, instance_mask.astype(np.int32))

    instances = _instances_from_label(
        instance_mask,
        probability,
        image_id=image_id,
        min_area_px=int(postprocess_cfg.get("min_area_px") or 20),
    )
    prediction_path = out_dir / "prediction.json"
    summary_path = out_dir / "summary.json"
    _write_json(prediction_path, instances)
    summary = {
        "status": "completed",
        "image_id": image_id,
        "image_path": str(image_path),
        "semantic_fraction": float(semantic_mask.mean()),
        "instance_count": len(instances),
        "artifacts": {
            "semantic_mask_png": str(semantic_png),
            "semantic_mask_npy": str(semantic_npy),
            "instance_mask_png": str(instance_png),
            "instance_mask_npy": str(instance_npy),
            "prediction_json": str(prediction_path),
        },
    }
    _write_json(summary_path, summary)
    print(
        f"[main_model] image_id={image_id} completed "
        f"instances={len(instances)} output={summary_path}",
        flush=True,
    )
    return {**summary, "instances": instances, "summary_json": str(summary_path)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = {}
    if args.config_json:
        cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    result = infer_one_image(
        image_path=args.image,
        output_dir=args.output_dir,
        image_id=args.image_id,
        model_cfg=cfg,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
