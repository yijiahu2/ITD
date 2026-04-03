from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Tuple


_MODULE_CACHE: dict[tuple[str, str], Any] = {}
_SEMANTIC_PRIOR_CACHE: dict[tuple[str, str, str | None], tuple[Any, Any, str, Any]] = {}
_SEGMENTATION_CACHE: dict[tuple[str, str], tuple[Any, str, Any]] = {}


def _load_module(work_dir: str, script_name: str):
    key = (work_dir, script_name)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    script_path = Path(work_dir) / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"runtime script not found: {script_path}")

    module_name = f"_cached_runtime_{script_path.stem}_{abs(hash(str(script_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def _normalize_tiff_profile(profile: Dict[str, Any], *, count: int, dtype: Any, nodata: Any) -> Dict[str, Any]:
    out = dict(profile)
    out["driver"] = "GTiff"
    out["count"] = count
    out["dtype"] = dtype
    out["nodata"] = nodata
    out["compress"] = "LZW"
    out["photometric"] = "MINISBLACK"
    return out


def _get_semantic_prior_cached_resources(cfg: Dict[str, Any]) -> tuple[Any, Any, str, Any]:
    mod = _load_module(cfg["work_dir"], cfg["semantic_prior_script"])
    device = "cuda" if mod.torch.cuda.is_available() else "cpu"
    cache_key = (cfg["semantic_prior_script"], str(mod.MODEL_PATH), cfg.get("semantic_prior_ckpt"))
    cached = _SEMANTIC_PRIOR_CACHE.get(cache_key)
    if cached is None or cached[2] != device:
        processor = mod.AutoImageProcessor.from_pretrained(mod.MODEL_PATH, local_files_only=True)
        model = mod.SegformerForSemanticSegmentation.from_pretrained(mod.MODEL_PATH, local_files_only=True).to(device).eval()
        model = mod.load_finetuned_ckpt_if_needed(model, cfg.get("semantic_prior_ckpt"), device)
        model = model.to(device).eval()
        cached = (processor, model, device, mod)
        _SEMANTIC_PRIOR_CACHE[cache_key] = cached
    return cached


def _predict_canopy_probability(tile_rgb, model, processor, device, mod):
    with mod.torch.no_grad():
        inputs = processor(images=tile_rgb, return_tensors="pt").to(device)
        out = model(**inputs)
        logits = out.logits
        logits = mod.torch.nn.functional.interpolate(
            logits,
            size=tile_rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0]
        probs = mod.torch.softmax(logits, dim=0)
        canopy = probs[1].detach().cpu().numpy().astype(mod.np.float32)
    return canopy


def _sliding_window_predict_with_probs(img_rgb, model, processor, device, mod):
    h, w = img_rgb.shape[:2]
    tile_size = int(mod.TILE_SIZE)
    overlap = int(mod.OVERLAP)
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"Invalid semantic prior tile setup: TILE_SIZE={tile_size}, OVERLAP={overlap}")

    vote_sum = mod.np.zeros((h, w), dtype=mod.np.float32)
    prob_sum = mod.np.zeros((h, w), dtype=mod.np.float32)
    weight = mod.np.zeros((h, w), dtype=mod.np.float32)

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            tile = img_rgb[y:y1, x:x1]

            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = mod.cv2.copyMakeBorder(
                    tile,
                    0,
                    pad_h,
                    0,
                    pad_w,
                    mod.cv2.BORDER_REFLECT,
                )

            canopy_prob = _predict_canopy_probability(tile, model, processor, device, mod)
            canopy_prob = canopy_prob[: y1 - y, : x1 - x]
            canopy_vote = (canopy_prob >= float(mod.THR)).astype(mod.np.float32)

            prob_sum[y:y1, x:x1] += canopy_prob
            vote_sum[y:y1, x:x1] += canopy_vote
            weight[y:y1, x:x1] += 1.0

    mean_prob = prob_sum / mod.np.maximum(weight, 1e-6)
    mean_vote = vote_sum / mod.np.maximum(weight, 1e-6)
    mask = (mean_vote > 0.5).astype(mod.np.uint8)
    return mask, mean_prob.astype(mod.np.float32)


def predict_semantic_prior_cached(cfg: Dict[str, Any]) -> Dict[str, Any]:
    processor, model, device, mod = _get_semantic_prior_cached_resources(cfg)
    with mod.rasterio.open(cfg["input_image"]) as src:
        img = src.read()
        rgb = mod.to_uint8_rgb(img)
        mask, prob = _sliding_window_predict_with_probs(rgb, model, processor, device, mod)
        return {
            "mask": mask,
            "probability": prob,
            "profile": src.profile.copy(),
            "tiff_profile_uint8": _normalize_tiff_profile(
                src.profile.copy(),
                count=1,
                dtype=mod.rasterio.uint8,
                nodata=0,
            ),
            "transform": src.transform,
            "crs": src.crs,
            "rgb_shape": rgb.shape,
            "device": device,
            "module": mod,
        }


def run_semantic_prior_cached(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = {
        "m_sem_tif": str(Path(cfg["output_dir"]) / "M_sem.tif"),
        "m_sem_png": str(Path(cfg["output_dir"]) / "M_sem.png"),
    }
    os.makedirs(cfg["output_dir"], exist_ok=True)
    pred = predict_semantic_prior_cached(cfg)
    mod = pred["module"]
    mask = pred["mask"]
    out_tif = paths["m_sem_tif"]

    print("Device:", pred["device"])
    print("Image size:", pred["rgb_shape"])
    mod.write_tif(out_tif, mask, pred["tiff_profile_uint8"])

    out_shp = os.path.join(cfg["output_dir"], "M_sem.shp")
    mod.mask_to_shp(
        mask=mask,
        transform=pred["transform"],
        crs=pred["crs"],
        out_shp=out_shp,
        min_area_m2=mod.MIN_AREA_M2,
        simplify_tol=mod.SIMPLIFY_TOL,
    )

    png = (mask * 255).astype(mod.np.uint8)
    out_png = paths["m_sem_png"]
    mod.cv2.imwrite(out_png, png)

    print("Saved:", out_tif)
    print("Saved:", out_png)
    print("Canopy fraction:", float(mask.mean()))
    return {
        "cmd": ["cached_semantic_prior", cfg["input_image"], cfg["output_dir"]],
        "m_sem_tif": paths["m_sem_tif"],
        "m_sem_png": paths["m_sem_png"],
    }


def _run_segmentation_cached_legacy(cfg: Dict[str, Any], m_sem_tif: str) -> Dict[str, Any]:
    mod = _load_module(cfg["work_dir"], cfg["segmentation_script"])
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if mod.torch.cuda.is_available() else "cpu"
    cache_key = (cfg["segmentation_script"], device)
    cached = _SEGMENTATION_CACHE.get(cache_key)
    if cached is None:
        cp = mod.models.CellposeModel(gpu=(device == "cuda"), pretrained_model="cpsam")
        _SEGMENTATION_CACHE[cache_key] = (cp, device, mod)
    else:
        cp, device, mod = cached

    print("Device:", device)
    diam_list = tuple(float(x) for x in str(cfg["diam_list"]).split(",") if str(x).strip())
    iou_merge_thr = float(cfg["iou_merge_thr"])
    tile = int(cfg["tile"])
    overlap = int(cfg["overlap"])
    bsize = int(cfg["bsize"])
    tile_overlap = float(cfg["tile_overlap"])
    augment = bool(cfg.get("augment", True))
    niter = int(cfg.get("niter", 0))
    use_gray = not bool(cfg.get("use_rgb", False))

    print("diam_list:", diam_list)
    print("TILE:", tile, "OVERLAP:", overlap)
    print("cellpose bsize:", bsize, "tile_overlap:", tile_overlap, "augment:", augment, "niter:", niter)

    step = tile - overlap
    if step <= 0:
        raise ValueError("tile must be > overlap")

    with mod.rasterio.open(cfg["input_image"]) as src, mod.rasterio.open(m_sem_tif) as ms:
        H, W = src.height, src.width
        profile = _normalize_tiff_profile(
            src.profile.copy(),
            count=1,
            dtype=mod.rasterio.int32,
            nodata=0,
        )

        inst_full = mod.np.zeros((H, W), dtype=mod.np.int32)
        weight_full = mod.np.zeros((H, W), dtype=mod.np.float32)
        next_id = 1

        for y in range(0, H, step):
            for x in range(0, W, step):
                h = min(tile, H - y)
                w = min(tile, W - x)
                win = mod.Window(x, y, w, h)

                rgb, nodata = mod.read_rgb_window(src, win)
                msem = mod.read_mask_window(ms, win)
                valid = (msem == 1) & (~nodata)

                rgb_in = rgb.copy()
                if valid.any():
                    med = mod.np.median(rgb_in[valid], axis=0).astype(mod.np.uint8)
                    rgb_in[~valid] = med
                else:
                    rgb_in[:] = 0

                img_in = rgb_in
                channels = [0, 0] if use_gray else [0, 0]

                pad_h = tile - h
                pad_w = tile - w
                if pad_h > 0 or pad_w > 0:
                    img_in = mod.cv2.copyMakeBorder(img_in, 0, pad_h, 0, pad_w, mod.cv2.BORDER_REFLECT)

                masks_list = []
                for d in diam_list:
                    masks = mod.cellpose_eval_safe(
                        cp,
                        img_in,
                        diameter=d,
                        flow_thr=mod.FLOW_THR,
                        cellprob_thr=mod.CELLPROB_THR,
                        channels=channels,
                        bsize=bsize,
                        tile_overlap=tile_overlap,
                        augment=augment,
                        niter=niter,
                    )
                    masks = masks[:h, :w].astype(mod.np.int32)
                    if valid.any():
                        masks[~valid] = 0
                    masks_list.append(masks)

                tile_local = mod.merge_tile_multiscale(masks_list, iou_merge_thr=iou_merge_thr, min_area=mod.MIN_AREA)
                if tile_local.max() == 0:
                    continue

                old = inst_full[y:y + h, x:x + w]
                tile_global, next_id = mod.assign_global_ids(tile_local, old, iou_merge_thr=iou_merge_thr, global_next_id=next_id)
                w_tile = mod.feather_weight(h, w, overlap=overlap) * valid.astype(mod.np.float32)
                w_old = weight_full[y:y + h, x:x + w]
                take = (w_tile > w_old) & (tile_global > 0)

                old_updated = old.copy()
                old_updated[take] = tile_global[take]
                w_updated = w_old.copy()
                w_updated[take] = w_tile[take]

                inst_full[y:y + h, x:x + w] = old_updated
                weight_full[y:y + h, x:x + w] = w_updated
                print(f"[tile] x={x} y={y} local_max={int(tile_local.max())} global_next_id={next_id}")

        out_tif = os.path.join(out_dir, "Y_inst.tif")
        with mod.rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(inst_full.astype(mod.np.int32), 1)
        print("Saved:", out_tif)

        out_shp = os.path.join(out_dir, "Y_inst.shp")
        mod.export_instances_to_shp(inst_full, transform=src.transform, crs=src.crs, out_shp=out_shp, min_area_px=mod.MIN_AREA)

    out_png = os.path.join(out_dir, "Y_inst_color.png")
    max_id = int(inst_full.max())
    rng = mod.np.random.default_rng(0)
    lut = mod.np.zeros((max_id + 1, 3), dtype=mod.np.uint8)
    if max_id > 0:
        lut[1:] = rng.integers(0, 255, size=(max_id, 3), dtype=mod.np.uint8)
    color = lut[inst_full]
    mod.cv2.imwrite(out_png, mod.cv2.cvtColor(color, mod.cv2.COLOR_RGB2BGR))
    print("Saved:", out_png)
    print("instances:", int(inst_full.max()))
    return {
        "cmd": ["cached_segmentation", cfg["input_image"], m_sem_tif, out_dir],
        "y_inst_tif": os.path.join(out_dir, "Y_inst.tif"),
        "y_inst_shp": os.path.join(out_dir, "Y_inst.shp"),
        "y_inst_color_png": out_png,
    }


def run_segmentation_cached(cfg: Dict[str, Any], m_sem_tif: str) -> Dict[str, Any]:
    algorithm = str(cfg.get("segmentation_algorithm", "")).strip().lower()
    if algorithm and algorithm != "legacy_cellpose_sam":
        from ITD_agent.segmentation.model_registry import run_segmentation_algorithm

        return run_segmentation_algorithm(cfg, m_sem_tif)

    return _run_segmentation_cached_legacy(cfg, m_sem_tif)
