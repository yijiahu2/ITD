from __future__ import annotations

import os
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


MODEL_PATH = "/home/xth/tcd/models/tcd-segformer-mit-b5"

TILE_SIZE = 1024
OVERLAP = 256
THR = 0.25

MIN_AREA_M2 = 0.0
SIMPLIFY_TOL = 0.0


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    rgb = arr[:3].astype(np.float32)

    if rgb.max() > 255:
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            channel = rgb[c]
            lo, hi = np.percentile(channel, (1, 99))
            if hi <= lo:
                hi = lo + 1
            channel = (channel - lo) / (hi - lo)
            channel = np.clip(channel, 0, 1) * 255
            out[c] = channel.astype(np.uint8)
        rgb = out
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return np.transpose(rgb, (1, 2, 0))


def _extract_state_dict(ckpt_obj: dict) -> dict:
    if not isinstance(ckpt_obj, dict):
        raise ValueError(f"Unsupported checkpoint object type: {type(ckpt_obj)}")

    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        state = ckpt_obj["state_dict"]
    elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        state = ckpt_obj["model"]
    else:
        state = ckpt_obj

    cleaned = {}
    for key, value in state.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("model."):
            normalized_key = normalized_key[len("model.") :]
        cleaned[normalized_key] = value

    return cleaned


def load_finetuned_ckpt_if_needed(model, ckpt_path: str | None, device: str):
    if not ckpt_path:
        print("[INFO] No finetuned ckpt provided, use base pretrained weights only.")
        return model

    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    print(f"[INFO] Loading finetuned ckpt: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(obj)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] finetuned ckpt loaded: {ckpt_path}")
    print(f"[INFO] missing keys: {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    if missing:
        print("[INFO] first missing keys:", missing[:10])
    if unexpected:
        print("[INFO] first unexpected keys:", unexpected[:10])

    return model


@torch.no_grad()
def predict_tile(rgb: np.ndarray, model, processor, device: str) -> np.ndarray:
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0]

    probs = torch.softmax(logits, dim=0)
    canopy = probs[1]
    return (canopy >= THR).cpu().numpy().astype(np.uint8)


def sliding_window_predict(img: np.ndarray, model, processor, device: str) -> np.ndarray:
    height, width = img.shape[:2]
    step = TILE_SIZE - OVERLAP

    result = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)

    for y in range(0, height, step):
        for x in range(0, width, step):
            y1 = min(y + TILE_SIZE, height)
            x1 = min(x + TILE_SIZE, width)

            tile = img[y:y1, x:x1]
            pad_h = TILE_SIZE - tile.shape[0]
            pad_w = TILE_SIZE - tile.shape[1]

            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            pred = predict_tile(tile, model, processor, device)
            pred = pred[: y1 - y, : x1 - x].astype(np.float32)

            result[y:y1, x:x1] += pred
            weight[y:y1, x:x1] += 1.0

    result = result / np.maximum(weight, 1e-6)
    return (result > 0.5).astype(np.uint8)


def write_tif(out_path: str, data: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress="LZW",
        nodata=0,
        photometric="MINISBLACK",
    )

    with rasterio.open(out_path, "w", **output_profile) as dst:
        dst.write(data.astype(np.uint8), 1)


def mask_to_shp(
    mask: np.ndarray,
    transform,
    crs,
    out_shp: str,
    min_area_m2: float = 0.0,
    simplify_tol: float = 0.0,
) -> None:
    os.makedirs(os.path.dirname(out_shp), exist_ok=True)

    geom_iter = shapes(mask.astype(np.uint8), mask=(mask == 1), transform=transform)

    geoms = []
    vals = []
    for geom, val in geom_iter:
        if int(val) != 1:
            continue
        geometry = shape(geom)
        if geometry.is_empty:
            continue
        if simplify_tol and simplify_tol > 0:
            geometry = geometry.simplify(simplify_tol, preserve_topology=True)
        if geometry.is_empty:
            continue
        geoms.append(geometry)
        vals.append(int(val))

    if not geoms:
        print("[WARN] No canopy polygons found, skip shp.")
        return

    gdf = gpd.GeoDataFrame({"class": vals}, geometry=geoms, crs=crs)

    if min_area_m2 and min_area_m2 > 0:
        try:
            areas = gdf.geometry.area
            gdf = gdf.loc[areas >= float(min_area_m2)].copy()
        except Exception as exc:
            print("[WARN] area filter failed:", exc)

    gdf.reset_index(drop=True, inplace=True)
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
    print("Saved:", out_shp)
    print("Polygons:", len(gdf))


def main(in_tif: str, out_dir: str, ckpt: str | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH, local_files_only=True).to(device).eval()
    model = load_finetuned_ckpt_if_needed(model, ckpt, device)
    model = model.to(device).eval()

    with rasterio.open(in_tif) as src:
        img = src.read()
        rgb = to_uint8_rgb(img)
        print("Image size:", rgb.shape)

        mask = sliding_window_predict(rgb, model, processor, device)

        out_tif = os.path.join(out_dir, "M_sem.tif")
        write_tif(out_tif, mask, src.profile)

        out_shp = os.path.join(out_dir, "M_sem.shp")
        mask_to_shp(
            mask=mask,
            transform=src.transform,
            crs=src.crs,
            out_shp=out_shp,
            min_area_m2=MIN_AREA_M2,
            simplify_tol=SIMPLIFY_TOL,
        )

    out_png = os.path.join(out_dir, "M_sem.png")
    cv2.imwrite(out_png, (mask * 255).astype(np.uint8))

    print("Saved:", out_tif)
    print("Saved:", out_png)
    print("Canopy fraction:", float(mask.mean()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tif", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt", default=None, help="Optional finetuned checkpoint path.")
    args = parser.parse_args()

    main(args.in_tif, args.out_dir, args.ckpt)
