from __future__ import annotations

import os

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from cellpose import models
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import shape


FLOW_THR = 1.0
CELLPROB_THR = 0.0
NODATA_VAL = 256
MIN_AREA = 50


def read_rgb_window(ds, win: Window) -> tuple[np.ndarray, np.ndarray]:
    rgb = ds.read([1, 2, 3], window=win)
    nodata = (rgb == NODATA_VAL).any(axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if nodata.any():
        rgb[nodata] = 0
    return rgb, nodata


def read_mask_window(ds, win: Window) -> np.ndarray:
    mask = ds.read(1, window=win)
    return (mask > 0).astype(np.uint8)


def remove_small_instances(lbl: np.ndarray, min_area: int) -> np.ndarray:
    if lbl.max() == 0:
        return lbl.astype(np.int32)
    out = np.zeros_like(lbl, dtype=np.int32)
    new_id = 1
    for instance_id in range(1, int(lbl.max()) + 1):
        instance_mask = lbl == instance_id
        if int(instance_mask.sum()) >= min_area:
            out[instance_mask] = new_id
            new_id += 1
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def feather_weight(height: int, width: int, overlap: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    d_top = yy
    d_left = xx
    d_bottom = (height - 1) - yy
    d_right = (width - 1) - xx
    d_edge = np.minimum(np.minimum(d_top, d_bottom), np.minimum(d_left, d_right)).astype(np.float32)
    if overlap <= 0:
        return np.ones((height, width), np.float32)
    return np.clip(d_edge / float(overlap), 0.0, 1.0)


def export_instances_to_shp(label_img: np.ndarray, transform, crs, out_shp: str, min_area_px: int = 50) -> None:
    mask = label_img > 0
    geoms = []
    ids = []
    areas_px = []

    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area <= 0:
        pixel_area = 1.0

    for geom, val in features.shapes(label_img.astype(np.int32), mask=mask, transform=transform):
        instance_id = int(val)
        if instance_id <= 0:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        area_px = float(poly.area / pixel_area)
        if area_px < float(min_area_px):
            continue
        geoms.append(poly)
        ids.append(instance_id)
        areas_px.append(area_px)

    if not geoms:
        print("[shp] No polygons to write (empty instances).")
        return

    gdf = gpd.GeoDataFrame({"id": ids, "area_px": areas_px}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve(by="id", as_index=False, aggfunc={"area_px": "sum"})
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="UTF-8")
    print("Saved:", out_shp, "features:", len(gdf))


def cellpose_eval_safe(
    cp,
    img: np.ndarray,
    diameter: float,
    flow_thr: float,
    cellprob_thr: float,
    channels: list[int],
    bsize: int = 256,
    tile_overlap: float = 0.35,
    augment: bool = True,
    niter: int = 0,
):
    kwargs = dict(
        diameter=float(diameter),
        channels=channels,
        flow_threshold=float(flow_thr),
        cellprob_threshold=float(cellprob_thr),
        bsize=bsize,
        tile_overlap=tile_overlap,
        augment=augment,
        niter=niter,
    )
    try:
        masks, _flows, _styles = cp.eval(img, **kwargs)
        return masks
    except TypeError:
        fallback_kwargs = dict(
            diameter=float(diameter),
            channels=channels,
            flow_threshold=float(flow_thr),
            cellprob_threshold=float(cellprob_thr),
        )
        masks, _flows, _styles = cp.eval(img, **fallback_kwargs)
        return masks


def merge_tile_multiscale(masks_list: list[np.ndarray], iou_merge_thr: float = 0.2, min_area: int = 50) -> np.ndarray:
    tile = np.zeros_like(masks_list[0], dtype=np.int32)
    next_id = 1

    for masks in masks_list:
        masks = remove_small_instances(masks.astype(np.int32), min_area)
        if masks.max() == 0:
            continue

        for local_id in range(1, int(masks.max()) + 1):
            new_mask = masks == local_id
            if new_mask.sum() < min_area:
                continue

            candidate_ids = np.unique(tile[new_mask])
            candidate_ids = candidate_ids[candidate_ids > 0]

            best_iou = 0.0
            best_existing_id = None
            for existing_id in candidate_ids:
                value = iou(new_mask, tile == existing_id)
                if value > best_iou:
                    best_iou = value
                    best_existing_id = int(existing_id)

            if best_existing_id is not None and best_iou >= float(iou_merge_thr):
                tile[new_mask] = best_existing_id
            else:
                tile[new_mask] = next_id
                next_id += 1

    return tile


def assign_global_ids(
    tile_local: np.ndarray,
    old_global: np.ndarray,
    iou_merge_thr: float,
    global_next_id: int,
) -> tuple[np.ndarray, int]:
    tile_global = np.zeros_like(tile_local, dtype=np.int32)

    local_ids = np.unique(tile_local)
    local_ids = local_ids[local_ids > 0]
    if local_ids.size == 0:
        return tile_global, global_next_id

    for local_id in local_ids:
        local_mask = tile_local == local_id
        candidate_old = np.unique(old_global[local_mask])
        candidate_old = candidate_old[candidate_old > 0]

        best_iou = 0.0
        best_old_id = None
        for old_id in candidate_old:
            value = iou(local_mask, old_global == old_id)
            if value > best_iou:
                best_iou = value
                best_old_id = int(old_id)

        if best_old_id is not None and best_iou >= float(iou_merge_thr):
            tile_global[local_mask] = best_old_id
        else:
            tile_global[local_mask] = int(global_next_id)
            global_next_id += 1

    return tile_global, global_next_id


def main(
    in_tif: str,
    msem_tif: str,
    out_dir: str,
    diam_list: tuple[float, ...] = (96.0, 160.0, 256.0),
    iou_merge_thr: float = 0.2,
    tile: int = 1536,
    overlap: int = 384,
    bsize: int = 256,
    tile_overlap: float = 0.35,
    augment: bool = True,
    niter: int = 0,
    use_gray: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("diam_list:", diam_list)
    print("TILE:", tile, "OVERLAP:", overlap)
    print("cellpose bsize:", bsize, "tile_overlap:", tile_overlap, "augment:", augment, "niter:", niter)

    step = tile - overlap
    if step <= 0:
        raise ValueError("tile must be > overlap")

    cp = models.CellposeModel(gpu=(device == "cuda"), pretrained_model="cpsam")

    with rasterio.open(in_tif) as src, rasterio.open(msem_tif) as ms:
        height, width = src.height, src.width
        profile = src.profile.copy()

        inst_full = np.zeros((height, width), dtype=np.int32)
        weight_full = np.zeros((height, width), dtype=np.float32)
        next_id = 1

        for y in range(0, height, step):
            for x in range(0, width, step):
                h = min(tile, height - y)
                w = min(tile, width - x)
                win = Window(x, y, w, h)

                rgb, nodata = read_rgb_window(src, win)
                msem = read_mask_window(ms, win)
                valid = (msem == 1) & (~nodata)

                rgb_in = rgb.copy()
                if valid.any():
                    median_rgb = np.median(rgb_in[valid], axis=0).astype(np.uint8)
                    rgb_in[~valid] = median_rgb
                else:
                    rgb_in[:] = 0

                img_in = rgb_in
                channels = [0, 0] if use_gray else [0, 0]

                pad_h = tile - h
                pad_w = tile - w
                if pad_h > 0 or pad_w > 0:
                    img_in = cv2.copyMakeBorder(img_in, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                masks_list = []
                for diameter in diam_list:
                    masks = cellpose_eval_safe(
                        cp,
                        img_in,
                        diameter=diameter,
                        flow_thr=FLOW_THR,
                        cellprob_thr=CELLPROB_THR,
                        channels=channels,
                        bsize=bsize,
                        tile_overlap=tile_overlap,
                        augment=augment,
                        niter=niter,
                    )
                    masks = masks[:h, :w].astype(np.int32)
                    if valid.any():
                        masks[~valid] = 0
                    masks_list.append(masks)

                tile_local = merge_tile_multiscale(masks_list, iou_merge_thr=iou_merge_thr, min_area=MIN_AREA)
                if tile_local.max() == 0:
                    continue

                old = inst_full[y : y + h, x : x + w]
                tile_global, next_id = assign_global_ids(tile_local, old, iou_merge_thr=iou_merge_thr, global_next_id=next_id)

                w_tile = feather_weight(h, w, overlap=overlap) * valid.astype(np.float32)
                w_old = weight_full[y : y + h, x : x + w]
                take = (w_tile > w_old) & (tile_global > 0)

                old_updated = old.copy()
                old_updated[take] = tile_global[take]
                w_updated = w_old.copy()
                w_updated[take] = w_tile[take]

                inst_full[y : y + h, x : x + w] = old_updated
                weight_full[y : y + h, x : x + w] = w_updated
                print(f"[tile] x={x} y={y} local_max={int(tile_local.max())} global_next_id={next_id}")

        out_tif = os.path.join(out_dir, "Y_inst.tif")
        profile.update(count=1, dtype=rasterio.int32, nodata=0, compress="LZW", photometric="MINISBLACK")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(inst_full.astype(np.int32), 1)
        print("Saved:", out_tif)

        out_shp = os.path.join(out_dir, "Y_inst.shp")
        export_instances_to_shp(inst_full, transform=src.transform, crs=src.crs, out_shp=out_shp, min_area_px=MIN_AREA)

    out_png = os.path.join(out_dir, "Y_inst_color.png")
    max_id = int(inst_full.max())
    rng = np.random.default_rng(0)
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    if max_id > 0:
        lut[1:] = rng.integers(0, 255, size=(max_id, 3), dtype=np.uint8)
    color = lut[inst_full]
    cv2.imwrite(out_png, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print("Saved:", out_png)
    print("instances:", int(inst_full.max()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tif", required=True)
    parser.add_argument("--msem_tif", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--diam_list", type=str, default="96,160,256")
    parser.add_argument("--iou_merge_thr", type=float, default=0.2)
    parser.add_argument("--tile", type=int, default=1536)
    parser.add_argument("--overlap", type=int, default=384)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--tile_overlap", type=float, default=0.35)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_true", default=False)
    parser.add_argument("--niter", type=int, default=0)
    parser.add_argument("--use_rgb", action="store_true", default=False)
    args = parser.parse_args()

    diam_list = tuple(float(x) for x in args.diam_list.split(",") if x.strip())
    augment = not args.no_augment

    main(
        args.in_tif,
        args.msem_tif,
        args.out_dir,
        diam_list=diam_list,
        iou_merge_thr=args.iou_merge_thr,
        tile=args.tile,
        overlap=args.overlap,
        bsize=args.bsize,
        tile_overlap=args.tile_overlap,
        augment=augment,
        niter=args.niter,
        use_gray=(not args.use_rgb),
    )
