from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from input_layer.contracts import DomInputContract, RemoteSensingImageSource
from input_layer.dom.preparer import derive_input_workspace


DEFAULT_DOM_CONTRACT = {
    "recommended_gsd_m": 0.02,
    "acceptable_gsd_range_m": [0.015, 0.05],
    "resample_if_finer_than_m": 0.015,
    "warn_if_coarser_than_m": 0.05,
    "processing_block_px": 5632,
    "processing_block_stride_px": 5120,
    "processing_block_overlap_px": 512,
    "processing_edge_absorb_px": 512,
    "processing_block_min_preferred_px": 5120,
    "processing_block_max_preferred_px": 6144,
    "tile_px": 1024,
    "tile_overlap_px": 256,
    "tile_stride_px": 768,
    "allow_elastic_model_input": False,
    "pad_if_smaller_than_model_input": True,
    "snap_last_tile_to_edge": True,
    "discard_padding_output": True,
    "bsize": 256,
    "processing_mode": "block_then_sliding_window",
    "output_clip_policy": "clip_to_original_bounds",
}


def _input_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return "geotiff"
    if suffix == ".vrt":
        return "vrt"
    return suffix.lstrip(".") or "unknown"


def _transform_to_list(transform: Any) -> list[float] | None:
    if transform is None:
        return None
    try:
        return [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)]
    except Exception:
        return None


def _infer_band_mapping(src: rasterio.io.DatasetReader, source: RemoteSensingImageSource) -> dict[str, int]:
    declared = [band.upper() for band in (source.bands or [])]
    if {"R", "G", "B"}.issubset(set(declared)):
        return {
            "red": declared.index("R") + 1,
            "green": declared.index("G") + 1,
            "blue": declared.index("B") + 1,
        }
    if src.count >= 3:
        return {"red": 1, "green": 2, "blue": 3}
    if src.count == 1:
        return {"red": 1, "green": 1, "blue": 1}
    raise ValueError("至少需要可构造 RGB 的波段。")


def _infer_normalization_policy(dtype: str) -> str:
    normalized = str(dtype).lower()
    if normalized == "uint8":
        return "uint8_passthrough"
    if normalized in {"uint16", "int16"}:
        return "uint16_to_uint8"
    return "percentile_clip_then_scale"


def _gsd_status(gsd_x_m: float | None, gsd_y_m: float | None, acceptable_range: list[float]) -> str:
    if gsd_x_m is None or gsd_y_m is None:
        return "unknown"
    min_allowed, max_allowed = acceptable_range
    gsd = max(abs(gsd_x_m), abs(gsd_y_m))
    if gsd < min_allowed:
        return "too_fine"
    if gsd > max_allowed:
        return "too_coarse"
    return "acceptable"


def _estimate_valid_pixel_ratio(src: rasterio.io.DatasetReader, nodata: int | float | str | None) -> float | None:
    try:
        sample = src.read(masked=True)
        if sample.size == 0:
            return 0.0
        if np.ma.isMaskedArray(sample):
            valid = ~np.ma.getmaskarray(sample)
            if valid.ndim == 3:
                valid = valid.any(axis=0)
            return float(valid.mean())
        if nodata is not None:
            valid = sample != nodata
            if valid.ndim == 3:
                valid = valid.any(axis=0)
            return float(valid.mean())
        return 1.0
    except Exception:
        return None


def _estimate_window_count(size: int, tile_size: int, stride: int) -> int:
    if size <= tile_size:
        return 1
    return int(ceil((size - tile_size) / float(stride))) + 1


def _build_dom_contract(
    *,
    source: RemoteSensingImageSource,
    runtime_cfg: dict[str, Any],
    working_dom_path: Path,
    valid_mask_path: Path,
    width: int,
    height: int,
    bounds: list[float],
    crs: str | None,
    transform: list[float] | None,
    gsd_x_m: float | None,
    gsd_y_m: float | None,
    band_count: int,
    dtype: str,
    band_mapping: dict[str, int],
    normalization_policy: str,
    nodata: int | float | str | None,
    nodata_policy: str,
    global_valid_pixel_ratio_estimate: float | None,
    warnings: list[str],
) -> DomInputContract:
    block_count_x = _estimate_window_count(
        width,
        DEFAULT_DOM_CONTRACT["processing_block_px"],
        DEFAULT_DOM_CONTRACT["processing_block_stride_px"],
    )
    block_count_y = _estimate_window_count(
        height,
        DEFAULT_DOM_CONTRACT["processing_block_px"],
        DEFAULT_DOM_CONTRACT["processing_block_stride_px"],
    )
    tile_count_x = _estimate_window_count(
        width,
        DEFAULT_DOM_CONTRACT["tile_px"],
        DEFAULT_DOM_CONTRACT["tile_stride_px"],
    )
    tile_count_y = _estimate_window_count(
        height,
        DEFAULT_DOM_CONTRACT["tile_px"],
        DEFAULT_DOM_CONTRACT["tile_stride_px"],
    )
    gsd_status = _gsd_status(gsd_x_m, gsd_y_m, DEFAULT_DOM_CONTRACT["acceptable_gsd_range_m"])
    all_warnings = list(warnings)
    gsd = max(abs(gsd_x_m), abs(gsd_y_m)) if gsd_x_m is not None and gsd_y_m is not None else None
    if gsd_status == "too_fine":
        all_warnings.append("gsd_too_fine")
    elif gsd_status == "too_coarse":
        all_warnings.append("gsd_too_coarse")
    if gsd is not None and gsd < DEFAULT_DOM_CONTRACT["resample_if_finer_than_m"]:
        all_warnings.append("resample_if_finer_than_threshold")
    if gsd is not None and gsd > DEFAULT_DOM_CONTRACT["warn_if_coarser_than_m"]:
        all_warnings.append("warn_if_coarser_than_threshold")
    if width * height >= 50_000_000:
        all_warnings.append("large_dom_enable_resume")

    ordered_warnings = list(dict.fromkeys(all_warnings))
    status = "ready" if not ordered_warnings else "warning"
    return DomInputContract(
        dom_id=source.id,
        source_path=source.path,
        working_dom_path=str(working_dom_path),
        input_type=_input_type(source.path),
        mainline_profile=str(runtime_cfg.get("mainline_profile") or "dom_image"),
        width=int(width),
        height=int(height),
        pixel_count=int(width * height),
        bounds=bounds,
        crs=crs,
        transform=transform,
        working_to_original_transform=None,
        gsd_x_m=gsd_x_m,
        gsd_y_m=gsd_y_m,
        recommended_gsd_m=DEFAULT_DOM_CONTRACT["recommended_gsd_m"],
        acceptable_gsd_range_m=list(DEFAULT_DOM_CONTRACT["acceptable_gsd_range_m"]),
        gsd_status=gsd_status,
        band_count=int(band_count),
        dtype=dtype,
        band_mapping=band_mapping,
        normalization_policy=normalization_policy,
        nodata=nodata,
        nodata_policy=nodata_policy,
        valid_mask_path=str(valid_mask_path),
        global_valid_pixel_ratio_estimate=global_valid_pixel_ratio_estimate,
        processing_block_px=DEFAULT_DOM_CONTRACT["processing_block_px"],
        processing_block_stride_px=DEFAULT_DOM_CONTRACT["processing_block_stride_px"],
        processing_block_overlap_px=DEFAULT_DOM_CONTRACT["processing_block_overlap_px"],
        processing_edge_absorb_px=DEFAULT_DOM_CONTRACT["processing_edge_absorb_px"],
        processing_block_min_preferred_px=DEFAULT_DOM_CONTRACT["processing_block_min_preferred_px"],
        processing_block_max_preferred_px=DEFAULT_DOM_CONTRACT["processing_block_max_preferred_px"],
        tile_px=DEFAULT_DOM_CONTRACT["tile_px"],
        tile_overlap_px=DEFAULT_DOM_CONTRACT["tile_overlap_px"],
        tile_stride_px=DEFAULT_DOM_CONTRACT["tile_stride_px"],
        allow_elastic_model_input=DEFAULT_DOM_CONTRACT["allow_elastic_model_input"],
        pad_if_smaller_than_model_input=DEFAULT_DOM_CONTRACT["pad_if_smaller_than_model_input"],
        snap_last_tile_to_edge=DEFAULT_DOM_CONTRACT["snap_last_tile_to_edge"],
        discard_padding_output=DEFAULT_DOM_CONTRACT["discard_padding_output"],
        bsize=DEFAULT_DOM_CONTRACT["bsize"],
        processing_mode=DEFAULT_DOM_CONTRACT["processing_mode"],
        estimated_block_count=int(block_count_x * block_count_y),
        estimated_tile_count=int(tile_count_x * tile_count_y),
        output_clip_policy=DEFAULT_DOM_CONTRACT["output_clip_policy"],
        warnings=ordered_warnings,
        status=status,
    )


def compile_dom_input_contract(
    *,
    source: RemoteSensingImageSource,
    runtime_cfg: dict[str, Any],
    config_path: str | None = None,
) -> DomInputContract:
    workspace = derive_input_workspace(runtime_cfg, config_path=config_path)
    dom_root = Path(workspace["prepared_root"]) / "dom" / source.id
    working_dom_path = dom_root / "working_dom.vrt"
    valid_mask_path = dom_root / "valid_mask.tif"

    try:
        with rasterio.open(source.path) as src:
            if src.width <= 0 or src.height <= 0:
                raise ValueError("DOM 宽高必须大于 0。")

            bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
            xres = abs(float(src.transform.a)) if src.transform is not None else None
            yres = abs(float(src.transform.e)) if src.transform is not None else None
            band_mapping = _infer_band_mapping(src, source)
            dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
            normalization_policy = _infer_normalization_policy(dtype)
            nodata = source.nodata if source.nodata is not None else src.nodata
            valid_pixel_ratio = _estimate_valid_pixel_ratio(src, nodata)

            warnings: list[str] = []
            if src.count < 1 or src.count > 4:
                warnings.append("band_count_unusual")
            if dtype.lower() not in {"uint8", "uint16", "int16"}:
                warnings.append("dtype_requires_special_normalization")
            if src.transform is None:
                warnings.append("missing_transform")
            elif src.transform.b != 0.0 or src.transform.d != 0.0:
                warnings.append("transform_has_rotation")
            if valid_pixel_ratio is not None and valid_pixel_ratio < 0.9:
                warnings.append("large_nodata_or_black_border")
            if src.crs is None:
                warnings.append("missing_crs")
            if xres is None or yres is None:
                warnings.append("missing_gsd")

            return _build_dom_contract(
                source=source,
                runtime_cfg=runtime_cfg,
                working_dom_path=working_dom_path,
                valid_mask_path=valid_mask_path,
                width=int(src.width),
                height=int(src.height),
                bounds=bounds,
                crs=str(src.crs) if src.crs is not None else None,
                transform=_transform_to_list(src.transform),
                gsd_x_m=xres,
                gsd_y_m=yres,
                band_count=int(src.count),
                dtype=dtype,
                band_mapping=band_mapping,
                normalization_policy=normalization_policy,
                nodata=nodata,
                nodata_policy="use_valid_mask",
                global_valid_pixel_ratio_estimate=valid_pixel_ratio,
                warnings=warnings,
            )
    except (RasterioIOError, ValueError):
        placeholder_warnings = ["dom_contract_unreadable_source"]
        return _build_dom_contract(
            source=source,
            runtime_cfg=runtime_cfg,
            working_dom_path=working_dom_path,
            valid_mask_path=valid_mask_path,
            width=0,
            height=0,
            bounds=[0.0, 0.0, 0.0, 0.0],
            crs=source.crs,
            transform=None,
            gsd_x_m=source.resolution_m,
            gsd_y_m=source.resolution_m,
            band_count=max(len(source.bands), 0),
            dtype="unknown",
            band_mapping={"red": 1, "green": 1, "blue": 1},
            normalization_policy="percentile_clip_then_scale",
            nodata=source.nodata,
            nodata_policy="use_valid_mask",
            global_valid_pixel_ratio_estimate=None,
            warnings=placeholder_warnings,
        )
