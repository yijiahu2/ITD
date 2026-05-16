from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from output_layer.contracts import FinalTreeCrownResult


VECTOR_DATASET_EXTS = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]
SCENARIO_COCO_GT = "coco_gt"
SCENARIO_DOM_WITH_GT = "dom_with_gt"
SCENARIO_DOM_WITHOUT_GT = "dom_without_gt"
NO_GT_QUALITY_KEYS = [
    "pred_instance_count",
    "pred_cover_ratio",
    "mean_area_m2",
    "mean_equivalent_crown_width_m",
    "small_fragment_ratio",
    "width_outlier_ratio",
    "duplicate_overlap_ratio",
    "edge_artifact_score",
    "semantic_instance_consistency",
    "semantic_coverage_gap",
    "fragmentation_score",
    "merge_blob_score",
    "online_risk_score",
    "quality_score",
]


def _first_metric(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _format_report_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]], *, align_right: bool = False) -> list[str]:
    align = "---:" if align_right else "---"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([align] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_report_value(value) for value in row) + " |")
    return lines


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_output_layout(root: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    layout = {
        "root": root_path,
        "results": root_path / "results",
        "masks": root_path / "masks",
        "visualization": root_path / "visualization",
        "report": root_path / "report",
    }
    for key, path in layout.items():
        if key != "root":
            ensure_dir(path)
    return layout


def resolve_output_scenario(result: FinalTreeCrownResult) -> str:
    requested = str(result.input_type or "auto").lower()
    if requested in {"coco", "coco_gt", "public_coco", "public_dataset"}:
        return SCENARIO_COCO_GT
    if requested in {"dom_with_gt", "dom_gt", "dom+gt"}:
        return SCENARIO_DOM_WITH_GT
    if requested in {"dom_without_gt", "dom_no_gt", "dom_inference", "dom"} and result.has_gt is False:
        return SCENARIO_DOM_WITHOUT_GT
    if requested in {"dom_without_gt", "dom_no_gt", "dom_inference"}:
        return SCENARIO_DOM_WITHOUT_GT
    if requested in {"dom_with_gt", "dom_gt", "dom+gt"}:
        return SCENARIO_DOM_WITH_GT

    metadata = result.metadata or {}
    source_adapter = str(metadata.get("source_adapter") or "").lower()
    output_type = str(metadata.get("output_type") or "").lower()
    if "coco" in source_adapter or "coco" in output_type or result.coco_predictions_path:
        return SCENARIO_COCO_GT
    if result.has_gt is True or result.gt_metrics:
        return SCENARIO_DOM_WITH_GT
    return SCENARIO_DOM_WITHOUT_GT


def copy_vector_dataset(src: str | Path, dst: str | Path) -> list[str]:
    src_path = Path(src)
    copied: list[str] = []
    for ext in VECTOR_DATASET_EXTS:
        cand = src_path.with_suffix(ext)
        if not cand.exists():
            continue
        out = Path(dst).with_suffix(ext)
        ensure_parent(out)
        shutil.copy2(cand, out)
        copied.append(str(out))
    return copied


def render_vector_preview(src: str | Path, dst: str | Path) -> str | None:
    src_path = Path(src)
    if not src_path.exists():
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-output-layer")
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except Exception:
        return None

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        return None

    dst_path = Path(dst)
    ensure_parent(dst_path)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    gdf.boundary.plot(ax=ax, linewidth=0.3, color="#0b5d1e")
    gdf.plot(ax=ax, linewidth=0, color="#7ccf7a", alpha=0.75)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0)
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(dst_path)


def render_segmentation_visualization(
    crowns_src: str | Path,
    points_src: str | Path | None,
    dst: str | Path,
    background_raster: str | Path | None = None,
) -> str | None:
    crowns_path = Path(crowns_src)
    if not crowns_path.exists():
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-output-layer")
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except Exception:
        return None

    crowns_gdf = gpd.read_file(crowns_path)
    if crowns_gdf.empty:
        return None

    points_gdf = None
    if points_src and Path(points_src).exists():
        try:
            points_gdf = gpd.read_file(points_src)
        except Exception:
            points_gdf = None

    raster_crs = None
    raster_bounds = None
    if background_raster and Path(background_raster).exists():
        try:
            import rasterio

            with rasterio.open(background_raster) as src:
                raster_crs = src.crs
                raster_bounds = src.bounds
        except Exception:
            raster_crs = None

    if raster_crs is not None and crowns_gdf.crs is not None and crowns_gdf.crs != raster_crs:
        try:
            crowns_gdf = crowns_gdf.to_crs(raster_crs)
        except Exception:
            raster_crs = None

    if (
        points_gdf is not None
        and not points_gdf.empty
        and raster_crs is not None
        and points_gdf.crs is not None
        and points_gdf.crs != raster_crs
    ):
        try:
            points_gdf = points_gdf.to_crs(raster_crs)
        except Exception:
            points_gdf = None

    dst_path = Path(dst)
    ensure_parent(dst_path)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)

    if background_raster and Path(background_raster).exists():
        try:
            import rasterio
            from rasterio.plot import show

            with rasterio.open(background_raster) as src:
                if src.count >= 3:
                    show((src, [1, 2, 3]), ax=ax)
                else:
                    show((src, 1), ax=ax, cmap="gray")
        except Exception:
            pass

    crowns_gdf.boundary.plot(ax=ax, linewidth=0.35, color="#0b5d1e")
    crowns_gdf.plot(ax=ax, linewidth=0, color="#7ccf7a", alpha=0.35)
    if points_gdf is not None and not points_gdf.empty:
        points_gdf.plot(ax=ax, markersize=6, color="#c92a2a", alpha=0.9)
    if raster_bounds is not None:
        minx, miny, maxx, maxy = raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    else:
        minx, miny, maxx, maxy = crowns_gdf.total_bounds
        if maxx > minx and maxy > miny:
            pad_x = max((maxx - minx) * 0.03, 1.0)
            pad_y = max((maxy - miny) * 0.03, 1.0)
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0)
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(dst_path)


def build_semantic_union(src: str | Path, dst: str | Path) -> list[str]:
    src_path = Path(src)
    if not src_path.exists():
        return []
    try:
        import geopandas as gpd
    except Exception:
        return []

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        return []
    union_geom = gdf.geometry.union_all()
    union_gdf = gpd.GeoDataFrame({"class": [1]}, geometry=[union_geom], crs=gdf.crs)
    union_gdf = union_gdf.explode(index_parts=False).reset_index(drop=True)
    union_gdf = union_gdf[~union_gdf.geometry.is_empty].copy()
    if union_gdf.empty:
        return []
    out_path = Path(dst)
    ensure_parent(out_path)
    union_gdf.to_file(out_path)
    return [str(p) for p in out_path.parent.glob(f"{out_path.stem}.*")]


def build_tree_points(src: str | Path, dst: str | Path) -> list[str]:
    src_path = Path(src)
    if not src_path.exists():
        return []
    try:
        import geopandas as gpd
    except Exception:
        return []

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        return []

    work_gdf = gdf.copy()
    centroid_series = work_gdf.geometry.centroid
    points_gdf = gpd.GeoDataFrame(work_gdf.drop(columns="geometry"), geometry=centroid_series, crs=work_gdf.crs)
    out_path = Path(dst)
    ensure_parent(out_path)
    points_gdf.to_file(out_path)
    return [str(p) for p in out_path.parent.glob(f"{out_path.stem}.*")]


def normalize_tree_crown_fields(src: str | Path, *, scenario: str, gt_matches: list[dict[str, Any]] | None = None) -> str | None:
    src_path = Path(src)
    if not src_path.exists():
        return None
    try:
        import geopandas as gpd
        import numpy as np
    except Exception:
        return None

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        return str(src_path)
    work_gdf = gdf.copy()
    matches_by_id: dict[str, dict[str, Any]] = {}
    for match in gt_matches or []:
        pred_id = match.get("tree_id") or match.get("pred_id") or match.get("id")
        if pred_id is not None:
            matches_by_id[str(pred_id)] = dict(match)

    centroid = work_gdf.geometry.centroid
    area = work_gdf.geometry.area
    perimeter = work_gdf.geometry.length
    eq_width = (4.0 * area / np.pi) ** 0.5
    source_id_col = next((name for name in ["tree_id", "id", "source_id", "pred_id"] if name in work_gdf.columns), None)

    if "tree_id" not in work_gdf.columns:
        work_gdf["tree_id"] = [int(idx) + 1 for idx in range(len(work_gdf))]
    if "score" not in work_gdf.columns:
        work_gdf["score"] = 1.0
    if "area_m2" not in work_gdf.columns:
        work_gdf["area_m2"] = area.astype(float)
    if "perim_m" not in work_gdf.columns:
        work_gdf["perim_m"] = perimeter.astype(float)
    if "eq_width_m" not in work_gdf.columns:
        work_gdf["eq_width_m"] = eq_width.astype(float)
    if "center_x" not in work_gdf.columns:
        work_gdf["center_x"] = centroid.x.astype(float)
    if "center_y" not in work_gdf.columns:
        work_gdf["center_y"] = centroid.y.astype(float)
    if "src_tile" not in work_gdf.columns:
        work_gdf["src_tile"] = ""

    if scenario == SCENARIO_DOM_WITH_GT:
        gt_ids: list[Any] = []
        ious: list[Any] = []
        eval_types: list[str] = []
        for _, row in work_gdf.iterrows():
            raw_id = row.get(source_id_col) if source_id_col else row.get("tree_id")
            match = matches_by_id.get(str(raw_id), {})
            gt_ids.append(match.get("gt_id", row.get("gt_id") if "gt_id" in work_gdf.columns else None))
            iou = match.get("iou_gt", match.get("iou", row.get("iou_gt") if "iou_gt" in work_gdf.columns else None))
            ious.append(iou)
            existing_eval = row.get("eval_type") if "eval_type" in work_gdf.columns else None
            if existing_eval:
                eval_types.append(str(existing_eval))
            elif iou is None:
                eval_types.append("FP")
            elif float(iou) >= 0.5:
                eval_types.append("TP")
            else:
                eval_types.append("LOW_IOU")
        work_gdf["gt_id"] = gt_ids
        work_gdf["iou_gt"] = ious
        work_gdf["eval_type"] = eval_types
    elif scenario == SCENARIO_DOM_WITHOUT_GT:
        if "quality" not in work_gdf.columns:
            work_gdf["quality"] = "unknown"
        if "risk_type" not in work_gdf.columns:
            work_gdf["risk_type"] = "normal"

    work_gdf.to_file(src_path)
    return str(src_path)


def build_height_structure_outputs(
    *,
    crowns_src: str | Path,
    chm_raster: str | Path | None,
    annotated_vector_dst: str | Path,
    summary_dst: str | Path,
) -> dict[str, Any]:
    if not chm_raster or not Path(chm_raster).exists() or not Path(crowns_src).exists():
        return {"available": False, "reason": "missing_chm_or_crowns", "annotated_vector": None, "summary_json": None}
    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        from rasterio.mask import mask
    except Exception as exc:
        return {"available": False, "reason": f"dependency_unavailable:{exc}", "annotated_vector": None, "summary_json": None}

    crowns_gdf = gpd.read_file(crowns_src)
    if crowns_gdf.empty:
        return {"available": False, "reason": "empty_crowns", "annotated_vector": None, "summary_json": None}

    rows: list[dict[str, Any]] = []
    with rasterio.open(chm_raster) as src:
        work_gdf = crowns_gdf
        if work_gdf.crs is not None and src.crs is not None and work_gdf.crs != src.crs:
            work_gdf = work_gdf.to_crs(src.crs)
        for idx, geom in enumerate(work_gdf.geometry):
            attrs = {
                "tree_height_p95": None,
                "tree_height_max": None,
                "crown_height_mean": None,
                "crown_height_std": None,
                "height_gradient": None,
                "structure_tag": "unknown",
            }
            if geom is None or geom.is_empty:
                rows.append(attrs)
                continue
            try:
                data, _ = mask(src, [geom.__geo_interface__], crop=True, filled=True)
                arr = data[0].astype("float32")
                valid = np.isfinite(arr)
                if src.nodata is not None and np.isfinite(float(src.nodata)):
                    valid &= ~np.isclose(arr, float(src.nodata))
                vals = arr[valid]
                vals = vals[vals > 0]
                if vals.size:
                    gy, gx = np.gradient(np.where(valid, arr, float(np.nanmean(vals))))
                    gradient = np.sqrt(gx * gx + gy * gy)
                    p95 = float(np.percentile(vals, 95))
                    std = float(np.std(vals))
                    attrs = {
                        "tree_height_p95": p95,
                        "tree_height_max": float(np.max(vals)),
                        "crown_height_mean": float(np.mean(vals)),
                        "crown_height_std": std,
                        "height_gradient": float(np.nanmean(gradient)),
                        "structure_tag": "tall_complex" if p95 >= 20.0 and std >= 4.0 else "tall_simple" if p95 >= 20.0 else "low_complex" if std >= 4.0 else "low_simple",
                    }
            except Exception:
                pass
            rows.append(attrs)

    annotated_gdf = crowns_gdf.copy()
    for key in ["tree_height_p95", "tree_height_max", "crown_height_mean", "crown_height_std", "height_gradient", "structure_tag"]:
        annotated_gdf[key] = [row.get(key) for row in rows]
    annotated_path = Path(annotated_vector_dst)
    ensure_parent(annotated_path)
    annotated_gdf.to_file(annotated_path, driver="GPKG")

    valid_heights = [row["tree_height_p95"] for row in rows if row.get("tree_height_p95") is not None]
    summary = {
        "available": True,
        "chm_raster": str(chm_raster),
        "annotated_vector": str(annotated_path),
        "instance_count": len(rows),
        "height_attributed_count": len(valid_heights),
        "tree_height_p95_mean": float(np.mean(valid_heights)) if valid_heights else None,
        "tree_height_p95_max": float(np.max(valid_heights)) if valid_heights else None,
        "structure_tag_counts": {tag: sum(1 for row in rows if row.get("structure_tag") == tag) for tag in sorted({str(row.get("structure_tag")) for row in rows})},
    }
    summary_path = Path(summary_dst)
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _copy_optional_file(src: str | Path | None, dst: str | Path | None) -> str | None:
    if not src or not dst:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_path = Path(dst)
    try:
        if src_path.resolve() == dst_path.resolve():
            return str(dst_path)
    except Exception:
        pass
    ensure_parent(dst_path)
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def _raster_context(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        import rasterio

        with rasterio.open(path) as src:
            return {
                "width": int(src.width),
                "height": int(src.height),
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
            }
    except Exception:
        return {}


def _pixel_box_to_geometry(bbox: list[float], transform: Any | None) -> Any:
    from shapely.geometry import box

    x, y, width, height = [float(value) for value in bbox[:4]]
    if transform is None:
        return box(x, y, x + width, y + height)
    corners = [
        transform * (x, y),
        transform * (x + width, y),
        transform * (x + width, y + height),
        transform * (x, y + height),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return box(min(xs), min(ys), max(xs), max(ys))


def _pixel_polygon_to_geometry(segmentation: Any, transform: Any | None) -> Any | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except Exception:
        return None
    polygons = []
    for raw_poly in segmentation:
        if not isinstance(raw_poly, list) or len(raw_poly) < 6:
            continue
        coords = []
        for idx in range(0, len(raw_poly) - 1, 2):
            x = float(raw_poly[idx])
            y = float(raw_poly[idx + 1])
            coords.append(transform * (x, y) if transform is not None else (x, y))
        poly = Polygon(coords)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    if not polygons:
        return None
    return unary_union(polygons)


def _write_instances_as_tree_crowns(
    *,
    instances: list[dict[str, Any]],
    dst: str | Path,
    input_dom_path: str | Path | None,
    image_width: int | None,
    image_height: int | None,
    scenario: str,
    gt_matches: list[dict[str, Any]] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    if not instances:
        return None, {"geometry_source": "none", "coordinate_mode": "unknown"}
    try:
        import geopandas as gpd
    except Exception:
        return None, {"geometry_source": "dependency_unavailable", "coordinate_mode": "unknown"}

    raster_ctx = _raster_context(input_dom_path)
    transform = raster_ctx.get("transform")
    crs = raster_ctx.get("crs")
    coordinate_mode = "geospatial" if transform is not None and crs is not None else "pixel"
    geometry_source = "segmentation" if any(item.get("segmentation") for item in instances) else "bbox_fallback"
    rows: list[dict[str, Any]] = []
    geometries = []
    matches_by_id = {
        str(match.get("tree_id") or match.get("pred_id") or match.get("id")): match
        for match in gt_matches or []
        if match.get("tree_id") is not None or match.get("pred_id") is not None or match.get("id") is not None
    }
    for idx, instance in enumerate(instances, start=1):
        bbox = list(instance.get("bbox") or instance.get("bbox_xywh") or [0, 0, 0, 0])
        geom = _pixel_polygon_to_geometry(instance.get("segmentation"), transform)
        if geom is None:
            geom = _pixel_box_to_geometry(bbox, transform)
        if geom is None or geom.is_empty:
            continue
        raw_id = instance.get("id") or instance.get("pred_id") or idx
        center = geom.centroid
        row = {
            "tree_id": idx,
            "source_id": str(raw_id)[:80],
            "image_id": str(instance.get("image_id") or ""),
            "score": float(instance.get("score", 1.0)),
            "category": int(instance.get("category_id") or 1),
            "area_m2": float(geom.area),
            "perim_m": float(geom.length),
            "eq_width_m": float((4.0 * geom.area / 3.141592653589793) ** 0.5) if geom.area > 0 else 0.0,
            "center_x": float(center.x),
            "center_y": float(center.y),
            "src_tile": str(instance.get("src_tile") or instance.get("tile_id") or instance.get("image_id") or "")[:80],
        }
        if scenario == SCENARIO_DOM_WITH_GT:
            match = matches_by_id.get(str(raw_id), {})
            iou = instance.get("iou_gt", match.get("iou_gt", match.get("iou")))
            row["gt_id"] = instance.get("gt_id", match.get("gt_id"))
            row["iou_gt"] = iou
            row["eval_type"] = instance.get("eval_type") or ("TP" if iou is not None and float(iou) >= 0.5 else "LOW_IOU" if iou is not None else "FP")
        elif scenario == SCENARIO_DOM_WITHOUT_GT:
            row["quality"] = str(instance.get("quality") or "unknown")[:32]
            row["risk_type"] = str(instance.get("risk_type") or "normal")[:64]
        rows.append(row)
        geometries.append(geom)
    if not geometries:
        return None, {"geometry_source": "empty", "coordinate_mode": coordinate_mode}
    gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)
    out_path = Path(dst)
    ensure_parent(out_path)
    gdf.to_file(out_path)
    normalize_tree_crown_fields(out_path, scenario=scenario, gt_matches=gt_matches)
    return str(out_path), {
        "geometry_source": geometry_source,
        "coordinate_mode": coordinate_mode,
        "image_width": raster_ctx.get("width") or image_width,
        "image_height": raster_ctx.get("height") or image_height,
        "crs": str(crs) if crs is not None else None,
    }


def _materialize_tree_crowns_vector(
    src: str | Path | None,
    dst: str | Path,
    *,
    scenario: str,
    gt_matches: list[dict[str, Any]] | None = None,
) -> str | None:
    if not src or not Path(src).exists():
        return None
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.suffix.lower() == ".shp":
        copied = copy_vector_dataset(src_path, dst_path)
        if copied and dst_path.exists():
            normalize_tree_crown_fields(dst_path, scenario=scenario, gt_matches=gt_matches)
            return str(dst_path)
        return None
    try:
        import geopandas as gpd

        gdf = gpd.read_file(src_path)
        if gdf.empty:
            return None
        ensure_parent(dst_path)
        gdf.to_file(dst_path)
        normalize_tree_crown_fields(dst_path, scenario=scenario, gt_matches=gt_matches)
        return str(dst_path) if dst_path.exists() else None
    except Exception:
        return None


def _write_fallback_semantic_mask(
    *,
    crowns_src: str | Path | None,
    input_dom_path: str | Path | None,
    dst_tif: str | Path,
    dst_png: str | Path,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, str | None]:
    if not crowns_src or not Path(crowns_src).exists():
        return {"semantic_mask_tif": None, "semantic_mask_png": None, "semantic_mask_source": "missing_tree_crowns"}
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except Exception as exc:
        return {"semantic_mask_tif": None, "semantic_mask_png": None, "semantic_mask_source": f"dependency_unavailable:{exc}"}

    raster_ctx = _raster_context(input_dom_path)
    width = int(raster_ctx.get("width") or image_width or 0)
    height = int(raster_ctx.get("height") or image_height or 0)
    if width <= 0 or height <= 0:
        return {"semantic_mask_tif": None, "semantic_mask_png": None, "semantic_mask_source": "missing_raster_shape"}
    transform = raster_ctx.get("transform") or from_origin(0, height, 1, 1)
    crs = raster_ctx.get("crs")
    gdf = gpd.read_file(crowns_src)
    if gdf.empty:
        return {"semantic_mask_tif": None, "semantic_mask_png": None, "semantic_mask_source": "empty_tree_crowns"}
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            pass
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    tif_path = Path(dst_tif)
    ensure_parent(tif_path)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
        "crs": crs,
        "nodata": 0,
        "compress": "LZW",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(mask, 1)
    png_path = Path(dst_png)
    ensure_parent(png_path)
    plt.imsave(png_path, mask, cmap="gray", vmin=0, vmax=1)
    return {
        "semantic_mask_tif": str(tif_path),
        "semantic_mask_png": str(png_path),
        "semantic_mask_source": "rasterized_tree_crowns_fallback",
    }


def _write_instance_mask_from_crowns(
    *,
    crowns_src: str | Path | None,
    input_dom_path: str | Path | None,
    dst_tif: str | Path,
    dst_png: str | Path | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, str | None]:
    if not crowns_src or not Path(crowns_src).exists():
        return {"instance_mask_tif": None, "instance_mask_png": None, "instance_mask_source": "missing_tree_crowns"}
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except Exception as exc:
        return {"instance_mask_tif": None, "instance_mask_png": None, "instance_mask_source": f"dependency_unavailable:{exc}"}

    raster_ctx = _raster_context(input_dom_path)
    width = int(raster_ctx.get("width") or image_width or 0)
    height = int(raster_ctx.get("height") or image_height or 0)
    if width <= 0 or height <= 0:
        return {"instance_mask_tif": None, "instance_mask_png": None, "instance_mask_source": "missing_raster_shape"}
    transform = raster_ctx.get("transform") or from_origin(0, height, 1, 1)
    crs = raster_ctx.get("crs")
    gdf = gpd.read_file(crowns_src)
    if gdf.empty:
        return {"instance_mask_tif": None, "instance_mask_png": None, "instance_mask_source": "empty_tree_crowns"}
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        try:
            gdf = gdf.to_crs(crs)
        except Exception:
            pass
    shapes = []
    for idx, geom in enumerate(gdf.geometry, start=1):
        if geom is not None and not geom.is_empty:
            shapes.append((geom, idx))
    if not shapes:
        return {"instance_mask_tif": None, "instance_mask_png": None, "instance_mask_source": "empty_geometry"}
    dtype = "uint16" if len(shapes) <= 65535 else "uint32"
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=dtype)
    tif_path = Path(dst_tif)
    ensure_parent(tif_path)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "transform": transform,
        "crs": crs,
        "nodata": 0,
        "compress": "LZW",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(mask, 1)
    png_path = None
    if dst_png:
        png_path = Path(dst_png)
        ensure_parent(png_path)
        preview = mask.astype("float32")
        if preview.max() > 0:
            preview = preview / preview.max()
        plt.imsave(png_path, preview, cmap="viridis", vmin=0, vmax=1)
    return {
        "instance_mask_tif": str(tif_path),
        "instance_mask_png": str(png_path) if png_path else None,
        "instance_mask_source": "rasterized_tree_crowns",
    }


def _write_coco_predictions(result: FinalTreeCrownResult, dst: str | Path) -> str | None:
    source_predictions: list[dict[str, Any]] = []
    if result.coco_predictions_path and Path(result.coco_predictions_path).exists():
        try:
            payload = json.loads(Path(result.coco_predictions_path).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("annotations") or payload.get("predictions") or []
            source_predictions = [dict(item) for item in payload or [] if isinstance(item, dict)]
        except Exception:
            source_predictions = []
    elif result.instances:
        source_predictions = [dict(item) for item in result.instances]
    if not source_predictions:
        return None
    predictions = []
    for instance in source_predictions:
        row = {
            "image_id": instance.get("image_id"),
            "category_id": instance.get("category_id", 1),
            "bbox": list(instance.get("bbox") or instance.get("bbox_xywh") or []),
            "score": float(instance.get("score", 1.0)),
        }
        if instance.get("segmentation") is not None:
            row["segmentation"] = instance.get("segmentation")
        predictions.append(row)
    dst_path = Path(dst)
    ensure_parent(dst_path)
    dst_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(dst_path)


def _write_coco_instance_masks(
    *,
    result: FinalTreeCrownResult,
    masks_dir: str | Path,
    max_masks: int | None = None,
) -> list[str]:
    if result.instance_mask_paths:
        copied = []
        image_lookup = _resolve_coco_image_lookup(result)
        image_ids = list(image_lookup.keys())
        for idx, src in enumerate(result.instance_mask_paths, start=1):
            image_id = image_ids[idx - 1] if idx - 1 < len(image_ids) else Path(src).stem.replace("_instance_mask", "")
            safe_id = _safe_output_name(image_id)
            copied_path = _copy_optional_file(src, Path(masks_dir) / f"image_{safe_id}_instance_mask.png")
            if copied_path:
                copied.append(copied_path)
        return copied
    if not result.instances:
        return []
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.path import Path as MplPath
    except Exception:
        return []

    width = int(result.image_width or 0)
    height = int(result.image_height or 0)
    if width <= 0 or height <= 0:
        raster_ctx = _raster_context(result.input_dom_path)
        width = int(raster_ctx.get("width") or 0)
        height = int(raster_ctx.get("height") or 0)
    if width <= 0 or height <= 0:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for instance in result.instances:
        image_id = str(instance["image_id"]) if instance.get("image_id") is not None else "image"
        grouped.setdefault(image_id, []).append(instance)

    out_dir = Path(masks_dir)
    ensure_dir(out_dir)
    written: list[str] = []
    for image_idx, (image_id, instances) in enumerate(sorted(grouped.items()), start=1):
        if max_masks is not None and image_idx > max_masks:
            break
        mask = np.zeros((height, width), dtype=np.uint16)
        yy, xx = np.mgrid[:height, :width]
        points = np.vstack((xx.ravel(), yy.ravel())).T
        for inst_idx, instance in enumerate(instances, start=1):
            segmentation = instance.get("segmentation")
            painted = False
            if isinstance(segmentation, list):
                for raw_poly in segmentation:
                    if not isinstance(raw_poly, list) or len(raw_poly) < 6:
                        continue
                    coords = [(float(raw_poly[i]), float(raw_poly[i + 1])) for i in range(0, len(raw_poly) - 1, 2)]
                    inside = MplPath(coords).contains_points(points).reshape((height, width))
                    mask[inside] = inst_idx
                    painted = True
            if not painted:
                bbox = list(instance.get("bbox") or instance.get("bbox_xywh") or [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, w, h = [int(round(float(value))) for value in bbox[:4]]
                    x0 = max(0, min(width, x))
                    y0 = max(0, min(height, y))
                    x1 = max(0, min(width, x + max(w, 0)))
                    y1 = max(0, min(height, y + max(h, 0)))
                    if x1 > x0 and y1 > y0:
                        mask[y0:y1, x0:x1] = inst_idx
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in image_id)
        out_path = out_dir / f"image_{safe_id}_instance_mask.png"
        preview = mask.astype("float32")
        if preview.max() > 0:
            preview = preview / preview.max()
        plt.imsave(out_path, preview, cmap="viridis", vmin=0, vmax=1)
        written.append(str(out_path))
    return written


def _safe_output_name(text: Any) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text)]
    return "".join(chars).strip("_") or "item"


def _resolve_coco_image_lookup(result: FinalTreeCrownResult) -> dict[str, dict[str, Any]]:
    metadata = result.metadata or {}
    image_root = metadata.get("image_root") or metadata.get("dataset_image_root")
    images = metadata.get("images") or metadata.get("coco_images") or []
    lookup: dict[str, dict[str, Any]] = {}
    if isinstance(images, list):
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id") or image.get("image_id")
            file_name = image.get("file_name") or image.get("image_name") or image.get("path")
            path = image.get("path")
            if not path and file_name and image_root:
                path = str(Path(str(image_root)) / str(file_name))
            if image_id is not None:
                lookup[str(image_id)] = {**image, "path": path}
    if result.input_dom_path and result.instances:
        first_image_id = result.instances[0].get("image_id")
        if first_image_id is not None and str(first_image_id) not in lookup:
            lookup[str(first_image_id)] = {"id": first_image_id, "path": result.input_dom_path}
    return lookup


def _draw_coco_overlay(
    *,
    image_path: str | Path | None,
    instances: list[dict[str, Any]],
    dst: str | Path,
    title: str | None = None,
    color_by_error: bool = False,
    show_labels: bool = False,
) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-output-layer")
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from PIL import Image
    except Exception:
        return None

    dst_path = Path(dst)
    ensure_parent(dst_path)
    canvas = None
    width = 0
    height = 0
    if image_path and Path(image_path).exists():
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            canvas = image
        except Exception:
            canvas = None
    if canvas is None:
        width = int(max([float((inst.get("bbox") or [0, 0, 1, 1])[0]) + float((inst.get("bbox") or [0, 0, 1, 1])[2]) for inst in instances] or [256]))
        height = int(max([float((inst.get("bbox") or [0, 0, 1, 1])[1]) + float((inst.get("bbox") or [0, 0, 1, 1])[3]) for inst in instances] or [256]))
        width = max(width, 64)
        height = max(height, 64)
        canvas = Image.new("RGB", (width, height), color=(245, 245, 245))

    fig_w = max(min(width / 120.0, 10.0), 3.0)
    fig_h = max(min(height / 120.0, 10.0), 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.imshow(canvas)
    combined_overlay = np.zeros((height, width, 4), dtype=float)
    label_points: list[tuple[float, float, str, tuple[float, float, float]]] = []
    for idx, instance in enumerate(instances, start=1):
        color = "#2f9e44"
        if color_by_error:
            eval_type = str(instance.get("eval_type") or instance.get("error_type") or "").lower()
            if eval_type in {"fp", "false_positive"}:
                color = "#e03131"
            elif eval_type in {"fn", "false_negative"}:
                color = "#1971c2"
            elif eval_type in {"low_iou", "low-iou"}:
                color = "#f08c00"
        segmentation = instance.get("segmentation")
        painted = False
        rgb = {
            "#2f9e44": (47 / 255, 158 / 255, 68 / 255),
            "#e03131": (224 / 255, 49 / 255, 49 / 255),
            "#1971c2": (25 / 255, 113 / 255, 194 / 255),
            "#f08c00": (240 / 255, 140 / 255, 0),
        }.get(color, (47 / 255, 158 / 255, 68 / 255))
        if isinstance(segmentation, dict):
            try:
                from pycocotools import mask as mask_utils

                rle = dict(segmentation)
                counts = rle.get("counts")
                if isinstance(counts, str):
                    rle["counts"] = counts.encode("ascii")
                mask = mask_utils.decode(rle).astype(bool)
                if mask.ndim == 3:
                    mask = mask[:, :, 0].astype(bool)
                if mask.shape[0] != height or mask.shape[1] != width:
                    mask = mask[:height, :width]
                boundary = np.zeros_like(mask, dtype=bool)
                boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
                boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
                boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
                boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
                combined_overlay[mask, :3] = rgb
                combined_overlay[mask, 3] = np.maximum(combined_overlay[mask, 3], 0.22)
                combined_overlay[boundary, :3] = rgb
                combined_overlay[boundary, 3] = 0.95
                painted = True
            except Exception:
                painted = False
        if isinstance(segmentation, list):
            for raw_poly in segmentation:
                if not isinstance(raw_poly, list) or len(raw_poly) < 6:
                    continue
                coords = [(float(raw_poly[i]), float(raw_poly[i + 1])) for i in range(0, len(raw_poly) - 1, 2)]
                ax.add_patch(MplPolygon(coords, closed=True, fill=True, alpha=0.22, edgecolor=color, facecolor=color, linewidth=1.0))
                painted = True
        if show_labels and idx <= 30:
            bbox = list(instance.get("bbox") or instance.get("bbox_xywh") or [])
            if len(bbox) >= 2:
                label_points.append((float(bbox[0]), float(bbox[1]), str(idx), rgb))
    if combined_overlay[:, :, 3].max() > 0:
        ax.imshow(combined_overlay)
    for x, y, label, rgb in label_points:
        ax.text(
            x,
            y,
            label,
            color="white",
            fontsize=6,
            bbox={"facecolor": rgb, "alpha": 0.7, "pad": 1},
        )
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(dst_path)


def _overlay_worker_count(task_count: int) -> int:
    if task_count <= 1:
        return 1
    raw = os.environ.get("ITD_OUTPUT_LAYER_OVERLAY_WORKERS")
    if raw:
        try:
            configured = int(raw)
            if configured > 0:
                return min(configured, task_count)
        except ValueError:
            pass
    return min(os.cpu_count() or 1, task_count, 4)


def _render_coco_overlay_worker(task: dict[str, Any]) -> dict[str, Any]:
    rendered = _draw_coco_overlay(
        image_path=task.get("image_path"),
        instances=list(task.get("instances") or []),
        dst=task["dst"],
        title=task.get("title"),
        color_by_error=bool(task.get("color_by_error", False)),
        show_labels=bool(task.get("show_labels", False)),
    )
    return {
        "index": int(task.get("index", 0)),
        "key": task.get("key"),
        "path": rendered,
    }


def _render_coco_overlays(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    workers = _overlay_worker_count(len(tasks))
    if workers <= 1:
        return [_render_coco_overlay_worker(task) for task in tasks]
    try:
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_render_coco_overlay_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda item: int(item.get("index", 0)))
    except Exception:
        return [_render_coco_overlay_worker(task) for task in tasks]


def _write_coco_sample_overlays(result: FinalTreeCrownResult, layout: dict[str, Path]) -> list[str]:
    max_overlays = int(result.visualization_config.get("max_sample_overlays", 20) or 0)
    if max_overlays <= 0:
        return []
    image_lookup = _resolve_coco_image_lookup(result)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for instance in result.instances:
        image_id = str(instance["image_id"]) if instance.get("image_id") is not None else "image"
        grouped.setdefault(image_id, []).append(instance)
    tasks: list[dict[str, Any]] = []
    overlay_dir = layout["visualization"] / "sample_overlays"
    if image_lookup:
        image_ids = list(image_lookup.keys())
    else:
        image_ids = sorted(grouped.keys())
    for image_idx, image_id in enumerate(image_ids, start=1):
        if image_idx > max_overlays:
            break
        instances = grouped.get(str(image_id), [])
        image_info = image_lookup.get(str(image_id), {})
        out_path = overlay_dir / f"sample_{image_idx:06d}_overlay.png"
        tasks.append(
            {
                "index": image_idx,
                "image_path": image_info.get("path"),
                "instances": instances,
                "dst": str(out_path),
                "title": f"image_id={image_id}",
                "color_by_error": False,
                "show_labels": False,
            }
        )
    return [str(item["path"]) for item in _render_coco_overlays(tasks) if item.get("path")]


def _write_placeholder_png(dst: str | Path, title: str) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-output-layer")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    dst_path = Path(dst)
    ensure_parent(dst_path)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(dst_path)


def _write_coco_selected_error_examples(result: FinalTreeCrownResult, layout: dict[str, Path]) -> dict[str, str | None]:
    save_errors = bool(result.visualization_config.get("save_error_examples", True))
    if not save_errors:
        return {}
    max_examples = int(result.visualization_config.get("max_error_examples", 20) or 20)
    requested_types = result.visualization_config.get("error_example_types") or ["fp", "fn", "low_iou"]
    requested = {str(item).strip().lower() for item in requested_types if str(item).strip()}
    image_lookup = _resolve_coco_image_lookup(result)
    examples: dict[str, list[dict[str, Any]]] = {"fp": [], "fn": [], "low_iou": []}
    metrics = result.gt_metrics or {}
    metric_examples = {
        "fp": metrics.get("false_positive_examples") or [],
        "fn": metrics.get("false_negative_examples") or [],
        "low_iou": metrics.get("low_iou_examples") or [],
    }
    for key, values in metric_examples.items():
        if isinstance(values, list):
            examples[key].extend(dict(item) for item in values if isinstance(item, dict))
    for instance in result.instances:
        eval_type = str(instance.get("eval_type") or instance.get("error_type") or "").lower()
        iou = instance.get("iou_gt", instance.get("best_iou", instance.get("iou")))
        if eval_type in {"fp", "false_positive"} or instance.get("is_fp") is True:
            examples["fp"].append(instance)
        elif eval_type in {"fn", "false_negative"} or instance.get("is_fn") is True:
            examples["fn"].append(instance)
        elif eval_type in {"low_iou", "low-iou"} or (iou is not None and float(iou) < 0.5):
            examples["low_iou"].append(instance)

    out_dir = layout["visualization"] / "selected_error_examples"
    outputs: dict[str, str | None] = {}
    file_by_type = {
        "fp": "false_positive_examples.png",
        "fn": "false_negative_examples.png",
        "low_iou": "low_iou_examples.png",
    }
    label_by_type = {
        "fp": "No false-positive examples selected",
        "fn": "No false-negative examples selected",
        "low_iou": "No low-IoU examples selected",
    }
    tasks: list[dict[str, Any]] = []
    task_keys: set[str] = set()
    for index, (error_type, filename) in enumerate(file_by_type.items(), start=1):
        if error_type not in requested:
            continue
        selected = sorted(examples[error_type], key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_examples]
        out_path = out_dir / filename
        if selected:
            image_id = str(selected[0]["image_id"]) if selected[0].get("image_id") is not None else "image"
            image_info = image_lookup.get(image_id, {})
            task_keys.add(error_type)
            tasks.append(
                {
                    "index": index,
                    "key": error_type,
                    "image_path": image_info.get("path"),
                    "instances": selected,
                    "dst": str(out_path),
                    "title": f"{error_type} top-{len(selected)}",
                    "color_by_error": True,
                    "show_labels": True,
                }
            )
        else:
            outputs[error_type] = _write_placeholder_png(out_path, label_by_type[error_type])
    for item in _render_coco_overlays(tasks):
        key = str(item.get("key") or "")
        if key in task_keys:
            outputs[key] = item.get("path")
    return outputs


def _render_vector_thematic_map(
    *,
    crowns_src: str | Path | None,
    dst: str | Path,
    column: str | None = None,
    background_raster: str | Path | None = None,
) -> str | None:
    if not crowns_src or not Path(crowns_src).exists():
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-output-layer")
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rasterio
        from rasterio.plot import show
    except Exception:
        return None

    gdf = gpd.read_file(crowns_src)
    if gdf.empty:
        return None
    dst_path = Path(dst)
    ensure_parent(dst_path)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    if background_raster and Path(background_raster).exists():
        try:
            with rasterio.open(background_raster) as src:
                if src.count >= 3:
                    show((src, [1, 2, 3]), ax=ax)
                else:
                    show((src, 1), ax=ax, cmap="gray")
        except Exception:
            pass
    if column and column in gdf.columns:
        gdf.plot(ax=ax, column=column, legend=True, alpha=0.5, linewidth=0.2, edgecolor="#1f2937")
    else:
        gdf.plot(ax=ax, alpha=0.35, color="#2f9e44", linewidth=0.2, edgecolor="#0b5d1e")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0)
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(dst_path)


def _copy_named_visualizations(result: FinalTreeCrownResult, layout: dict[str, Path]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name, src in (result.visualizations or {}).items():
        if not src:
            continue
        suffix = Path(src).suffix or ".png"
        dst_name = name if Path(name).suffix else f"{name}{suffix}"
        copied_path = _copy_optional_file(src, layout["visualization"] / dst_name)
        if copied_path:
            copied[Path(dst_name).stem] = copied_path
    return copied


def _write_optional_vector_exports(crowns_src: str | Path, layout: dict[str, Path], export_config: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if not any(bool(export_config.get(key)) for key in ["export_geojson", "export_gpkg", "export_csv"]):
        return outputs
    try:
        import geopandas as gpd
    except Exception:
        return outputs
    crowns_path = Path(crowns_src)
    if not crowns_path.exists():
        return outputs
    gdf = gpd.read_file(crowns_path)
    if gdf.empty:
        return outputs
    if bool(export_config.get("export_geojson")):
        geojson_path = layout["results"] / "tree_crowns.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        outputs["tree_crowns_geojson"] = str(geojson_path)
    if bool(export_config.get("export_gpkg")):
        gpkg_path = layout["results"] / "tree_crowns.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG")
        outputs["tree_crowns_gpkg"] = str(gpkg_path)
    if bool(export_config.get("export_csv")):
        csv_path = layout["results"] / "tree_crowns.csv"
        table = gdf.drop(columns="geometry", errors="ignore")
        table.to_csv(csv_path, index=False)
        outputs["tree_crowns_csv"] = str(csv_path)
    return outputs


def _render_scene_reports(result: FinalTreeCrownResult, scenario: str, output_metadata: dict[str, Any]) -> str:
    if result.report_markdown:
        return result.report_markdown
    if scenario == SCENARIO_COCO_GT:
        metrics = result.gt_metrics or {}
        metadata = result.metadata or {}
        images = metadata.get("images") or []
        categories = result.categories or metadata.get("categories") or []
        selected_errors = output_metadata.get("selected_error_examples") or {}
        fp_count = _first_metric(metrics, "false_positive_count", "fp")
        fn_count = _first_metric(metrics, "false_negative_count", "fn")
        low_iou_count = _first_metric(metrics, "low_iou_count", "boundary_error_count")
        if low_iou_count is None:
            low_iou_count = sum(
                1
                for item in result.instances
                if item.get("is_fp") is not True
                and item.get("is_fn") is not True
                and item.get("iou_gt", item.get("best_iou", item.get("iou"))) is not None
                and float(item.get("iou_gt", item.get("best_iou", item.get("iou")))) < 0.5
            )
        dataset_name = _first_metric(metrics, "dataset", "dataset_name") or metadata.get("dataset") or metadata.get("dataset_name") or "-"
        image_count = _first_metric(metrics, "image_count", "images") or len(images) or None
        gt_count = _first_metric(metrics, "gt_instance_count", "num_ground_truth")
        pred_count = _first_metric(metrics, "pred_instance_count", "num_predictions") or len(result.instances) or None

        lines = [
            "# COCO Tree Crown Evaluation Report",
            "",
            "## 1. 数据集基本信息",
            "",
            *_markdown_table(
                ["字段", "值"],
                [
                    ["Dataset", dataset_name],
                    ["Dataset Type", metadata.get("dataset_type") or "coco_instance_segmentation_with_gt"],
                    ["Image Root", metadata.get("image_root")],
                    ["Annotation", metadata.get("annotation_path")],
                    ["Images", image_count],
                    ["GT Instances", gt_count],
                    ["Pred Instances", pred_count],
                    ["Categories", len(categories) if categories else "-"],
                ],
            ),
            "",
            "## 2. 模型与推理配置",
            "",
            *_markdown_table(
                ["字段", "值"],
                [
                    ["Run ID", result.run_id],
                    ["Source Adapter", metadata.get("source_adapter")],
                    ["Model", metadata.get("model_name") or metadata.get("model") or metadata.get("algorithm")],
                    ["Checkpoint", metadata.get("checkpoint") or metadata.get("checkpoint_path")],
                    ["Input Type", result.input_type],
                    ["Coordinate Mode", output_metadata.get("coordinate_mode") or result.coordinate_mode],
                    ["Image Size", f"{result.image_width}x{result.image_height}" if result.image_width and result.image_height else "-"],
                    ["Inference Config", metadata.get("inference_config") or metadata.get("runtime_config")],
                ],
            ),
            "",
            "## 3. COCO 指标结果表",
            "",
            *_markdown_table(
                [
                    "Dataset",
                    "Images",
                    "GT Instances",
                    "Pred Instances",
                    "Precision",
                    "Recall",
                    "F1",
                    "Bbox AP50",
                    "Bbox AP75",
                    "Bbox AP",
                    "Mask AP50",
                    "Mask AP75",
                    "Mask AP",
                    "mIoU",
                    "FP",
                    "FN",
                ],
                [
                    [
                        dataset_name,
                        image_count,
                        gt_count,
                        pred_count,
                        metrics.get("precision"),
                        metrics.get("recall"),
                        _first_metric(metrics, "f1", "f1_score50"),
                        _first_metric(metrics, "bbox_ap50", "ap50"),
                        _first_metric(metrics, "bbox_ap75", "ap75"),
                        _first_metric(metrics, "bbox_ap", "ap", "bbox_ap_50_95"),
                        _first_metric(metrics, "mask_ap50", "segm_ap50"),
                        _first_metric(metrics, "mask_ap75", "segm_ap75"),
                        _first_metric(metrics, "mask_ap", "ap_50_95", "segm_ap"),
                        _first_metric(metrics, "miou", "mean_iou_matched"),
                        fp_count,
                        fn_count,
                    ]
                ],
            ),
            "",
            "## 4. 检测指标结果表",
            "",
            *_markdown_table(
                ["指标", "值"],
                [
                    ["Precision", metrics.get("precision")],
                    ["Recall", metrics.get("recall")],
                    ["F1", _first_metric(metrics, "f1", "f1_score50")],
                    ["Bbox AP50", _first_metric(metrics, "bbox_ap50", "ap50")],
                    ["Bbox AP75", _first_metric(metrics, "bbox_ap75", "ap75")],
                    ["Bbox AP", _first_metric(metrics, "bbox_ap", "ap", "bbox_ap_50_95")],
                    ["False Positives", fp_count],
                    ["False Negatives", fn_count],
                ],
                align_right=True,
            ),
            "",
            "## 5. 掩码分割指标结果表",
            "",
            *_markdown_table(
                ["指标", "值"],
                [
                    ["Mask AP50", _first_metric(metrics, "mask_ap50", "segm_ap50")],
                    ["Mask AP75", _first_metric(metrics, "mask_ap75", "segm_ap75")],
                    ["Mask AP", _first_metric(metrics, "mask_ap", "ap_50_95", "segm_ap")],
                    ["mIoU", _first_metric(metrics, "miou", "mean_iou_matched")],
                ],
                align_right=True,
            ),
            "",
            "## 6. 错误类型统计",
            "",
            *_markdown_table(
                ["错误类型", "数量", "案例图"],
                [
                    ["false_positive", fp_count, selected_errors.get("fp")],
                    ["false_negative", fn_count, selected_errors.get("fn")],
                    ["low_iou", low_iou_count, selected_errors.get("low_iou")],
                ],
            ),
            "",
            "## 7. 典型失败案例",
            "",
        ]
        case_rows = [
            [name, path]
            for name, path in [
                ["误检案例图", selected_errors.get("fp")],
                ["漏检案例图", selected_errors.get("fn")],
                ["低 IoU 案例图", selected_errors.get("low_iou")],
                ["样例叠加图", output_metadata.get("sample_overlay_png")],
            ]
        ]
        lines.extend(_markdown_table(["案例", "路径"], case_rows))
        if not any(value not in {None, "-", ""} for _, value in case_rows[:3]):
            lines.extend(["", "- 当前输出未提供典型错误案例图。"])
        elif (fp_count in {0, None}) and (fn_count in {0, None}) and (low_iou_count in {0, None}):
            lines.extend(["", "- 当前样例未检出误检、漏检或低 IoU 案例；对应图片为占位说明或空案例汇总。"])
        lines.extend(
            [
                "",
                "## 8. 总体结论",
                "",
                f"- 本次评估共覆盖 {_format_report_value(image_count)} 张图像、{_format_report_value(gt_count)} 个 GT 实例和 {_format_report_value(pred_count)} 个预测实例。",
                f"- 检测 F1 为 {_format_report_value(_first_metric(metrics, 'f1', 'f1_score50'))}，Mask AP 为 {_format_report_value(_first_metric(metrics, 'mask_ap', 'ap_50_95', 'segm_ap'))}。",
                f"- 错误统计：FP={_format_report_value(fp_count)}，FN={_format_report_value(fn_count)}，low_iou={_format_report_value(low_iou_count)}。",
                "",
                "## Output Summary",
                "",
                f"- coco_predictions: `{output_metadata.get('coco_predictions_json')}`",
            ]
        )
        return "\n".join(lines) + "\n"

    if scenario == SCENARIO_DOM_WITH_GT:
        metrics = result.gt_metrics or {}
        geometry = result.geometry_metrics or {}
        rows = [
            ("GT Count", metrics.get("gt_count") or metrics.get("num_ground_truth")),
            ("Pred Count", metrics.get("pred_count") or metrics.get("num_predictions")),
            ("Precision", metrics.get("precision")),
            ("Recall", metrics.get("recall")),
            ("F1", metrics.get("f1") or metrics.get("f1_score50")),
            ("Mask AP50", metrics.get("mask_ap50") or metrics.get("ap50")),
            ("Mask AP75", metrics.get("mask_ap75") or metrics.get("ap75")),
            ("mIoU", metrics.get("miou") or metrics.get("mean_iou_matched")),
            ("Mean Area Error(%)", geometry.get("mean_area_error_percent")),
            ("Mean Crown Width Error(%)", geometry.get("mean_crown_width_error_percent")),
        ]
        lines = ["# DOM Tree Crown Extraction Evaluation Report", "", "## Overall Evaluation Results", "", "| Metric | Value |", "|---|---:|"]
        lines.extend([f"| {name} | {value if value is not None else '-'} |" for name, value in rows])
        lines.extend(["", "## Visualization Summary", "", f"- pred_overlay: `{output_metadata.get('pred_overlay_png')}`"])
        return "\n".join(lines) + "\n"

    quality = result.no_gt_quality_metrics or result.geometry_metrics or {}
    lines = ["# DOM Tree Crown Extraction Inference Report", "", "## Prediction Result Summary", "", "| Metric | Value |", "|---|---:|"]
    for key in NO_GT_QUALITY_KEYS:
        lines.append(f"| {key} | {quality.get(key, '-')} |")
    lines.extend(["", "## Visualization Summary", "", f"- risk_map: `{output_metadata.get('risk_map_png')}`"])
    return "\n".join(lines) + "\n"


def _render_final_report(result: FinalTreeCrownResult, publish_root: Path, output_metadata: dict[str, Any], scenario: str) -> tuple[str, str]:
    layout = build_output_layout(publish_root)
    report_json = {
        "run_id": result.run_id,
        "status": "frozen",
        "output_contract": "FinalTreeCrownResult",
        "scenario": scenario,
        "gt_metrics": result.gt_metrics,
        "geometry_metrics": result.geometry_metrics,
        "no_gt_quality_metrics": result.no_gt_quality_metrics,
        "trajectory_paths": list(result.trajectory_paths),
        "outputs": output_metadata,
        "metadata": result.metadata,
    }
    if result.report_json:
        report_json["source_report"] = result.report_json
    report_stem = "inference_report" if scenario == SCENARIO_DOM_WITHOUT_GT else "evaluation_report"
    json_path = layout["report"] / f"{report_stem}.json"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    text = _render_scene_reports(result, scenario, output_metadata)
    md_path = layout["report"] / f"{report_stem}.md"
    md_path.write_text(text, encoding="utf-8")
    _copy_optional_file(md_path, publish_root / "final_report.md")
    _copy_optional_file(json_path, publish_root / "final_report.json")
    return str(md_path), str(json_path)


def publish_final_tree_crown_outputs(*, result: FinalTreeCrownResult, publish_root: str | Path) -> dict[str, Any]:
    root = Path(publish_root)
    ensure_dir(root)
    layout = build_output_layout(root)
    scenario = resolve_output_scenario(result)

    copied_visualizations = _copy_named_visualizations(result, layout)

    if scenario == SCENARIO_COCO_GT:
        coco_predictions = _write_coco_predictions(result, layout["results"] / "coco_predictions.json")
        instance_mask_paths = _write_coco_instance_masks(
            result=result,
            masks_dir=layout["masks"] / "instance_masks",
            max_masks=int(result.visualization_config.get("max_instance_masks", 0) or 0) or None,
        )
        if result.instance_mask_png:
            copied_instance_png = _copy_optional_file(result.instance_mask_png, layout["masks"] / "instance_masks" / "image_primary_instance_mask.png")
            if copied_instance_png:
                instance_mask_paths = [copied_instance_png, *instance_mask_paths]

        sample_overlay_paths = _write_coco_sample_overlays(result, layout)
        selected_error_examples = _write_coco_selected_error_examples(result, layout)
        sample_overlay = copied_visualizations.get("sample_overlay") or copied_visualizations.get("pred_overlay")

        # Backward compatibility for existing evolution callers that consumed geospatial-looking paths from COCO runs.
        write_legacy_compat = bool(result.metadata.get("write_legacy_compat_outputs")) or str(result.metadata.get("source_adapter") or "") == "coco_evolve_infer"
        legacy_tree_crowns = root / "tree_crowns.shp"
        legacy_tree_points = root / "tree_points.shp"
        legacy_materialized = None
        legacy_metadata: dict[str, Any] = {"geometry_source": "not_required_for_coco", "coordinate_mode": result.coordinate_mode}
        if write_legacy_compat and (result.crown_vector_path or result.instances):
            legacy_materialized = _materialize_tree_crowns_vector(
                result.crown_vector_path,
                legacy_tree_crowns,
                scenario=SCENARIO_DOM_WITH_GT,
                gt_matches=result.gt_matches,
            )
            if legacy_materialized is None and result.instances:
                legacy_materialized, legacy_metadata = _write_instances_as_tree_crowns(
                    instances=result.instances,
                    dst=legacy_tree_crowns,
                    input_dom_path=result.input_dom_path,
                    image_width=result.image_width,
                    image_height=result.image_height,
                    scenario=SCENARIO_DOM_WITH_GT,
                    gt_matches=result.gt_matches,
                )
            if legacy_materialized:
                build_tree_points(legacy_tree_crowns, legacy_tree_points)

        if sample_overlay_paths:
            sample_overlay = sample_overlay or sample_overlay_paths[0]
        elif not sample_overlay and legacy_materialized:
            sample_overlay = render_segmentation_visualization(
                legacy_tree_crowns,
                legacy_tree_points if legacy_tree_points.exists() else None,
                layout["visualization"] / "sample_overlays" / "sample_000001_overlay.png",
                background_raster=result.input_dom_path,
            )

        fallback_semantic = {"semantic_mask_tif": None, "semantic_mask_png": None, "semantic_mask_source": "not_required_for_coco"}
        if legacy_materialized:
            fallback_semantic = _write_fallback_semantic_mask(
                crowns_src=legacy_tree_crowns,
                input_dom_path=result.input_dom_path,
                dst_tif=root / "semantic_mask.tif",
                dst_png=root / "semantic_mask.png",
                image_width=result.image_width,
                image_height=result.image_height,
            )

        output_metadata = {
            "run_name": result.run_id,
            "publish_root": str(root),
            "scenario": scenario,
            "status": "published" if coco_predictions or result.instances else "missing_coco_predictions",
            "coco_predictions_json": coco_predictions,
            "instance_mask_paths": instance_mask_paths,
            "instance_mask_png": instance_mask_paths[0] if instance_mask_paths else None,
            "sample_overlay_paths": sample_overlay_paths,
            "selected_error_examples": selected_error_examples,
            "tree_crowns_shp": str(legacy_tree_crowns) if legacy_tree_crowns.exists() else None,
            "tree_points_shp": str(legacy_tree_points) if legacy_tree_points.exists() else None,
            "semantic_mask_tif": fallback_semantic.get("semantic_mask_tif"),
            "semantic_mask_png": fallback_semantic.get("semantic_mask_png"),
            "semantic_prior_tif": fallback_semantic.get("semantic_mask_tif"),
            "semantic_prior_png": fallback_semantic.get("semantic_mask_png"),
            "semantic_mask_source": fallback_semantic.get("semantic_mask_source"),
            "segmentation_visualization_png": sample_overlay,
            "sample_overlay_png": sample_overlay,
            "crown_geometry_source": legacy_metadata.get("geometry_source"),
            "coordinate_mode": legacy_metadata.get("coordinate_mode"),
        }
        report_md, report_json = _render_final_report(result, root, output_metadata, scenario)
        bundle_path = root / "final_result_bundle.json"
        final_bundle = {
            **output_metadata,
            "final_report_md": report_md,
            "final_report_json": report_json,
            "final_evaluation_report_md": report_md,
            "final_evaluation_report_json": report_json,
        }
        bundle_path.write_text(json.dumps(final_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        final_bundle["final_result_bundle_json"] = str(bundle_path)
        return final_bundle

    tree_crowns_shp = layout["results"] / "tree_crowns.shp"
    vector_metadata: dict[str, Any] = {"geometry_source": "source_vector", "coordinate_mode": result.coordinate_mode}
    materialized = _materialize_tree_crowns_vector(
        result.crown_vector_path,
        tree_crowns_shp,
        scenario=scenario,
        gt_matches=result.gt_matches,
    )
    if materialized is None and result.instances:
        materialized, vector_metadata = _write_instances_as_tree_crowns(
            instances=result.instances,
            dst=tree_crowns_shp,
            input_dom_path=result.input_dom_path,
            image_width=result.image_width,
            image_height=result.image_height,
            scenario=scenario,
            gt_matches=result.gt_matches,
        )
    if materialized is None:
        return {
            "run_name": result.run_id,
            "publish_root": str(root),
            "status": "missing_tree_crowns",
            "tree_crowns_shp": None,
            "tree_points_shp": None,
            "semantic_mask_tif": None,
            "semantic_mask_png": None,
            "segmentation_visualization_png": None,
            "final_report_md": None,
            "final_report_json": None,
        }

    tree_points_shp = layout["results"] / "tree_points.shp"
    point_files = build_tree_points(tree_crowns_shp, tree_points_shp)
    semantic_mask_tif = _copy_optional_file(result.semantic_mask_tif, layout["masks"] / "semantic_mask.tif")
    semantic_mask_png = _copy_optional_file(result.semantic_mask_png, layout["masks"] / "semantic_mask.png")
    semantic_mask_source = "source_semantic_mask" if semantic_mask_tif or semantic_mask_png else "none"
    if not semantic_mask_tif:
        fallback = _write_fallback_semantic_mask(
            crowns_src=tree_crowns_shp,
            input_dom_path=result.input_dom_path,
            dst_tif=layout["masks"] / "semantic_mask.tif",
            dst_png=layout["masks"] / "semantic_mask.png",
            image_width=result.image_width,
            image_height=result.image_height,
        )
        semantic_mask_tif = fallback.get("semantic_mask_tif")
        semantic_mask_png = fallback.get("semantic_mask_png")
        semantic_mask_source = str(fallback.get("semantic_mask_source"))
    instance_mask_tif = _copy_optional_file(result.instance_mask_tif, layout["masks"] / "instance_mask.tif")
    instance_mask_png = _copy_optional_file(result.instance_mask_png, layout["masks"] / "instance_mask.png")
    instance_mask_source = "source_instance_mask" if instance_mask_tif or instance_mask_png else "none"
    if not instance_mask_tif:
        instance_fallback = _write_instance_mask_from_crowns(
            crowns_src=tree_crowns_shp,
            input_dom_path=result.input_dom_path,
            dst_tif=layout["masks"] / "instance_mask.tif",
            dst_png=layout["masks"] / "instance_mask.png",
            image_width=result.image_width,
            image_height=result.image_height,
        )
        instance_mask_tif = instance_fallback.get("instance_mask_tif")
        instance_mask_png = instance_fallback.get("instance_mask_png")
        instance_mask_source = str(instance_fallback.get("instance_mask_source"))
    semantic_prior_tif = _copy_optional_file(semantic_mask_tif, root / "M_sem.tif") if semantic_mask_tif else None
    semantic_prior_png = _copy_optional_file(semantic_mask_png, root / "M_sem.png") if semantic_mask_png else None
    legacy_tree_crowns = root / "tree_crowns.shp"
    legacy_tree_points = root / "tree_points.shp"
    legacy_semantic_tif = root / "semantic_mask.tif"
    legacy_semantic_png = root / "semantic_mask.png"
    legacy_instance_tif = root / "instance_mask.tif"
    legacy_instance_png = root / "instance_mask.png"
    _copy_optional_file(tree_crowns_shp, legacy_tree_crowns)
    copy_vector_dataset(tree_crowns_shp, root / "tree_crowns.shp")
    copy_vector_dataset(tree_points_shp, root / "tree_points.shp")
    _copy_optional_file(semantic_mask_tif, legacy_semantic_tif) if semantic_mask_tif else None
    _copy_optional_file(semantic_mask_png, legacy_semantic_png) if semantic_mask_png else None
    _copy_optional_file(instance_mask_tif, legacy_instance_tif) if instance_mask_tif else None
    _copy_optional_file(instance_mask_png, legacy_instance_png) if instance_mask_png else None

    visualization_png = layout["visualization"] / "pred_overlay.png"
    pred_overlay = copied_visualizations.get("pred_overlay") or render_segmentation_visualization(
        tree_crowns_shp,
        tree_points_shp if point_files else None,
        visualization_png,
        background_raster=result.input_dom_path,
    )
    instance_boundaries = copied_visualizations.get("instance_boundaries") or _render_vector_thematic_map(
        crowns_src=tree_crowns_shp,
        dst=layout["visualization"] / "instance_boundaries.png",
        background_raster=result.input_dom_path,
    )
    gt_pred_overlay = None
    evaluation_map = None
    confidence_map = None
    risk_map = None
    if scenario == SCENARIO_DOM_WITH_GT:
        gt_pred_overlay = copied_visualizations.get("gt_pred_overlay") or pred_overlay
        evaluation_map = copied_visualizations.get("evaluation_map") or _render_vector_thematic_map(
            crowns_src=tree_crowns_shp,
            dst=layout["visualization"] / "evaluation_map.png",
            column="eval_type",
            background_raster=result.input_dom_path,
        )
    else:
        confidence_map = copied_visualizations.get("confidence_map") or _render_vector_thematic_map(
            crowns_src=tree_crowns_shp,
            dst=layout["visualization"] / "confidence_map.png",
            column="score",
            background_raster=result.input_dom_path,
        )
        risk_map = copied_visualizations.get("risk_map") or _render_vector_thematic_map(
            crowns_src=tree_crowns_shp,
            dst=layout["visualization"] / "risk_map.png",
            column="risk_type",
            background_raster=result.input_dom_path,
        )
    _copy_optional_file(pred_overlay, root / "segmentation_visualization.png") if pred_overlay else None
    height_vector = root / "tree_crowns_height_structure.gpkg"
    height_summary = root / "height_structure_summary.json"
    height_outputs = (
        build_height_structure_outputs(
            crowns_src=tree_crowns_shp,
            chm_raster=result.chm_raster,
            annotated_vector_dst=height_vector,
            summary_dst=height_summary,
        )
        if result.chm_raster
        else {"available": False, "reason": "missing_chm"}
    )
    optional_exports = _write_optional_vector_exports(tree_crowns_shp, layout, result.export_config or {})
    output_metadata = {
        "run_name": result.run_id,
        "publish_root": str(root),
        "scenario": scenario,
        "status": "published",
        "tree_crowns_shp": str(legacy_tree_crowns) if legacy_tree_crowns.exists() else str(tree_crowns_shp),
        "tree_points_shp": str(legacy_tree_points) if legacy_tree_points.exists() else str(tree_points_shp) if tree_points_shp.exists() else None,
        "results_tree_crowns_shp": str(tree_crowns_shp),
        "results_tree_points_shp": str(tree_points_shp) if tree_points_shp.exists() else None,
        "semantic_mask_tif": str(legacy_semantic_tif) if legacy_semantic_tif.exists() else semantic_mask_tif,
        "semantic_mask_png": str(legacy_semantic_png) if legacy_semantic_png.exists() else semantic_mask_png,
        "instance_mask_tif": str(legacy_instance_tif) if legacy_instance_tif.exists() else instance_mask_tif,
        "instance_mask_png": str(legacy_instance_png) if legacy_instance_png.exists() else instance_mask_png,
        "masks_semantic_mask_tif": semantic_mask_tif,
        "masks_semantic_mask_png": semantic_mask_png,
        "masks_instance_mask_tif": instance_mask_tif,
        "masks_instance_mask_png": instance_mask_png,
        "instance_mask_source": instance_mask_source,
        "semantic_prior_tif": semantic_prior_tif or semantic_mask_tif,
        "semantic_prior_png": semantic_prior_png or semantic_mask_png,
        "semantic_mask_source": semantic_mask_source,
        "segmentation_visualization_png": pred_overlay,
        "pred_overlay_png": pred_overlay,
        "gt_pred_overlay_png": gt_pred_overlay,
        "instance_boundaries_png": instance_boundaries,
        "evaluation_map_png": evaluation_map,
        "confidence_map_png": confidence_map,
        "risk_map_png": risk_map,
        "tree_crowns_height_structure_gpkg": height_outputs.get("annotated_vector") if height_outputs.get("available") else None,
        "height_structure_summary_json": height_outputs.get("summary_json") if height_outputs.get("available") else None,
        "height_structure_summary": height_outputs,
        "optional_exports": optional_exports,
        "crown_geometry_source": vector_metadata.get("geometry_source"),
        "coordinate_mode": vector_metadata.get("coordinate_mode"),
    }
    report_md, report_json = _render_final_report(result, root, output_metadata, scenario)
    bundle_path = root / "final_result_bundle.json"
    final_bundle = {
        **output_metadata,
        "final_report_md": report_md,
        "final_report_json": report_json,
        "final_evaluation_report_md": report_md,
        "final_evaluation_report_json": report_json,
    }
    bundle_path.write_text(json.dumps(final_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    final_bundle["final_result_bundle_json"] = str(bundle_path)
    return final_bundle


def publish_segmentation_deliverables(
    *,
    inst_shp: str | None,
    publish_root: str | Path,
    semantic_prior_tif: str | None = None,
    semantic_prior_png: str | None = None,
    report_path: str | None = None,
    report_json_path: str | None = None,
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
    run_name: str | None = None,
    background_raster: str | None = None,
    mainline_profile: str | None = None,
    chm_raster: str | None = None,
    enable_height_structure: bool = False,
) -> dict[str, Any]:
    report_markdown = Path(report_path).read_text(encoding="utf-8") if report_path and Path(report_path).exists() else None
    report_json = None
    if report_json_path and Path(report_json_path).exists():
        try:
            report_json = json.loads(Path(report_json_path).read_text(encoding="utf-8"))
        except Exception:
            report_json = None
    result = FinalTreeCrownResult(
        run_id=str(run_name or "unknown_run"),
        output_dir=str(publish_root),
        input_dom_path=background_raster,
        crown_vector_path=inst_shp,
        semantic_mask_tif=semantic_prior_tif,
        semantic_mask_png=semantic_prior_png,
        chm_raster=chm_raster if enable_height_structure else None,
        report_markdown=report_markdown,
        report_json=report_json,
        metadata={
            "mainline_profile": mainline_profile,
            "metrics_json": metrics_json,
            "details_csv": details_csv,
            "summary_json": summary_json,
            "compat_entrypoint": "publish_segmentation_deliverables",
        },
    )
    published = publish_final_tree_crown_outputs(result=result, publish_root=publish_root)
    if published.get("status") == "missing_tree_crowns":
        published["status"] = "missing_inst_shp"
    return published
