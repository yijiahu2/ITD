from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


VECTOR_DATASET_EXTS = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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
    if background_raster and Path(background_raster).exists():
        try:
            import rasterio

            with rasterio.open(background_raster) as src:
                raster_crs = src.crs
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

    if background_raster and Path(background_raster).exists() and raster_crs is not None:
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
    root = Path(publish_root)
    ensure_dir(root)
    if not inst_shp or not Path(inst_shp).exists():
        return {
            "run_name": run_name,
            "publish_root": str(root),
            "status": "missing_inst_shp",
            "tree_crowns_shp": None,
            "tree_points_shp": None,
            "semantic_prior_tif": None,
            "semantic_prior_png": None,
            "segmentation_visualization_png": None,
            "final_evaluation_report_md": None,
            "final_evaluation_report_json": None,
            "tree_crowns_height_structure_gpkg": None,
            "height_structure_summary_json": None,
        }

    tree_crowns_shp = root / "tree_crowns.shp"
    tree_points_shp = root / "tree_points.shp"
    semantic_prior_tif_copy = root / "M_sem.tif"
    semantic_prior_png_copy = root / "M_sem.png"
    visualization_png = root / "segmentation_visualization.png"
    report_copy = root / "final_evaluation_report.md"
    report_json_copy = root / "final_evaluation_report.json"
    height_vector = root / "tree_crowns_height_structure.gpkg"
    height_summary = root / "height_structure_summary.json"

    copy_vector_dataset(inst_shp, tree_crowns_shp)
    point_files = build_tree_points(inst_shp, tree_points_shp)
    render_segmentation_visualization(
        tree_crowns_shp,
        tree_points_shp if point_files else None,
        visualization_png,
        background_raster=background_raster,
    )
    semantic_prior_tif_published = _copy_optional_file(semantic_prior_tif, semantic_prior_tif_copy)
    semantic_prior_png_published = _copy_optional_file(semantic_prior_png, semantic_prior_png_copy)
    height_outputs = (
        build_height_structure_outputs(
            crowns_src=tree_crowns_shp,
            chm_raster=chm_raster,
            annotated_vector_dst=height_vector,
            summary_dst=height_summary,
        )
        if enable_height_structure
        else {"available": False, "reason": "disabled"}
    )

    return {
        "run_name": run_name,
        "mainline_profile": mainline_profile,
        "publish_root": str(root),
        "tree_crowns_shp": str(tree_crowns_shp),
        "tree_points_shp": str(tree_points_shp) if tree_points_shp.exists() else None,
        "semantic_prior_tif": semantic_prior_tif_published,
        "semantic_prior_png": semantic_prior_png_published,
        "segmentation_visualization_png": str(visualization_png) if visualization_png.exists() else None,
        "final_evaluation_report_md": _copy_optional_file(report_path, report_copy),
        "final_evaluation_report_json": _copy_optional_file(report_json_path, report_json_copy),
        "tree_crowns_height_structure_gpkg": height_outputs.get("annotated_vector") if height_outputs.get("available") else None,
        "height_structure_summary_json": height_outputs.get("summary_json") if height_outputs.get("available") else None,
        "height_structure_summary": height_outputs,
    }
