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
    report_path: str | None = None,
    report_json_path: str | None = None,
    metrics_json: str | None = None,
    details_csv: str | None = None,
    summary_json: str | None = None,
    run_name: str | None = None,
    background_raster: str | None = None,
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
            "segmentation_visualization_png": None,
            "final_evaluation_report_md": None,
            "final_evaluation_report_json": None,
        }

    tree_crowns_shp = root / "tree_crowns.shp"
    tree_points_shp = root / "tree_points.shp"
    visualization_png = root / "segmentation_visualization.png"
    report_copy = root / "final_evaluation_report.md"
    report_json_copy = root / "final_evaluation_report.json"

    copy_vector_dataset(inst_shp, tree_crowns_shp)
    point_files = build_tree_points(inst_shp, tree_points_shp)
    render_segmentation_visualization(
        tree_crowns_shp,
        tree_points_shp if point_files else None,
        visualization_png,
        background_raster=background_raster,
    )

    return {
        "run_name": run_name,
        "publish_root": str(root),
        "tree_crowns_shp": str(tree_crowns_shp),
        "tree_points_shp": str(tree_points_shp) if tree_points_shp.exists() else None,
        "segmentation_visualization_png": str(visualization_png) if visualization_png.exists() else None,
        "final_evaluation_report_md": _copy_optional_file(report_path, report_copy),
        "final_evaluation_report_json": _copy_optional_file(report_json_path, report_json_copy),
    }
