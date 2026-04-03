from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


def assign_instances_to_polygons(
    inst_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    id_field: str,
    method: str = "max_overlap",
) -> gpd.GeoDataFrame:
    inst = inst_gdf.copy()
    inst = inst.drop(columns=[id_field], errors="ignore")
    inst["inst_id"] = range(len(inst))

    if inst.empty:
        return inst

    polygons = polygon_gdf[[id_field, "geometry"]].copy()
    polygons = polygons[polygons.geometry.notnull() & (~polygons.geometry.is_empty)].copy()
    if polygons.empty:
        inst[id_field] = pd.NA
        return inst

    inst = inst.to_crs(polygons.crs)

    if method == "centroid":
        cent = inst.copy()
        cent.geometry = cent.centroid
        joined = gpd.sjoin(cent, polygons, how="left", predicate="within")
        return inst.merge(joined[["inst_id", id_field]], on="inst_id", how="left")

    ov = gpd.overlay(inst[["inst_id", "geometry"]], polygons, how="intersection")
    ov = ov[ov.geometry.notnull() & (~ov.geometry.is_empty)].copy()

    if ov.empty:
        cent = inst.copy()
        cent.geometry = cent.centroid
        joined = gpd.sjoin(cent, polygons, how="left", predicate="within")
        return inst.merge(joined[["inst_id", id_field]], on="inst_id", how="left")

    ov["overlap_area_m2"] = ov.geometry.area
    ov = ov.sort_values(["inst_id", "overlap_area_m2"], ascending=[True, False])
    best = ov.groupby("inst_id", as_index=False).first()[["inst_id", id_field, "overlap_area_m2"]]
    return inst.merge(best, on="inst_id", how="left")


def filter_instances_to_ids_by_overlap(
    inst_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    id_field: str,
    allowed_ids: Iterable[str],
) -> gpd.GeoDataFrame:
    assigned = assign_instances_to_polygons(inst_gdf, polygon_gdf, id_field=id_field, method="max_overlap")
    allowed = {str(x) for x in allowed_ids}
    assigned[id_field] = assigned[id_field].astype(str)
    filtered = assigned[assigned[id_field].isin(allowed)].copy()
    return filtered.drop(columns=["inst_id", "overlap_area_m2"], errors="ignore")


def overlap_share_with_geom(geom, region_geom) -> float:
    if geom is None or geom.is_empty or region_geom is None or region_geom.is_empty:
        return 0.0
    area = float(getattr(geom, "area", 0.0) or 0.0)
    if area <= 0:
        return 0.0
    inter_area = float(geom.intersection(region_geom).area)
    return inter_area / area


def dedupe_instances_by_overlap(
    inst_gdf: gpd.GeoDataFrame,
    overlap_ratio_thr: float = 0.6,
) -> gpd.GeoDataFrame:
    if inst_gdf.empty:
        return inst_gdf.copy()

    ordered = inst_gdf.copy()
    ordered["_orig_idx"] = range(len(ordered))
    ordered["_area_m2"] = ordered.geometry.area.astype(float)
    ordered = ordered.sort_values("_area_m2", ascending=False).reset_index(drop=True)
    sindex = ordered.sindex
    keep = [True] * len(ordered)

    for i, geom in enumerate(ordered.geometry):
        if not keep[i] or geom is None or geom.is_empty:
            continue
        for j in sindex.intersection(geom.bounds):
            if j <= i or not keep[j]:
                continue
            other = ordered.geometry.iloc[j]
            if other is None or other.is_empty or not geom.intersects(other):
                continue
            inter_area = float(geom.intersection(other).area)
            if inter_area <= 0:
                continue
            denom = max(min(float(ordered["_area_m2"].iloc[i]), float(ordered["_area_m2"].iloc[j])), 1e-6)
            if inter_area / denom >= overlap_ratio_thr:
                keep[j] = False

    deduped = ordered[pd.Series(keep, index=ordered.index)].copy()
    deduped = deduped.sort_values("_orig_idx").drop(columns=["_orig_idx", "_area_m2"], errors="ignore")
    return gpd.GeoDataFrame(deduped, geometry="geometry", crs=inst_gdf.crs)


def _metric_crs_for_instances(inst_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame | None = None):
    ref = boundary_gdf if boundary_gdf is not None and not boundary_gdf.empty else inst_gdf
    if ref.crs is None:
        return inst_gdf.crs
    if getattr(ref.crs, "is_projected", False):
        return ref.crs
    try:
        utm = ref.estimate_utm_crs()
        if utm is not None:
            return utm
    except Exception:
        pass
    return "EPSG:3857"


def _equivalent_width(area_m2: float) -> float:
    if area_m2 <= 0:
        return 0.0
    return 2.0 * ((area_m2 / 3.141592653589793) ** 0.5)


def _bridge_split_geometry(geom, close_gap_m: float):
    if geom is None or geom.is_empty:
        return geom
    bridged = geom.buffer(close_gap_m).buffer(-close_gap_m) if close_gap_m > 0 else geom
    try:
        cleaned = bridged.buffer(0)
    except Exception:
        cleaned = bridged
    return cleaned if cleaned is not None and not cleaned.is_empty else geom


def merge_split_instances_by_proximity(
    inst_gdf: gpd.GeoDataFrame,
    *,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    boundary_band_m: float = 1.5,
    merge_gap_m: float = 0.8,
    centroid_distance_factor: float = 1.35,
    max_centroid_distance_m: float = 6.0,
    overlap_ratio_guard: float = 0.2,
    min_fill_ratio: float = 0.42,
    max_area_inflation: float = 1.30,
) -> gpd.GeoDataFrame:
    if inst_gdf.empty or len(inst_gdf) < 2 or merge_gap_m <= 0:
        return inst_gdf.copy()

    metric_crs = _metric_crs_for_instances(inst_gdf, boundary_gdf)
    work = inst_gdf.to_crs(metric_crs).copy()
    work["_merge_idx"] = range(len(work))
    work["_area_m2"] = work.geometry.area.astype(float)
    work = work[work.geometry.notnull() & (~work.geometry.is_empty) & (work["_area_m2"] > 0)].copy()
    if len(work) < 2:
        return inst_gdf.copy()

    boundary_band_geom = None
    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_metric = boundary_gdf.to_crs(metric_crs)
        try:
            boundary_union = boundary_metric.boundary.union_all()
        except Exception:
            boundary_union = unary_union(boundary_metric.boundary.tolist())
        boundary_band_geom = boundary_union.buffer(boundary_band_m)
        candidates = work[work.geometry.intersects(boundary_band_geom)].copy()
    else:
        candidates = work.copy()

    if len(candidates) < 2:
        return inst_gdf.copy()

    candidate_ids = list(candidates["_merge_idx"])
    candidate_set = set(candidate_ids)
    parent = {idx: idx for idx in candidate_ids}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    sindex = candidates.sindex
    close_gap_m = min(float(merge_gap_m) * 0.5, 0.6)

    for _, row in candidates.iterrows():
        idx_i = int(row["_merge_idx"])
        geom_i = row.geometry
        area_i = float(row["_area_m2"])
        if geom_i is None or geom_i.is_empty or area_i <= 0:
            continue
        search_geom = geom_i.buffer(merge_gap_m)
        for loc in sindex.intersection(search_geom.bounds):
            other = candidates.iloc[loc]
            idx_j = int(other["_merge_idx"])
            if idx_j <= idx_i or idx_j not in candidate_set:
                continue
            geom_j = other.geometry
            area_j = float(other["_area_m2"])
            if geom_j is None or geom_j.is_empty or area_j <= 0:
                continue
            if _find(idx_i) == _find(idx_j):
                continue
            if geom_i.distance(geom_j) > merge_gap_m:
                continue

            inter_area = float(geom_i.intersection(geom_j).area)
            min_area = max(min(area_i, area_j), 1e-6)
            if inter_area / min_area > overlap_ratio_guard:
                continue

            width_i = _equivalent_width(area_i)
            width_j = _equivalent_width(area_j)
            centroid_limit = min(max_centroid_distance_m, max(width_i, width_j) * centroid_distance_factor + merge_gap_m)
            if geom_i.centroid.distance(geom_j.centroid) > centroid_limit:
                continue

            raw_union = unary_union([geom_i, geom_j])
            merged_geom = _bridge_split_geometry(raw_union, close_gap_m=close_gap_m)
            if merged_geom is None or merged_geom.is_empty:
                continue

            raw_area = max(float(raw_union.area), 1e-6)
            merged_area = float(merged_geom.area)
            area_inflation = merged_area / raw_area
            hull_area = max(float(merged_geom.convex_hull.area), 1e-6)
            fill_ratio = merged_area / hull_area
            if area_inflation > max_area_inflation or fill_ratio < min_fill_ratio:
                continue

            if boundary_band_geom is not None and not merged_geom.intersects(boundary_band_geom):
                continue

            _union(idx_i, idx_j)

    clusters: dict[int, list[int]] = {}
    for idx in candidate_ids:
        root = _find(idx)
        clusters.setdefault(root, []).append(idx)

    if not any(len(members) > 1 for members in clusters.values()):
        return inst_gdf.copy()

    work_indexed = work.set_index("_merge_idx", drop=False)
    kept_rows = []
    consumed: set[int] = set()

    for merge_idx, row in work_indexed.iterrows():
        if merge_idx in consumed:
            continue
        if int(merge_idx) not in candidate_set:
            kept_rows.append(row.copy())
            consumed.add(int(merge_idx))
            continue
        cluster = clusters.get(_find(int(merge_idx)), [int(merge_idx)])
        if len(cluster) == 1:
            kept_rows.append(row.copy())
            consumed.add(int(merge_idx))
            continue

        cluster_rows = work_indexed.loc[cluster].copy()
        cluster_rows = cluster_rows.sort_values("_area_m2", ascending=False)
        base = cluster_rows.iloc[0].copy()
        merged_geom = unary_union(cluster_rows.geometry.tolist())
        merged_geom = _bridge_split_geometry(merged_geom, close_gap_m=close_gap_m)
        base.geometry = merged_geom
        base["_area_m2"] = float(merged_geom.area)
        kept_rows.append(base)
        consumed.update(int(v) for v in cluster_rows["_merge_idx"].tolist())

    merged = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=metric_crs)
    merged = merged.sort_values("_merge_idx").drop(columns=["_merge_idx", "_area_m2"], errors="ignore")
    return merged.to_crs(inst_gdf.crs) if inst_gdf.crs else merged


def suppress_small_boundary_fragments(
    inst_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    boundary_band_m: float = 1.5,
    min_area_m2: float = 6.0,
) -> gpd.GeoDataFrame:
    if inst_gdf.empty or polygon_gdf.empty or boundary_band_m <= 0 or min_area_m2 <= 0:
        return inst_gdf.copy()

    inst = inst_gdf.to_crs(polygon_gdf.crs).copy()
    boundaries = polygon_gdf.boundary
    boundary_band = boundaries.buffer(boundary_band_m)
    try:
        band_geom = boundary_band.union_all()
    except Exception:
        band_geom = boundary_band.unary_union

    areas = inst.geometry.area.astype(float)
    near_boundary = inst.geometry.intersects(band_geom)
    small = areas < float(min_area_m2)
    kept = inst[~(near_boundary & small)].copy()
    return gpd.GeoDataFrame(kept, geometry="geometry", crs=polygon_gdf.crs)
