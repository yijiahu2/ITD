from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ITD_agent.evolution.bbox import clamp_bbox, union_bbox

from .roi_candidate_builder import ROICandidate


@dataclass(frozen=True)
class ROICluster:
    cluster_id: str
    image_id: str
    failure_family: str
    level1_error_type: str
    roi_ids: list[str] = field(default_factory=list)
    tile_window_px: list[float] = field(default_factory=list)
    fusion_bboxes: dict[str, list[float]] = field(default_factory=dict)
    severity_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tile_for_bbox(bbox: tuple[float, float, float, float], image_size: tuple[int, int], tile_size: int) -> list[float]:
    width, height = image_size
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    x1 = max(0.0, min(float(width - tile_size), center_x - tile_size / 2.0)) if width > tile_size else 0.0
    y1 = max(0.0, min(float(height - tile_size), center_y - tile_size / 2.0)) if height > tile_size else 0.0
    return list(clamp_bbox((x1, y1, x1 + tile_size, y1 + tile_size), image_size))


def cluster_rois_for_expert_tiles(
    *,
    roi_candidates: list[ROICandidate],
    image_size: tuple[int, int],
    tile_size: int = 1024,
) -> list[ROICluster]:
    groups: dict[tuple[str, str, str], list[ROICandidate]] = {}
    for roi in roi_candidates:
        if not roi.expert_eligible:
            continue
        key = (roi.image_id, roi.failure_family, roi.level1_error_type)
        groups[key] = [*groups.get(key, []), roi]

    clusters: list[ROICluster] = []
    for idx, ((image_id, failure_family, level1), rois) in enumerate(sorted(groups.items()), start=1):
        merged = union_bbox([tuple(roi.bbox_px) for roi in rois])
        clusters.append(
            ROICluster(
                cluster_id=f"cluster_{image_id}_{idx:04d}",
                image_id=image_id,
                failure_family=failure_family,
                level1_error_type=level1,
                roi_ids=[roi.roi_id for roi in rois],
                tile_window_px=_tile_for_bbox(merged, image_size, tile_size),
                fusion_bboxes={roi.roi_id: list(roi.bbox_px) for roi in rois},
                severity_score=max(roi.severity_score for roi in rois),
            )
        )
    return clusters
