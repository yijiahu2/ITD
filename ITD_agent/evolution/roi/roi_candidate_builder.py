from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ITD_agent.evaluation_analysis.coco_error_decomposition import CocoErrorDecompositionResult
from ITD_agent.evolution.bbox import clamp_bbox


FAILURE_FAMILY_BY_ERROR = {
    "false_negative": "small_crown_recall",
    "false_positive": "false_positive_cleanup",
    "under_segmentation": "crown_split",
    "over_segmentation": "crown_merge_cleanup",
    "tiny_false_positive": "false_positive_cleanup",
    "fragmented_boundary": "crown_merge_cleanup",
    "unstable_edge_mask": "boundary_refinement",
}


@dataclass(frozen=True)
class ROICandidate:
    roi_id: str
    image_id: str
    level1_error_type: str
    failure_family: str
    affected_gt_ids: list[str] = field(default_factory=list)
    affected_pred_ids: list[str] = field(default_factory=list)
    bbox_px: list[float] = field(default_factory=list)
    severity_score: float = 0.0
    confidence_level: str = "high"
    tags: list[str] = field(default_factory=list)
    geometry: dict[str, Any] = field(default_factory=dict)
    review_status: str = "monitor"
    expert_eligible: bool = False
    training_eligible: bool = False
    distill_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_roi_candidates(
    *,
    image_id: str,
    image_size: tuple[int, int],
    error_decomposition: CocoErrorDecompositionResult,
    geometry_review: dict[str, Any] | None = None,
) -> list[ROICandidate]:
    tags_by_type = [item.get("tag") for item in (geometry_review or {}).get("failure_tags", []) if item.get("tag")]
    rois: list[ROICandidate] = []
    for idx, error in enumerate(error_decomposition.errors, start=1):
        level1 = str(error["level1_error_type"])
        bbox = clamp_bbox(tuple(float(v) for v in error.get("bbox_px", [0, 0, 0, 0])), image_size)
        family = FAILURE_FAMILY_BY_ERROR.get(level1, "boundary_refinement")
        rois.append(
            ROICandidate(
                roi_id=f"roi_{image_id}_{idx:04d}",
                image_id=str(image_id),
                level1_error_type=level1,
                failure_family=family,
                affected_gt_ids=[str(v) for v in error.get("affected_gt_ids", [])],
                affected_pred_ids=[str(v) for v in error.get("affected_pred_ids", [])],
                bbox_px=list(bbox),
                severity_score=max(0.0, min(1.0, float(error.get("severity_score", 0.0)))),
                confidence_level="high" if error.get("affected_gt_ids") else "medium",
                tags=[level1, *tags_by_type],
                geometry={"source_error_id": error.get("error_id")},
            )
        )
    return rois
