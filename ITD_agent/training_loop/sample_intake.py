from __future__ import annotations

from ITD_agent.evolution.roi.roi_candidate_builder import ROICandidate

from .contracts import TrainingCandidate


def intake_training_candidates_dry_run(
    *,
    trajectory_id: str,
    roi_candidates: list[ROICandidate],
) -> list[TrainingCandidate]:
    candidates: list[TrainingCandidate] = []
    for idx, roi in enumerate(roi_candidates, start=1):
        if not roi.training_eligible:
            continue
        candidates.append(
            TrainingCandidate(
                candidate_id=f"traincand_{trajectory_id}_{idx:04d}",
                trajectory_id=trajectory_id,
                roi_id=roi.roi_id,
                sample_type="supervised_coco_roi",
                target_model_role="main_model",
                failure_category=roi.level1_error_type,
                artifact_refs={"roi_id": roi.roi_id},
            )
        )
    return candidates
