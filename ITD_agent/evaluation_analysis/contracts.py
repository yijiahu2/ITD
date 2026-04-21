from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class InputAssessment:
    readiness_score: float
    modality_status: dict[str, Any] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    scene_analysis: dict[str, Any] = field(default_factory=dict)
    terrain_summary: dict[str, Any] = field(default_factory=dict)
    data_processing_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceQualityResult:
    assessment_phase: str
    metrics_json: str | None = None
    details_csv: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    detail_summary: dict[str, Any] = field(default_factory=dict)
    quality_score: float | None = None
    terrain_error_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ROIAssessment:
    assessment_phase: str
    round_idx: int
    quality_label: str
    current_score: float | None = None
    previous_score: float | None = None
    improvement: float | None = None
    trigger_metrics: list[str] = field(default_factory=list)
    details_summary: dict[str, Any] = field(default_factory=dict)
    candidate_rois: list[dict[str, Any]] = field(default_factory=list)
    continue_refinement: bool = False
    decision_source: str = "heuristic"
    decision_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertModelAssessment:
    assessment_phase: str
    round_idx: int
    metrics_json: str | None = None
    details_csv: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    detail_summary: dict[str, Any] = field(default_factory=dict)
    current_score: float | None = None
    previous_score: float | None = None
    improvement: float | None = None
    roi_assessment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ChildModelAssessment = ExpertModelAssessment


@dataclass
class FinalAssessmentResult:
    evaluation_mode: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"evaluation_mode": self.evaluation_mode, **self.payload}


@dataclass
class FinetuneEffectAssessment:
    summary_json: str | None = None
    compare_csv: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
