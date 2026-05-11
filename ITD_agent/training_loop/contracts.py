from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingCandidate:
    candidate_id: str
    trajectory_id: str
    roi_id: str
    sample_type: str
    target_model_role: str
    failure_category: str
    quality_status: str = "pending_review"
    approved: bool = False
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    trigger_training: bool = False
    reason: str = "Adaptive inference only marks training candidates; training decisions stay in training_loop."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingTriggerContext:
    source_run_id: str
    source_review_asset_dir: str
    target_model_role: str
    target_model_id: str
    target_expert_family: str | None
    failure_category: str | None
    training_ready_sample_count: int
    weak_supervision_candidate_count: int
    replay_sample_count: int
    public_dataset_candidate_count: int
    dataset_bundle_path: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPlan:
    training_job_id: str
    training_mode: str
    target_model_role: str
    target_model_id: str
    algorithm_name: str
    target_expert_family: str | None
    failure_category: str | None
    source_config_path: str
    generated_config_path: str
    output_dir: str
    command: list[str]
    expected_checkpoint_glob: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunResult:
    training_job_id: str
    training_mode: str
    status: str
    returncode: int | None
    command: list[str]
    stdout_log: str
    stderr_log: str
    best_checkpoint_path: str | None
    training_metrics_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelVersionRecord:
    model_version_id: str
    model_id: str
    model_role: str
    algorithm_name: str
    checkpoint_path: str
    source_training_job_id: str
    status: str
    metrics_summary: dict[str, Any]
    replay_guard_summary: dict[str, Any]
    model_card_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
