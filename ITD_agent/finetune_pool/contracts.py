from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ITD_agent.model_roles import normalize_model_role


@dataclass
class FinetunePoolSample:
    sample_id: str
    run_name: str
    timestamp: str
    source_type: str
    target_module: str
    target_model_role: str
    target_expert_family: str | None = None
    failure_category: str | None = None
    scene_profile: dict[str, Any] = field(default_factory=dict)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    label_status: str = "weak"
    ready_for_training: bool = False
    tags: list[str] = field(default_factory=list)
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_model_role = normalize_model_role(self.target_model_role)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicDatasetCandidate:
    candidate_id: str
    dataset_id: str
    dataset_name: str
    target_model_role: str
    target_expert_family: str | None = None
    supported_failure_categories: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    terrain_tags: list[str] = field(default_factory=list)
    forest_type: str | None = None
    sensor_type: str | None = None
    resolution_range: str | None = None
    annotation_type: str | None = None
    label_quality: str | None = None
    data_volume: int | None = None
    index_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_model_role = normalize_model_role(self.target_model_role)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinetunePoolCluster:
    cluster_id: str
    target_model_role: str
    failure_category: str
    target_expert_family: str | None = None
    source_types: list[str] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    sample_count: int = 0
    ready_sample_count: int = 0
    label_status_breakdown: dict[str, int] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    scene_profiles: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_model_role = normalize_model_role(self.target_model_role)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinetuneTriggerSnapshot:
    timestamp: str
    run_name: str
    trigger_ready: bool
    recommended_target_module: str | None = None
    recommended_target_model_role: str | None = None
    recommended_target_expert_family: str | None = None
    recommended_failure_category: str | None = None
    trigger_reason: str | None = None
    sample_counts: dict[str, int] = field(default_factory=dict)
    ready_counts: dict[str, int] = field(default_factory=dict)
    cluster_summaries: list[dict[str, Any]] = field(default_factory=list)
    public_dataset_candidates: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recommended_target_model_role is not None:
            self.recommended_target_model_role = normalize_model_role(self.recommended_target_model_role)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
