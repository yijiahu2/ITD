from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlanningRequest:
    planning_stage: str
    template_path: str
    output_path: str
    runtime_cfg: dict[str, Any] = field(default_factory=dict)
    metrics_json: str | None = None
    details_csv: str | None = None
    summary_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningDecision:
    enabled: bool
    use_llm: bool
    planning_stage: str
    template_path: str
    generated_config_path: str
    parameter_updates: dict[str, Any] = field(default_factory=dict)
    llm_result: dict[str, Any] | None = None
    llm_gateway_result: dict[str, Any] | None = None
    scheduler_context: dict[str, Any] = field(default_factory=dict)
    effective_runtime_cfg: dict[str, Any] = field(default_factory=dict)
    runtime_plan: dict[str, Any] = field(default_factory=dict)
    roi_refine_plan: dict[str, Any] = field(default_factory=dict)
    child_model_call_plan: dict[str, Any] = field(default_factory=dict)
    finetune_training_plan: dict[str, Any] = field(default_factory=dict)
    knowledge_embedding_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ROIRefinePlan:
    enabled: bool
    use_llm: bool
    max_rounds: int
    top_k: int
    buffer_m: float
    strategy_mode: str
    preferred_child_model: str | None = None
    candidate_child_models: list[str] = field(default_factory=list)
    selection_rules: list[str] = field(default_factory=list)
    stop_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChildModelCallPlan:
    enabled: bool
    planning_stage: str
    routing_mode: str
    preferred_child_model: str | None = None
    candidate_models: list[str] = field(default_factory=list)
    routing_rules: list[str] = field(default_factory=list)
    escalation_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeEmbeddingPlan:
    enabled: bool
    knowledge_sources: list[str] = field(default_factory=list)
    backbone_rules: list[dict[str, Any]] = field(default_factory=list)
    neck_rules: list[dict[str, Any]] = field(default_factory=list)
    initial_prediction_rules: list[dict[str, Any]] = field(default_factory=list)
    head_rules: list[dict[str, Any]] = field(default_factory=list)
    config_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinetuneTrainingPlan:
    should_prepare: bool
    target_module: str
    trigger_mode: str
    template_config_path: str | None = None
    generated_config_path: str | None = None
    train_mode: str = "head_only"
    freeze_backbone: bool = False
    epochs: int = 4
    batch_size: int = 1
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    data_selection_rule: str | None = None
    supervision_mode: str = "hybrid"
    dataset_bundle_path: str | None = None
    dataset_selection_summary: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
