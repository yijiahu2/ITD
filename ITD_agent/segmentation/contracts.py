from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SegmentationExecutionRequest:
    phase: str
    model_role: str
    algorithm_name: str
    config_path: str | None = None
    runtime_cfg: dict[str, Any] = field(default_factory=dict)
    plan_summary: dict[str, Any] = field(default_factory=dict)
    required_inputs: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationExecutionResult:
    phase: str
    model_role: str
    algorithm_name: str
    status: str
    output_paths: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationFinetuneRequest:
    target_module: str
    algorithm_name: str | None = None
    config_path: str | None = None
    training_plan: dict[str, Any] = field(default_factory=dict)
    dataset_context: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationFinetuneResult:
    target_module: str
    status: str
    generated_config: str | None = None
    best_checkpoint: str | None = None
    summary_json: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
