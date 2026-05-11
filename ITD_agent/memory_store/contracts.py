from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ITD_agent.common.serialization import DataclassDictMixin


@dataclass(frozen=True)
class ExecutionTraceMemory(DataclassDictMixin):
    memory_id: str
    memory_type: str
    timestamp: str
    run_name: str
    mode: str
    scene_profile: dict[str, Any] = field(default_factory=dict)
    input_profile: dict[str, Any] = field(default_factory=dict)
    planning_summary: dict[str, Any] = field(default_factory=dict)
    segmentation_summary: dict[str, Any] = field(default_factory=dict)
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "orchestrator"



@dataclass(frozen=True)
class SuccessfulStrategyMemory(DataclassDictMixin):
    memory_id: str
    memory_type: str
    timestamp: str
    run_name: str
    scene_profile: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    strategy_summary: dict[str, Any] = field(default_factory=dict)
    llm_success_strategies: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "orchestrator"



@dataclass(frozen=True)
class FailurePatternMemory(DataclassDictMixin):
    memory_id: str
    memory_type: str
    timestamp: str
    run_name: str
    scene_profile: dict[str, Any] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    trigger_mode: str | None = None
    recommended_actions: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "orchestrator"



@dataclass(frozen=True)
class RunRetrospectiveMemory(DataclassDictMixin):
    memory_id: str
    memory_type: str
    timestamp: str
    run_name: str
    scene_profile: dict[str, Any] = field(default_factory=dict)
    llm_gateway_result: dict[str, Any] = field(default_factory=dict)
    parsed_result: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "llm_gateway"
