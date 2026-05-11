from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from output_layer.contracts import FinalDeliverables


__all__ = ["ExecutionPlan", "FinalDeliverables"]


@dataclass
class ExecutionPlan:
    mode: str
    run_name: str
    stage_flags: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
