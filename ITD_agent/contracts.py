from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExecutionPlan:
    mode: str
    run_name: str
    stage_flags: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalDeliverables:
    publish_root: str
    tree_crowns_shp: str | None = None
    tree_points_shp: str | None = None
    semantic_prior_tif: str | None = None
    semantic_prior_png: str | None = None
    segmentation_visualization_png: str | None = None
    final_evaluation_report_md: str | None = None
    final_evaluation_report_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
