from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ITD_agent.evolution.roi.roi_clusterer import ROICluster
from ITD_agent.planning.scheduler.expert_routing_policy import route_expert_model


@dataclass(frozen=True)
class ExpertTask:
    expert_task_id: str
    trajectory_id: str
    image_id: str
    expert_model: str
    failure_family: str
    level1_error_type: str
    roi_ids: list[str] = field(default_factory=list)
    tile_window_px: list[float] = field(default_factory=list)
    fusion_bboxes: dict[str, list[float]] = field(default_factory=dict)
    execution_mode: str = "mock"
    status: str = "pending"
    trigger_reason: dict[str, Any] = field(default_factory=dict)
    routing_event: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_expert_tasks(
    *,
    trajectory_id: str,
    roi_clusters: list[ROICluster],
    routing_policy: dict[str, Any] | None = None,
    execution_mode: str = "mock",
) -> list[ExpertTask]:
    tasks: list[ExpertTask] = []
    for idx, cluster in enumerate(roi_clusters, start=1):
        route = route_expert_model(cluster.level1_error_type, routing_policy)
        tasks.append(
            ExpertTask(
                expert_task_id=f"task_{trajectory_id}_{idx:04d}",
                trajectory_id=trajectory_id,
                image_id=cluster.image_id,
                expert_model=route["expert_model"],
                failure_family=cluster.failure_family,
                level1_error_type=cluster.level1_error_type,
                roi_ids=list(cluster.roi_ids),
                tile_window_px=list(cluster.tile_window_px),
                fusion_bboxes=dict(cluster.fusion_bboxes),
                execution_mode=execution_mode,
                trigger_reason={
                    "severity_score": cluster.severity_score,
                    "roi_count": len(cluster.roi_ids),
                    "failure_family": cluster.failure_family,
                },
                routing_event=route,
            )
        )
    return tasks
