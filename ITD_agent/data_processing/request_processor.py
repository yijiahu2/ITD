from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.data_processing.artifact_store import write_json
from ITD_agent.data_processing.contracts import IntermediateArtifactRef, ProcessingTaskRequest


def build_default_processing_requests(
    *,
    runtime_cfg: dict[str, Any],
    image_profiles: list[dict[str, Any]],
    dem_profiles: list[dict[str, Any]],
) -> list[ProcessingTaskRequest]:
    requests: list[ProcessingTaskRequest] = []
    if image_profiles:
        plan = image_profiles[0].get("tile_plan") or {}
        requests.append(
            ProcessingTaskRequest(
                request_id="image_preparation_main",
                action="prepare_image_execution_layout",
                source_type="remote_sensing",
                source_id=image_profiles[0].get("source_id"),
                parameters={
                    "mode": plan.get("mode"),
                    "requires_geometry_crop": plan.get("requires_geometry_crop"),
                    "requires_sliding_window": plan.get("requires_sliding_window"),
                },
            )
        )
    if dem_profiles:
        requests.append(
            ProcessingTaskRequest(
                request_id="dem_alignment_main",
                action="align_dem_to_image",
                source_type="terrain_dem",
                source_id=dem_profiles[0].get("source_id"),
                parameters=dem_profiles[0].get("alignment_with_image") or {},
            )
        )
    requests.append(
        ProcessingTaskRequest(
            request_id="final_fusion",
            action="fuse_and_dedupe_instances",
            source_type="intermediate_result",
            parameters={
                "overlap_ratio_thr": runtime_cfg.get("fusion_overlap_ratio_thr", 0.6),
                "boundary_band_m": runtime_cfg.get("fusion_boundary_band_m", 1.5),
                "min_area_m2": runtime_cfg.get("fusion_min_area_m2", 6.0),
            },
        )
    )
    return requests


def persist_processing_requests(
    requests: list[ProcessingTaskRequest],
    requests_dir: str | Path,
) -> tuple[list[ProcessingTaskRequest], list[IntermediateArtifactRef]]:
    out_dir = Path(requests_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[IntermediateArtifactRef] = []
    for request in requests:
        out_path = out_dir / f"{request.request_id}.json"
        write_json(request.to_dict(), out_path)
        artifacts.append(
            IntermediateArtifactRef(
                artifact_id=f"request::{request.request_id}",
                artifact_type="processing_request",
                path=str(out_path),
                producer="data_processing.request_processor",
                description=request.action,
            )
        )
    return requests, artifacts
