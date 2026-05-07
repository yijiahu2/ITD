from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.contracts import InputManifest
from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile

from ITD_agent.context_engine import build_online_scene_state
from ITD_agent.data_processing.artifact_store import ensure_data_processing_dirs, write_json
from ITD_agent.data_processing.contracts import DataProcessingSummary
from ITD_agent.data_processing.height import build_height_raster_profiles
from ITD_agent.data_processing.request_processor import build_default_processing_requests, persist_processing_requests
from ITD_agent.data_processing.remote_sensing.profiles import build_image_profiles, build_remote_sensing_preflight
from ITD_agent.data_processing.terrain.dem_pipeline import build_dem_profiles


def summarize_data_processing_stage(
    *,
    runtime_cfg: dict[str, Any],
    input_manifest: InputManifest,
    terrain_info: dict[str, Any],
) -> DataProcessingSummary:
    mainline_profile = resolve_mainline_profile(runtime_cfg)
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(mainline_profile)
    storage_layout = ensure_data_processing_dirs(runtime_cfg)
    image_profiles = build_image_profiles(input_manifest, runtime_cfg)
    remote_sensing_preflight = build_remote_sensing_preflight(input_manifest, runtime_cfg, storage_layout, image_profiles)
    dem_profiles = build_dem_profiles(input_manifest, image_profiles, terrain_info) if capabilities.get("allow_dem") else []
    height_profiles = (
        build_height_raster_profiles(input_manifest, image_profiles, storage_layout)
        if capabilities.get("allow_chm") or capabilities.get("allow_dsm")
        else []
    )
    request_objs = build_default_processing_requests(
        runtime_cfg=runtime_cfg,
        image_profiles=[item.to_dict() for item in image_profiles],
        dem_profiles=[item.to_dict() for item in dem_profiles],
    )
    _, request_artifacts = persist_processing_requests(request_objs, storage_layout["requests"])

    summary = DataProcessingSummary(
        image_profiles=image_profiles,
        dem_profiles=dem_profiles,
        height_raster_profiles=height_profiles,
        requested_tasks=request_objs,
        intermediate_artifacts=request_artifacts,
        storage_layout=storage_layout,
        metadata={
            "mainline_profile": mainline_profile,
            "mainline_capabilities": capabilities,
            "image_count": len(image_profiles),
            "dem_count": len(dem_profiles),
            "height_raster_count": len(height_profiles),
            "survey_table_count": len(input_manifest.survey_tables) if capabilities.get("allow_inventory") else 0,
            "industry_vector_count": len(input_manifest.industry_vectors) if capabilities.get("allow_inventory") else 0,
            "knowledge_count": len(input_manifest.domain_knowledge_items) if capabilities.get("allow_domain_knowledge") else 0,
            "public_dataset_count": len(input_manifest.public_datasets) if capabilities.get("allow_public_datasets") else 0,
            "remote_sensing_preflight": remote_sensing_preflight.to_dict() if remote_sensing_preflight else {},
            "input_manifest_summary": {
                "survey_tables": [item.to_dict() for item in input_manifest.survey_tables] if capabilities.get("allow_inventory") else [],
                "industry_vectors": [item.to_dict() for item in input_manifest.industry_vectors] if capabilities.get("allow_inventory") else [],
                "domain_knowledge_items": [item.to_dict() for item in input_manifest.domain_knowledge_items] if capabilities.get("allow_domain_knowledge") else [],
                "public_datasets": [item.to_dict() for item in input_manifest.public_datasets] if capabilities.get("allow_public_datasets") else [],
            },
        },
    )
    summary_path = Path(storage_layout["summaries"]) / "data_processing_summary.json"
    online_scene_state = build_online_scene_state(
        runtime_cfg=runtime_cfg,
        input_manifest=input_manifest,
        terrain_info=terrain_info,
        data_processing_summary=summary.to_dict(),
    )
    online_scene_state_path = Path(storage_layout["summaries"]) / "online_scene_state.json"
    summary.metadata["summary_json"] = str(summary_path)
    summary.metadata["online_scene_state_json"] = str(online_scene_state_path)
    summary.metadata["online_scene_state"] = online_scene_state
    write_json(online_scene_state, online_scene_state_path)
    write_json(summary.to_dict(), summary_path)
    return summary
