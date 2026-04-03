from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.artifact_store import ensure_data_processing_dirs, write_json
from ITD_agent.data_processing.contracts import DataProcessingSummary
from ITD_agent.data_processing.imagery.priors import build_image_profiles
from ITD_agent.data_processing.inventory.normalizer import build_industry_vector_profiles, build_survey_table_profiles
from ITD_agent.data_processing.knowledge.normalizer import build_knowledge_profiles
from ITD_agent.data_processing.public_data.indexer import build_public_dataset_profiles
from ITD_agent.data_processing.request_processor import build_default_processing_requests, persist_processing_requests
from ITD_agent.data_processing.terrain.dem_pipeline import build_dem_profiles


def summarize_data_processing_stage(
    *,
    runtime_cfg: dict[str, Any],
    input_manifest: InputManifest,
    terrain_info: dict[str, Any],
) -> DataProcessingSummary:
    storage_layout = ensure_data_processing_dirs(runtime_cfg)
    image_profiles = build_image_profiles(input_manifest, runtime_cfg)
    dem_profiles = build_dem_profiles(input_manifest, image_profiles, terrain_info)
    survey_profiles = build_survey_table_profiles(input_manifest)
    vector_profiles = build_industry_vector_profiles(input_manifest)
    knowledge_profiles = build_knowledge_profiles(input_manifest)
    public_dataset_profiles = build_public_dataset_profiles(input_manifest)

    request_objs = build_default_processing_requests(
        runtime_cfg=runtime_cfg,
        image_profiles=[item.to_dict() for item in image_profiles],
        dem_profiles=[item.to_dict() for item in dem_profiles],
    )
    _, request_artifacts = persist_processing_requests(request_objs, storage_layout["requests"])

    summary = DataProcessingSummary(
        image_profiles=image_profiles,
        dem_profiles=dem_profiles,
        survey_table_profiles=survey_profiles,
        industry_vector_profiles=vector_profiles,
        knowledge_profiles=knowledge_profiles,
        public_dataset_profiles=public_dataset_profiles,
        requested_tasks=request_objs,
        intermediate_artifacts=request_artifacts,
        storage_layout=storage_layout,
        metadata={
            "image_count": len(image_profiles),
            "dem_count": len(dem_profiles),
            "survey_table_count": len(survey_profiles),
            "industry_vector_count": len(vector_profiles),
            "knowledge_count": len(knowledge_profiles),
            "public_dataset_count": len(public_dataset_profiles),
        },
    )
    summary_path = Path(storage_layout["summaries"]) / "data_processing_summary.json"
    summary.metadata["summary_json"] = str(summary_path)
    write_json(summary.to_dict(), summary_path)
    return summary
