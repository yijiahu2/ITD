from __future__ import annotations

from typing import Any

from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile


def _unique_text(values: list[Any]) -> list[str]:
    tags: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in tags:
            tags.append(text)
    return tags


def _profile_capabilities(source: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mainline_profile = source.get("mainline_profile") or resolve_mainline_profile(source)
    capabilities = source.get("_mainline_capabilities") or get_mainline_capabilities(mainline_profile)
    return mainline_profile, capabilities


def extract_input_profile(input_manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = input_manifest.get("metadata") or {}
    mainline_profile = metadata.get("mainline_profile") or resolve_mainline_profile(input_manifest)
    capabilities = metadata.get("mainline_capabilities") or get_mainline_capabilities(mainline_profile)
    remote_sensing = input_manifest.get("remote_sensing")
    terrain = input_manifest.get("terrain")
    remote_sensing_count = (
        len((remote_sensing or {}).get("images") or [])
        if isinstance(remote_sensing, dict)
        else len(input_manifest.get("remote_sensing_images") or [])
    )
    dem_count = (
        len((terrain or {}).get("dem") or [])
        if isinstance(terrain, dict)
        else len(input_manifest.get("dem_items") or [])
    )
    return {
        "mainline_profile": mainline_profile,
        "mainline_capabilities": capabilities,
        "input_modalities": metadata.get("input_modalities") or {},
        "remote_sensing_count": remote_sensing_count,
        "dem_count": dem_count,
        "survey_table_count": len(input_manifest.get("survey_tables") or []),
        "industry_vector_count": len(input_manifest.get("industry_vectors") or []),
        "knowledge_count": len(input_manifest.get("domain_knowledge_items") or []),
        "public_dataset_count": len(input_manifest.get("public_datasets") or []),
    }


def scene_profile_from_summary(summary: dict[str, Any], input_manifest: dict[str, Any]) -> dict[str, Any]:
    input_profile = extract_input_profile(input_manifest)
    mainline_profile = input_profile.get("mainline_profile")
    capabilities = input_profile.get("mainline_capabilities") or {}
    allow_external_knowledge = bool(capabilities.get("allow_external_knowledge"))
    allow_public_datasets = bool(capabilities.get("allow_public_datasets"))
    run_meta = summary.get("run_meta") or {}
    terrain_info = run_meta.get("terrain_info") or {}
    input_assessment = (
        ((summary.get("data_processing") or {}).get("input_assessment") or {})
        or ((summary.get("evaluation_analysis") or {}).get("input_assessment") or {})
    )
    scene_analysis = input_assessment.get("scene_analysis") or {}
    terrain_analysis = scene_analysis.get("terrain_analysis") or {}
    image_texture_analysis = scene_analysis.get("image_texture_analysis") or {}
    image_quality_analysis = scene_analysis.get("image_quality_analysis") or {}
    processing_summary = ((summary.get("data_processing") or {}).get("processing_summary") or {})
    image_profiles = processing_summary.get("image_profiles") or []
    manifest_summary = ((processing_summary.get("metadata") or {}).get("input_manifest_summary") or {})
    knowledge_profiles = manifest_summary.get("domain_knowledge_items") or []
    public_dataset_profiles = manifest_summary.get("public_datasets") or []
    image_resolution = None
    if image_profiles:
        image_resolution = image_profiles[0].get("resolution_x_m") or image_profiles[0].get("resolution_y_m")
    forest_type = run_meta.get("forest_type") or scene_analysis.get("forest_type")
    terrain_type = (
        terrain_info.get("landform_type")
        or run_meta.get("terrain_type")
        or (terrain_analysis.get("dom_context") or {}).get("landform_type")
        or (terrain_analysis.get("global_background") or {}).get("landform_type")
    ) if capabilities.get("allow_dem") else None
    stand_labels = ((scene_analysis.get("stand_condition") or {}).get("labels") or [])
    texture_labels = image_texture_analysis.get("labels") or []
    quality_labels = image_quality_analysis.get("labels") or []
    terrain_labels = (terrain_analysis.get("labels") or []) if capabilities.get("allow_dem") else []
    knowledge_profile_types = sorted(
        {
            str(item.get("normalized_type"))
            for item in knowledge_profiles
            if item.get("normalized_type")
        }
    ) if allow_external_knowledge else []
    public_dataset_roles = sorted(
        {
            str(role)
            for item in public_dataset_profiles
            for role in (item.get("usage_roles") or [])
            if role
        }
    ) if allow_public_datasets else []
    return {
        "mainline_profile": mainline_profile,
        "input_modalities": input_profile.get("input_modalities") or {},
        "forest_type": forest_type,
        "terrain_type": terrain_type,
        "image_resolution_m": image_resolution,
        "knowledge_profile_types": knowledge_profile_types,
        "public_dataset_roles": public_dataset_roles,
        "tags": _unique_text(
            [
                forest_type,
                terrain_type,
                *stand_labels,
                *texture_labels,
                *quality_labels,
                *terrain_labels,
                *knowledge_profile_types,
            ]
        ),
        "stand_condition_labels": stand_labels,
        "texture_labels": texture_labels,
        "terrain_labels": terrain_labels,
        "image_texture_levels": image_texture_analysis.get("levels") or {},
        "quality_labels": quality_labels,
        "image_quality_levels": image_quality_analysis.get("levels") or {},
    }


def scene_profile_from_runtime(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    mainline_profile, capabilities = _profile_capabilities(runtime_cfg)
    allow_external_knowledge = bool(capabilities.get("allow_external_knowledge"))
    allow_public_datasets = bool(capabilities.get("allow_public_datasets"))
    data_processing_summary = runtime_cfg.get("_data_processing_summary") or {}
    input_assessment = runtime_cfg.get("_input_assessment") or {}
    scene_analysis = input_assessment.get("scene_analysis") or {}
    image_texture_analysis = scene_analysis.get("image_texture_analysis") or {}
    image_quality_analysis = scene_analysis.get("image_quality_analysis") or {}
    terrain_analysis = scene_analysis.get("terrain_analysis") or {}
    image_profiles = data_processing_summary.get("image_profiles") or []
    manifest_summary = ((data_processing_summary.get("metadata") or {}).get("input_manifest_summary") or {})
    knowledge_profiles = manifest_summary.get("domain_knowledge_items") or []
    public_datasets = manifest_summary.get("public_datasets") or []
    terrain_info = runtime_cfg.get("terrain_info") or {}
    forest_type = runtime_cfg.get("forest_type") or scene_analysis.get("forest_type")
    stand_labels = ((scene_analysis.get("stand_condition") or {}).get("labels") or [])
    texture_labels = image_texture_analysis.get("labels") or []
    quality_labels = image_quality_analysis.get("labels") or []
    terrain_labels = (terrain_analysis.get("labels") or []) if capabilities.get("allow_dem") else []
    terrain_type = (
        (terrain_analysis.get("dom_context") or {}).get("landform_type")
        or (terrain_analysis.get("global_background") or {}).get("landform_type")
        or terrain_info.get("landform_type")
        or runtime_cfg.get("terrain_type")
    ) if capabilities.get("allow_dem") else None
    knowledge_profile_types = sorted(
        {
            str(item.get("normalized_type"))
            for item in knowledge_profiles
            if item.get("normalized_type")
        }
    ) if allow_external_knowledge else []
    public_dataset_roles = sorted(
        {
            str(role)
            for item in public_datasets
            for role in (item.get("usage_roles") or [])
            if role
        }
    ) if allow_public_datasets else []
    return {
        "mainline_profile": mainline_profile,
        "input_modalities": (((runtime_cfg.get("_input_manifest") or {}).get("metadata") or {}).get("input_modalities") or {}),
        "forest_type": forest_type,
        "terrain_type": terrain_type,
        "image_resolution_m": ((image_profiles[0] or {}).get("resolution_x_m") if image_profiles else None),
        "knowledge_profile_types": knowledge_profile_types,
        "public_dataset_roles": public_dataset_roles,
        "tags": _unique_text([forest_type, terrain_type, *stand_labels, *texture_labels, *quality_labels, *terrain_labels]),
        "stand_condition_labels": stand_labels,
        "texture_labels": texture_labels,
        "image_texture_levels": image_texture_analysis.get("levels") or {},
        "quality_labels": quality_labels,
        "terrain_labels": terrain_labels,
        "image_quality_levels": image_quality_analysis.get("levels") or {},
    }
