from __future__ import annotations

from copy import deepcopy
from typing import Any


DOM_IMAGE_PROFILE = "dom_image"
COCO_DATASET_PROFILE = "coco_dataset"
DEFAULT_MAINLINE_PROFILE = DOM_IMAGE_PROFILE


_PROFILE_CAPABILITIES: dict[str, dict[str, Any]] = {
    DOM_IMAGE_PROFILE: {
        "online_inputs": ["DOM"],
        "allow_dem": False,
        "allow_chm": False,
        "allow_dsm": False,
        "allow_external_knowledge": False,
        "allow_inventory": False,
        "allow_domain_knowledge": False,
        "allow_public_datasets": True,
        "allow_coco_dataset": True,
        "allow_memory_context": True,
        "allow_finetune_pool_context": True,
        "output_height_structure": False,
    },
    COCO_DATASET_PROFILE: {
        "online_inputs": [],
        "allow_dem": False,
        "allow_chm": False,
        "allow_dsm": False,
        "allow_external_knowledge": False,
        "allow_inventory": False,
        "allow_domain_knowledge": False,
        "allow_public_datasets": True,
        "allow_coco_dataset": True,
        "allow_memory_context": True,
        "allow_finetune_pool_context": True,
        "output_height_structure": False,
    },
}


def normalize_mainline_profile(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_MAINLINE_PROFILE
    aliases = {
        "DOM": DOM_IMAGE_PROFILE,
        "DOM_IMAGE": DOM_IMAGE_PROFILE,
        "DOM_ONLY": DOM_IMAGE_PROFILE,
        "COCO": COCO_DATASET_PROFILE,
        "COCO_DATASET": COCO_DATASET_PROFILE,
        "PUBLIC_COCO": COCO_DATASET_PROFILE,
    }
    upper = text.upper().replace("-", "_").replace(" ", "_")
    return aliases.get(upper, upper if upper in _PROFILE_CAPABILITIES else DEFAULT_MAINLINE_PROFILE)


def get_mainline_capabilities(profile: Any) -> dict[str, Any]:
    normalized = normalize_mainline_profile(profile)
    capabilities = deepcopy(_PROFILE_CAPABILITIES[normalized])
    capabilities["mainline_profile"] = normalized
    return capabilities


def _dict_has_any(mapping: Any, keys: tuple[str, ...]) -> bool:
    if not isinstance(mapping, dict):
        return False
    return any(bool(mapping.get(key)) for key in keys)


def resolve_mainline_profile(cfg: dict[str, Any] | None) -> str:
    if not isinstance(cfg, dict):
        return DEFAULT_MAINLINE_PROFILE
    runtime = cfg.get("runtime") or {}
    inputs = cfg.get("inputs") or {}
    value = (
        runtime.get("mainline_profile")
        or runtime.get("input_profile")
        or cfg.get("mainline_profile")
        or cfg.get("input_profile")
        or inputs.get("mainline_profile")
        or inputs.get("input_profile")
    )
    if value is not None and str(value).strip():
        return normalize_mainline_profile(value)

    terrain = inputs.get("terrain") or {}
    canopy = inputs.get("canopy") or {}
    surface = inputs.get("surface") or {}
    survey_data = inputs.get("survey_data") or {}
    industry_vectors = inputs.get("industry_vectors") or {}
    domain_knowledge = inputs.get("domain_knowledge") or inputs.get("knowledge") or {}
    has_b_inputs = any(
        [
            _dict_has_any(terrain, ("dem", "dem_tif")),
            _dict_has_any(canopy, ("chm", "chm_tif")),
            _dict_has_any(surface, ("dsm", "dsm_tif")),
            _dict_has_any(survey_data, ("tables", "survey_vector")),
            _dict_has_any(industry_vectors, ("vectors", "default_vector")),
            _dict_has_any(domain_knowledge, ("items", "sources")),
            bool(
                cfg.get("dem_tif")
                or cfg.get("chm_tif")
                or cfg.get("dsm_tif")
                or cfg.get("reference_vector_path")
                or cfg.get("inventory_vector_path")
                or cfg.get("xiaoban_shp")
                or cfg.get("domain_knowledge")
            ),
        ]
    )
    if (cfg.get("input_type") == "coco_dataset") or bool((inputs.get("public_datasets") or {}).get("datasets")):
        return COCO_DATASET_PROFILE
    return COCO_DATASET_PROFILE if has_b_inputs else DEFAULT_MAINLINE_PROFILE


def profile_allows_external_knowledge(runtime_cfg: dict[str, Any] | None) -> bool:
    if not isinstance(runtime_cfg, dict):
        return False
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(resolve_mainline_profile(runtime_cfg))
    return bool(capabilities.get("allow_external_knowledge"))


def filter_manifest_sources_for_profile(manifest: Any, profile: Any) -> dict[str, Any]:
    capabilities = get_mainline_capabilities(profile)
    ignored_counts: dict[str, int] = {}

    def clear_if_disallowed(attr: str, allowed_key: str) -> None:
        if capabilities.get(allowed_key):
            return
        values = getattr(manifest, attr, []) or []
        ignored_counts[attr] = len(values)
        setattr(manifest, attr, [])

    clear_if_disallowed("terrain_dem", "allow_dem")
    clear_if_disallowed("canopy_height", "allow_chm")
    clear_if_disallowed("surface_models", "allow_dsm")
    clear_if_disallowed("survey_tables", "allow_inventory")
    clear_if_disallowed("industry_vectors", "allow_inventory")
    clear_if_disallowed("domain_knowledge_items", "allow_domain_knowledge")
    clear_if_disallowed("public_datasets", "allow_public_datasets")

    return {
        "mainline_profile": capabilities["mainline_profile"],
        "capabilities": capabilities,
        "ignored_counts": {key: value for key, value in ignored_counts.items() if value > 0},
    }


def gate_data_processing_summary_for_profile(summary: dict[str, Any], runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(resolve_mainline_profile(runtime_cfg))
    gated = deepcopy(summary or {})
    if not capabilities.get("allow_dem"):
        gated["dem_profiles"] = []
    if not capabilities.get("allow_chm") and not capabilities.get("allow_dsm"):
        gated["height_raster_profiles"] = []
    metadata = dict(gated.get("metadata") or {})
    manifest_summary = dict(metadata.get("input_manifest_summary") or {})
    if not capabilities.get("allow_inventory"):
        manifest_summary["survey_tables"] = []
        manifest_summary["industry_vectors"] = []
    if not capabilities.get("allow_domain_knowledge"):
        manifest_summary["domain_knowledge_items"] = []
    if not capabilities.get("allow_public_datasets"):
        manifest_summary["public_datasets"] = []
    metadata["input_manifest_summary"] = manifest_summary
    metadata["mainline_profile"] = capabilities["mainline_profile"]
    metadata["mainline_capabilities"] = capabilities
    gated["metadata"] = metadata
    return gated
