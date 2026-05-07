from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from input_layer.mainline_profiles import get_mainline_capabilities, resolve_mainline_profile

from ITD_agent.memory_store.compact import compact_memory_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "ITD_agent" / "memory_store"
LEGACY_MEMORY_ROOT = PROJECT_ROOT / "ITD_agent" / "ITD_agent" / "memory_store"


def _memory_roots(memory_root: str | Path = DEFAULT_MEMORY_ROOT) -> list[Path]:
    root = Path(memory_root)
    roots = [root]
    if root == DEFAULT_MEMORY_ROOT and LEGACY_MEMORY_ROOT != DEFAULT_MEMORY_ROOT:
        roots.append(LEGACY_MEMORY_ROOT)
    return roots


def _records_roots(memory_root: str | Path = DEFAULT_MEMORY_ROOT) -> list[Path]:
    return [root / "records" for root in _memory_roots(memory_root)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _read_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for item in _read_jsonl(path):
            memory_id = str(item.get("memory_id") or "")
            dedupe_key = memory_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(item)
    return rows


def _tail(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows[-limit:] if limit > 0 else rows


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_tags(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, int, float)):
        text = str(values).strip()
        return {text} if text else set()
    if isinstance(values, dict):
        tags: set[str] = set()
        for value in values.values():
            tags.update(_normalize_tags(value))
        return tags
    if isinstance(values, (list, tuple, set)):
        tags: set[str] = set()
        for value in values:
            tags.update(_normalize_tags(value))
        return tags
    text = str(values).strip()
    return {text} if text else set()


def _collect_similarity_tags(profile: dict[str, Any] | None, item: dict[str, Any] | None = None) -> set[str]:
    source = profile or {}
    tags: set[str] = set()
    for key in [
        "tags",
        "stand_condition_labels",
        "texture_labels",
        "quality_labels",
        "terrain_labels",
        "knowledge_profile_types",
        "public_dataset_roles",
        "forest_type",
        "terrain_type",
    ]:
        tags.update(_normalize_tags(source.get(key)))
    if item:
        tags.update(_normalize_tags(item.get("tags")))
        tags.update(_normalize_tags(item.get("failure_categories")))
        tags.update(_normalize_tags(item.get("failure_category")))
    return tags


def infer_scene_profile_from_runtime(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    mainline_profile = resolve_mainline_profile(runtime_cfg)
    capabilities = runtime_cfg.get("_mainline_capabilities") or get_mainline_capabilities(mainline_profile)
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
    tags: list[str] = []
    for value in [forest_type, terrain_type, *stand_labels, *texture_labels, *quality_labels, *terrain_labels]:
        text = str(value or "").strip()
        if text and text not in tags:
            tags.append(text)
    return {
        "mainline_profile": mainline_profile,
        "input_modalities": (((runtime_cfg.get("_input_manifest") or {}).get("metadata") or {}).get("input_modalities") or {}),
        "forest_type": forest_type,
        "terrain_type": terrain_type,
        "image_resolution_m": ((image_profiles[0] or {}).get("resolution_x_m") if image_profiles else None),
        "knowledge_profile_types": sorted(
            {
                str(item.get("normalized_type"))
                for item in knowledge_profiles
                if item.get("normalized_type")
            }
        ) if allow_external_knowledge else [],
        "public_dataset_roles": sorted(
            {
                str(role)
                for item in public_datasets
                for role in (item.get("usage_roles") or [])
                if role
            }
        ) if allow_public_datasets else [],
        "tags": tags,
        "stand_condition_labels": stand_labels,
        "texture_labels": texture_labels,
        "image_texture_levels": image_texture_analysis.get("levels") or {},
        "quality_labels": quality_labels,
        "terrain_labels": terrain_labels,
        "image_quality_levels": image_quality_analysis.get("levels") or {},
    }


def load_recent_execution_traces(
    *,
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    limit: int = 10,
) -> list[dict[str, Any]]:
    paths = [root / "execution_trace.jsonl" for root in _records_roots(memory_root)]
    return [compact_memory_record(item) for item in _tail(_read_jsonl_many(paths), limit)]


def load_recent_success_strategies(
    *,
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    limit: int = 10,
) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for root in _records_roots(memory_root):
        paths.append(root / "successful_strategy.jsonl")
        paths.append(root / "successful_strategies.jsonl")
    rows = _read_jsonl_many(paths)
    return [compact_memory_record(item) for item in _tail(rows, limit)]


def load_recent_failure_patterns(
    *,
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    limit: int = 10,
) -> list[dict[str, Any]]:
    paths = [root / "failure_pattern.jsonl" for root in _records_roots(memory_root)]
    return [compact_memory_record(item) for item in _tail(_read_jsonl_many(paths), limit)]


def load_recent_run_retrospectives(
    *,
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    limit: int = 10,
) -> list[dict[str, Any]]:
    paths = [root / "run_retrospective.jsonl" for root in _records_roots(memory_root)]
    return [compact_memory_record(item) for item in _tail(_read_jsonl_many(paths), limit)]


def load_scene_similar_memories(
    *,
    scene_profile: dict[str, Any],
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not scene_profile:
        return []
    target_tags = _collect_similarity_tags(scene_profile)
    candidates = (
        load_recent_success_strategies(memory_root=memory_root, limit=100)
        + load_recent_failure_patterns(memory_root=memory_root, limit=100)
        + load_recent_execution_traces(memory_root=memory_root, limit=100)
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        profile = item.get("scene_profile") or {}
        score = 0.0
        if profile.get("forest_type") and profile.get("forest_type") == scene_profile.get("forest_type"):
            score += 2.0
        if profile.get("terrain_type") and profile.get("terrain_type") == scene_profile.get("terrain_type"):
            score += 1.5
        image_resolution = _safe_float(profile.get("image_resolution_m"))
        target_resolution = _safe_float(scene_profile.get("image_resolution_m"))
        if image_resolution is not None and target_resolution is not None:
            score += max(0.0, 1.0 - abs(image_resolution - target_resolution))
        source_tags = _collect_similarity_tags(profile, item)
        shared_tags = source_tags & target_tags
        if shared_tags:
            score += 0.75 * len(shared_tags)
            score += 0.75 * (len(shared_tags) / max(len(target_tags), 1))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]
