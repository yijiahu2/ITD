from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.common.json_store import load_jsonl_many
from ITD_agent.common.scene_profile import scene_profile_from_runtime
from ITD_agent.common.values import safe_float
from ITD_agent.memory_store.compact import compact_memory_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "outputs" / "ITD_agent_runtime" / "memory_store"
SOURCE_LEGACY_MEMORY_ROOT = PROJECT_ROOT / "ITD_agent" / "memory_store"
LEGACY_MEMORY_ROOT = PROJECT_ROOT / "ITD_agent" / "ITD_agent" / "memory_store"


def _memory_roots(memory_root: str | Path = DEFAULT_MEMORY_ROOT) -> list[Path]:
    root = Path(memory_root)
    roots = [root]
    if root == DEFAULT_MEMORY_ROOT:
        for legacy_root in (SOURCE_LEGACY_MEMORY_ROOT, LEGACY_MEMORY_ROOT):
            if legacy_root != root:
                roots.append(legacy_root)
    return roots


def _records_roots(memory_root: str | Path = DEFAULT_MEMORY_ROOT) -> list[Path]:
    return [root / "records" for root in _memory_roots(memory_root)]


def _read_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    return load_jsonl_many(paths, dedupe_key=lambda item: str(item.get("memory_id") or ""))


def _tail(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows[-limit:] if limit > 0 else rows


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
    return scene_profile_from_runtime(runtime_cfg)


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
        image_resolution = safe_float(profile.get("image_resolution_m"))
        target_resolution = safe_float(scene_profile.get("image_resolution_m"))
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
