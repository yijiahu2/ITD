from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ITD_agent.config_adapter import load_raw_yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy" / "expert_families.yaml"


def _normalize_tag(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_normalize_tag(item) for item in value if _normalize_tag(item)]
    if isinstance(value, str):
        return [_normalize_tag(item) for item in value.split(",") if _normalize_tag(item)]
    normalized = _normalize_tag(value)
    return [normalized] if normalized else []


@lru_cache(maxsize=4)
def load_expert_taxonomy(taxonomy_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(taxonomy_path or DEFAULT_TAXONOMY_PATH)
    if not path.exists():
        return {"taxonomy_version": 0, "default_expert_family": "cross_domain_generalist", "expert_families": []}
    payload = load_raw_yaml(path)
    families = payload.get("expert_families") or []
    by_id = {
        _normalize_tag(item.get("family_id")): item
        for item in families
        if isinstance(item, dict) and item.get("family_id")
    }
    payload["expert_families_by_id"] = by_id
    payload["default_expert_family"] = _normalize_tag(payload.get("default_expert_family") or "cross_domain_generalist")
    return payload


def get_expert_family_definition(family_id: str | None, taxonomy_path: str | Path | None = None) -> dict[str, Any]:
    taxonomy = load_expert_taxonomy(taxonomy_path)
    key = _normalize_tag(family_id) or taxonomy.get("default_expert_family") or "cross_domain_generalist"
    return dict((taxonomy.get("expert_families_by_id") or {}).get(key) or {})


def infer_expert_family_from_entry(entry: dict[str, Any], taxonomy_path: str | Path | None = None) -> str:
    explicit = _normalize_tag(entry.get("expert_family") or entry.get("expert_family_id"))
    taxonomy = load_expert_taxonomy(taxonomy_path)
    if explicit and explicit in (taxonomy.get("expert_families_by_id") or {}):
        return explicit

    algorithm = _normalize_tag(entry.get("algorithm"))
    scene_tags = set(_normalize_list(entry.get("scene_tags") or entry.get("scene_labels")))
    terrain_tags = set(_normalize_list(entry.get("terrain_tags")))
    failure_categories = set(_normalize_list(entry.get("failure_categories")))
    error_patterns = set(_normalize_list(entry.get("target_error_patterns")))

    best_family = str(taxonomy.get("default_expert_family") or "cross_domain_generalist")
    best_score = -1
    for family in taxonomy.get("expert_families") or []:
        if not isinstance(family, dict):
            continue
        family_id = _normalize_tag(family.get("family_id"))
        score = 0
        if algorithm and algorithm in {_normalize_tag(item) for item in family.get("algorithms_priority") or []}:
            score += 8
        rules = family.get("selection_rules") or {}
        score += 4 * len(scene_tags & set(_normalize_list(rules.get("scene_tags"))))
        score += 5 * len(terrain_tags & set(_normalize_list(rules.get("terrain_tags"))))
        score += 8 * len(failure_categories & set(_normalize_list(rules.get("failure_categories"))))
        score += 4 * len(error_patterns & set(_normalize_list(rules.get("error_patterns"))))
        if score > best_score:
            best_score = score
            best_family = family_id or best_family
    return best_family


def resolve_expert_template_path(
    expert_family: str | None,
    algorithm_name: str | None,
    taxonomy_path: str | Path | None = None,
) -> str | None:
    family = get_expert_family_definition(expert_family, taxonomy_path)
    if not family:
        return None
    candidates = family.get("template_candidates") or {}
    if algorithm_name:
        path = candidates.get(str(algorithm_name))
        if path:
            return str(path)
    for algorithm in family.get("algorithms_priority") or []:
        path = candidates.get(str(algorithm))
        if path:
            return str(path)
    return None


def build_expert_training_defaults(
    expert_family: str | None,
    algorithm_name: str | None,
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    family = get_expert_family_definition(expert_family, taxonomy_path)
    defaults = dict(family.get("training_defaults") or {})
    defaults["target_expert_family"] = _normalize_tag(expert_family) or _normalize_tag(family.get("family_id"))
    defaults["segmentation_algorithm"] = str(algorithm_name or "")
    defaults["template_config_path"] = resolve_expert_template_path(expert_family, algorithm_name, taxonomy_path)
    return defaults


def infer_target_expert_families(
    *,
    metadata: dict[str, Any] | None,
    root_path: str | None = None,
    source_id: str | None = None,
    usage_roles: list[str] | None = None,
    taxonomy_path: str | Path | None = None,
) -> list[str]:
    metadata = metadata or {}
    for key in ["target_expert_families", "supported_expert_families", "expert_families", "expert_roles"]:
        values = _normalize_list(metadata.get(key))
        if values:
            return values

    corpus = " ".join(
        [
            str(source_id or ""),
            str(root_path or ""),
            str(metadata.get("dataset_name") or ""),
            str(metadata.get("description") or ""),
            " ".join(str(item) for item in (usage_roles or [])),
        ]
    ).lower()
    if "isprs" in corpus:
        return ["cross_domain_generalist", "large_crown_over_split", "boundary_calibration"]
    if "oam" in corpus or "tcd" in corpus:
        return ["cross_domain_generalist", "dense_adhesion", "shadow_topography"]
    taxonomy = load_expert_taxonomy(taxonomy_path)
    return [str(taxonomy.get("default_expert_family") or "cross_domain_generalist")]


def infer_domain_tags(
    *,
    metadata: dict[str, Any] | None,
    root_path: str | None = None,
    source_id: str | None = None,
) -> list[str]:
    metadata = metadata or {}
    tags = _normalize_list(metadata.get("domain_tags") or metadata.get("tags"))
    if tags:
        return tags
    corpus = " ".join([str(source_id or ""), str(root_path or ""), str(metadata.get("dataset_name") or "")]).lower()
    inferred: list[str] = []
    if "isprs" in corpus:
        inferred.extend(["benchmark", "cross_domain", "high_resolution"])
    if "oam" in corpus or "tcd" in corpus:
        inferred.extend(["cross_biome", "orthomosaic", "high_resolution"])
    if not inferred:
        inferred.append("generic_dataset")
    return sorted(dict.fromkeys(inferred))
