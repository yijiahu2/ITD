from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import load_structured
from ITD_agent.segmentation.model_registry.mmdet_specs import is_mmdet_algorithm


MASKDINO_ALGORITHMS = {"maskdino_official"}


def resolve_family_training_config(
    *,
    taxonomy_path: str | Path,
    target_expert_family: str | None,
    target_model_id: str,
    failure_category: str | None = None,
) -> dict[str, Any]:
    taxonomy = load_structured(taxonomy_path)
    families = list(taxonomy.get("expert_families") or [])
    family_id = str(target_expert_family or taxonomy.get("default_expert_family") or "cross_domain_generalist")
    family = next((item for item in families if str(item.get("family_id")) == family_id), None)
    if family is None:
        return {
            "available": False,
            "reason": "target_expert_family_not_found",
            "target_expert_family": family_id,
            "taxonomy_path": str(taxonomy_path),
        }

    templates = dict(family.get("template_candidates") or {})
    priority = [str(item) for item in family.get("algorithms_priority") or []]
    algorithm_name = _select_algorithm(target_model_id=target_model_id, templates=templates, priority=priority)
    source_config_path = str(templates.get(algorithm_name) or "")
    training_defaults = dict(family.get("training_defaults") or {})
    return {
        "available": bool(algorithm_name and source_config_path),
        "target_expert_family": family_id,
        "failure_category": failure_category,
        "algorithm_name": algorithm_name,
        "source_config_path": source_config_path,
        "template_candidates": templates,
        "algorithms_priority": priority,
        "training_defaults": training_defaults,
        "replay_ratio": float(training_defaults.get("replay_ratio") or 0.0),
        "hard_case_ratio": float(training_defaults.get("hard_case_ratio") or 0.0),
        "prior_axes": list(training_defaults.get("prior_axes") or []),
        "selection_rules": dict(family.get("selection_rules") or {}),
        "taxonomy_path": str(taxonomy_path),
        "training_entry_available": _training_entry_available(algorithm_name),
    }


def _select_algorithm(*, target_model_id: str, templates: dict[str, Any], priority: list[str]) -> str:
    requested = str(target_model_id or "").strip()
    if requested in templates:
        return requested
    for candidate in priority:
        if candidate in templates:
            return candidate
    return requested


def _training_entry_available(algorithm_name: str) -> bool:
    key = str(algorithm_name or "").strip().lower()
    return key in MASKDINO_ALGORITHMS or is_mmdet_algorithm(key)
