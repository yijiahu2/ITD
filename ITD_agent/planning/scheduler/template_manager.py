from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ITD_agent.config_adapter import load_raw_yaml, save_runtime_config

CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"


def load_config_template(template_path: str | Path) -> dict[str, Any]:
    return load_raw_yaml(template_path)


def resolve_template_metadata(template_path: str | Path) -> dict[str, str]:
    path = Path(template_path).resolve()
    default_category = "adhoc"
    default_name = path.stem
    try:
        rel = path.relative_to(CONFIGS_ROOT.resolve())
    except Exception:
        return {
            "template_path": str(path),
            "template_category": default_category,
            "template_group": default_category,
            "template_name": default_name,
            "template_relative_path": path.name,
        }

    parts = rel.parts
    category = default_category
    group = default_category
    if len(parts) >= 3 and parts[0] == "templates":
        category = parts[1]
        group = "templates"
    elif len(parts) >= 2 and parts[0] == "examples":
        category = "runtime"
        group = "examples"
    return {
        "template_path": str(path),
        "template_category": category,
        "template_group": group,
        "template_name": path.stem,
        "template_relative_path": str(rel),
    }


def _deep_merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_parameter_updates(
    template_cfg: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    return _deep_merge_dict(template_cfg, updates)


def materialize_generated_config(
    *,
    template_cfg: dict[str, Any],
    parameter_updates: dict[str, Any],
    output_path: str | Path,
) -> str:
    cfg = apply_parameter_updates(template_cfg, parameter_updates)
    return save_runtime_config(cfg, output_path)
