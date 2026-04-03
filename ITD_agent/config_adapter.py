from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from input_layer.adapters import build_input_manifest, normalize_agent_runtime_config
from input_layer.contracts import InputManifest


def _to_yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yaml_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_yaml_safe(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _to_yaml_safe(value.item())
        except Exception:
            pass
    return value


def load_raw_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_runtime_config(path: str | Path) -> tuple[dict[str, Any], InputManifest]:
    raw_cfg = load_raw_yaml(path)
    return normalize_agent_runtime_config(raw_cfg, config_path=str(path))


def save_runtime_config(cfg: dict[str, Any], path: str | Path) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_yaml_safe(cfg), f, allow_unicode=True, sort_keys=False)
    return str(out_path)


def ensure_input_manifest(cfg: dict[str, Any], config_path: str | None = None) -> InputManifest:
    manifest_data = cfg.get("_input_manifest")
    if isinstance(manifest_data, dict):
        public_datasets = manifest_data.get("public_datasets") or []
        manifest_data = dict(manifest_data)
        manifest_data["public_datasets"] = public_datasets
        return build_input_manifest(cfg, config_path=config_path)
    return build_input_manifest(cfg, config_path=config_path)
