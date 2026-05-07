from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from input_layer.contracts import InputManifest


def _write_json(payload: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def register_input_bundle(
    manifest: InputManifest,
    runtime_cfg: dict[str, Any],
) -> dict[str, str | None]:
    preparation = manifest.preparation.to_dict() if manifest.preparation else {}
    registry_root = (
        Path(preparation.get("registry_root"))
        if preparation.get("registry_root")
        else Path(runtime_cfg["output_dir"]).resolve() / "input_registry"
    )
    registry_root.mkdir(parents=True, exist_ok=True)

    manifest_path = _write_json(manifest.to_dict(), registry_root / "input_manifest.json")
    dom_contract_path = None
    if manifest.dom_input_contract:
        dom_contract_path = _write_json(manifest.dom_input_contract.to_dict(), registry_root / "dom_input_contract.json")
    validation_path = None
    prepared_index_path = None
    if manifest.validation:
        validation_path = _write_json(manifest.validation.to_dict(), registry_root / "input_validation_report.json")
    if manifest.preparation:
        prepared_index_path = _write_json(manifest.preparation.to_dict(), registry_root / "prepared_input_index.json")

    registry_payload = {
        "registry_root": str(registry_root),
        "manifest_json": manifest_path,
        "dom_input_contract_json": dom_contract_path,
        "validation_json": validation_path,
        "prepared_index_json": prepared_index_path,
    }
    registry_index = _write_json(registry_payload, registry_root / "registry_index.json")
    registry_payload["registry_index_json"] = registry_index
    return registry_payload
