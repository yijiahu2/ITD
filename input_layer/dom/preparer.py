from __future__ import annotations

from shutil import copy2
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.shutil import copy as rasterio_copy

from input_layer.common import resolve_path
from input_layer.contracts import DomInputContract, PreparedAsset, PreparedInputIndex


def _workspace_root(cfg: dict[str, Any], config_path: str | None = None) -> Path:
    output_dir = cfg.get("output_dir")
    if output_dir:
        return Path(str(output_dir)).expanduser().resolve()

    outputs = cfg.get("outputs") or {}
    config_dir = Path(config_path).expanduser().resolve().parent if config_path else None
    root_dir = outputs.get("root_dir")
    if root_dir:
        return Path(str(resolve_path(root_dir, config_dir))).expanduser().resolve()
    root_base_dir = outputs.get("root_base_dir")
    if root_base_dir:
        runtime = cfg.get("runtime") or {}
        run_name = runtime.get("run_name") or cfg.get("run_name") or "itd_agent_run"
        return (Path(str(resolve_path(root_base_dir, config_dir))).expanduser() / str(run_name)).resolve()

    runtime = cfg.get("runtime") or {}
    run_name = runtime.get("run_name") or cfg.get("run_name") or "itd_agent_run"
    project_root = Path.cwd()
    return (project_root / "outputs" / str(run_name)).resolve()


def derive_input_workspace(cfg: dict[str, Any], config_path: str | None = None) -> dict[str, str]:
    root = _workspace_root(cfg, config_path=config_path)
    registry_root = root / "input_registry"
    prepared_root = root / "prepared_inputs"
    return {
        "workspace_root": str(root),
        "registry_root": str(registry_root),
        "prepared_root": str(prepared_root),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_working_dom(dom_input_contract: DomInputContract) -> str:
    src_path = Path(dom_input_contract.source_path)
    dst_path = Path(dom_input_contract.working_dom_path)
    _ensure_parent(dst_path)
    suffix = src_path.suffix.lower()
    if suffix == ".vrt":
        copy2(src_path, dst_path)
        return str(dst_path)

    with rasterio.open(src_path) as src:
        rasterio_copy(src, str(dst_path), driver="VRT")
    return str(dst_path)


def _prepare_valid_mask(dom_input_contract: DomInputContract, working_dom_path: str) -> str:
    src_path = Path(working_dom_path)
    dst_path = Path(dom_input_contract.valid_mask_path)
    _ensure_parent(dst_path)
    with rasterio.open(src_path) as src:
        data = src.read()
        mask = np.all(np.isfinite(data), axis=0)
        nodata = src.nodata if dom_input_contract.nodata is None else dom_input_contract.nodata
        if nodata is not None:
            mask &= ~np.any(np.isclose(data, float(nodata)), axis=0)
        out_profile = src.profile.copy()
        out_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="LZW")
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
    return str(dst_path)


def prepare_dom_runtime_assets(dom_input_contract: DomInputContract | None) -> dict[str, str]:
    if dom_input_contract is None:
        return {}
    try:
        working_dom_path = _prepare_working_dom(dom_input_contract)
        valid_mask_path = _prepare_valid_mask(dom_input_contract, working_dom_path)
        return {
            "working_dom_path": working_dom_path,
            "valid_mask_path": valid_mask_path,
        }
    except (RasterioIOError, ValueError):
        return {}


def build_dom_prepared_input_index(
    dom_input_contract: DomInputContract | None,
    cfg: dict[str, Any],
    config_path: str | None = None,
) -> PreparedInputIndex:
    workspace = derive_input_workspace(cfg, config_path=config_path)
    assets: list[PreparedAsset] = []
    prepared_paths = prepare_dom_runtime_assets(dom_input_contract)

    if dom_input_contract:
        dom_root = Path(prepared_paths.get("working_dom_path") or dom_input_contract.working_dom_path).parent
        assets.append(
            PreparedAsset(
                source_type="dom_input_contract",
                source_id=dom_input_contract.dom_id,
                raw_path=dom_input_contract.source_path,
                prepared_path=prepared_paths.get("working_dom_path") or dom_input_contract.working_dom_path,
                registry_key=f"dom/{dom_input_contract.dom_id}",
                preparation_actions=[
                    "validate_dom_input_contract",
                    "prepare_working_dom_vrt",
                    "prepare_valid_mask",
                    "reserve_block_plan_dir",
                    "reserve_tile_plan_dir",
                ],
                notes=[
                    f"valid_mask_path={prepared_paths.get('valid_mask_path') or dom_input_contract.valid_mask_path}",
                    f"block_plan_dir={dom_root / 'block_plan'}",
                    f"tile_plan_dir={dom_root / 'tile_plan'}",
                    f"dom_debug_dir={dom_root / 'debug'}",
                ],
            )
        )

    return PreparedInputIndex(
        registry_root=workspace["registry_root"],
        prepared_root=workspace["prepared_root"],
        assets=assets,
        metadata={
            "workspace": workspace,
            "prepared_paths": prepared_paths,
            "asset_counts": {
                "dom_input_contract": 1 if dom_input_contract else 0,
            },
        },
    )
