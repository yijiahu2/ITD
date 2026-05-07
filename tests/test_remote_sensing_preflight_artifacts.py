from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_layer.adapters import build_input_manifest
from ITD_agent.data_processing.artifact_store import ensure_data_processing_dirs
from ITD_agent.data_processing.remote_sensing.profiles import build_image_profiles, build_remote_sensing_preflight


def _write_dom(path: Path, data: np.ndarray, *, nodata: int | float = 0) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(100.0, 31.0, 0.02, 0.02),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_build_remote_sensing_preflight_writes_required_artifacts(tmp_path: Path) -> None:
    dom = tmp_path / "dom.tif"
    rgb = np.full((3, 320, 320), 128, dtype=np.uint8)
    rgb[:, 100:180, 100:180] = 40
    _write_dom(dom, rgb)

    cfg = {
        "output_dir": str(tmp_path / "outputs"),
        "runtime": {"run_name": "preflight_test"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": str(dom), "required": True}]},
        },
    }
    manifest = build_input_manifest(cfg)
    storage_layout = ensure_data_processing_dirs({"output_dir": str(tmp_path / "outputs"), "run_name": "preflight_test"})
    image_profiles = build_image_profiles(manifest, cfg)
    summary = build_remote_sensing_preflight(manifest, cfg, storage_layout, image_profiles)

    assert summary is not None
    assert Path(manifest.dom_input_contract.working_dom_path).exists()
    assert Path(manifest.dom_input_contract.valid_mask_path).exists()
    assert summary.working_dom_path == manifest.dom_input_contract.working_dom_path
    assert summary.valid_mask_path == manifest.dom_input_contract.valid_mask_path
    artifacts = summary.artifacts
    assert Path(artifacts["processing_block_profile_jsonl"]).exists()
    assert Path(artifacts["inference_tile_plan_csv"]).exists()
    assert Path(artifacts["preflight_report_json"]).exists()
    assert summary.tile_context_count > 0
