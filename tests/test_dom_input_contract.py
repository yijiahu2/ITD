from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_layer.adapters import build_input_manifest, normalize_agent_runtime_config


def _write_dom(path: Path, data: np.ndarray, *, resolution: float = 0.02, nodata: int | float = 0, count: int = 3) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": count,
        "dtype": str(data.dtype),
        "crs": "EPSG:4547",
        "transform": from_origin(100.0, 31.0, resolution, resolution),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            for band_idx in range(count):
                dst.write(data[band_idx], band_idx + 1)


def test_build_input_manifest_attaches_dom_input_contract(tmp_path: Path) -> None:
    dom = tmp_path / "dom.tif"
    rgb = np.full((3, 2048, 3072), 128, dtype=np.uint8)
    rgb[:, :128, :] = 0
    _write_dom(dom, rgb, resolution=0.02, nodata=0, count=3)

    cfg = {
        "runtime": {"run_name": "dom_contract_test", "mainline_profile": "A_DOM_ONLY"},
        "inputs": {
            "remote_sensing": {
                "images": [
                    {
                        "id": "dom_main",
                        "path": str(dom),
                        "required": True,
                        "sensor": "aerial_rgb",
                    }
                ]
            }
        },
    }

    manifest = build_input_manifest(cfg)
    contract = manifest.metadata.get("dom_input_contract")

    assert contract is not None
    assert contract["dom_id"] == "dom_main"
    assert contract["source_path"] == str(dom)
    assert contract["working_dom_path"].endswith("prepared_inputs/dom/dom_main/working_dom.vrt")
    assert contract["valid_mask_path"].endswith("prepared_inputs/dom/dom_main/valid_mask.tif")
    assert Path(contract["working_dom_path"]).exists()
    assert Path(contract["valid_mask_path"]).exists()
    assert contract["mainline_profile"] == "A_DOM_ONLY"
    assert contract["width"] == 3072
    assert contract["height"] == 2048
    assert contract["pixel_count"] == 3072 * 2048
    assert contract["crs"] == "EPSG:4547"
    assert contract["gsd_x_m"] == 0.02
    assert contract["gsd_y_m"] == 0.02
    assert contract["gsd_status"] == "acceptable"
    assert contract["band_count"] == 3
    assert contract["band_mapping"] == {"red": 1, "green": 2, "blue": 3}
    assert contract["normalization_policy"] == "uint8_passthrough"
    assert contract["nodata"] == 0
    assert contract["nodata_policy"] == "use_valid_mask"
    assert contract["processing_block_px"] == 5632
    assert contract["processing_block_stride_px"] == 5120
    assert contract["processing_block_overlap_px"] == 512
    assert contract["tile_px"] == 1024
    assert contract["tile_overlap_px"] == 256
    assert contract["tile_stride_px"] == 768
    assert contract["bsize"] == 256
    assert contract["processing_mode"] == "block_then_sliding_window"
    assert contract["output_clip_policy"] == "clip_to_original_bounds"
    assert contract["status"] == "ready"
    assert "large_dom_enable_resume" not in contract["warnings"]
    assert contract["estimated_block_count"] >= 1
    assert contract["estimated_tile_count"] >= 1


def test_dom_input_contract_warns_for_gsd_and_large_dom(tmp_path: Path) -> None:
    dom = tmp_path / "dom_large.tif"
    rgb = np.full((3, 9000, 9000), 128, dtype=np.uint8)
    _write_dom(dom, rgb, resolution=0.01, nodata=0, count=3)

    cfg = {
        "runtime": {"run_name": "dom_contract_warning_test", "mainline_profile": "A_DOM_ONLY"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom_large", "path": str(dom), "required": True}]}
        },
    }

    manifest = build_input_manifest(cfg)
    contract = manifest.metadata["dom_input_contract"]

    assert contract["gsd_status"] == "too_fine"
    assert "gsd_too_fine" in contract["warnings"]
    assert "resample_if_finer_than_threshold" in contract["warnings"]
    assert "large_dom_enable_resume" in contract["warnings"]
    assert contract["status"] == "warning"


def test_dom_input_contract_warns_for_gsd_coarser_than_warn_threshold(tmp_path: Path) -> None:
    dom = tmp_path / "dom_coarse.tif"
    rgb = np.full((3, 2048, 2048), 128, dtype=np.uint8)
    _write_dom(dom, rgb, resolution=0.06, nodata=0, count=3)

    cfg = {
        "runtime": {"run_name": "dom_contract_coarse_warning_test", "mainline_profile": "A_DOM_ONLY"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom_coarse", "path": str(dom), "required": True}]}
        },
    }

    manifest = build_input_manifest(cfg)
    contract = manifest.metadata["dom_input_contract"]

    assert contract["gsd_status"] == "too_coarse"
    assert "warn_if_coarser_than_threshold" in contract["warnings"]


def test_normalize_runtime_config_exports_dom_input_contract(tmp_path: Path) -> None:
    dom = tmp_path / "dom_norm.tif"
    rgb = np.full((3, 1024, 1024), 128, dtype=np.uint8)
    _write_dom(dom, rgb, resolution=0.02, nodata=0, count=3)

    cfg = {
        "runtime": {
            "run_name": "dom_runtime_contract",
            "mainline_profile": "A_DOM_ONLY",
            "conda_sh": "/tmp/conda.sh",
            "conda_env": "forest_agent",
            "work_dir": "/tmp",
        },
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom_norm", "path": str(dom), "required": True}]}
        },
        "ITD_agent": {"segmentation_models": {"main_model": {"script": "/tmp/seg.py"}}},
        "outputs": {"root_base_dir": str(tmp_path / "outputs")},
    }

    runtime_cfg, manifest = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["input_image"] == str(dom)
    assert runtime_cfg["_dom_input_contract"]["dom_id"] == "dom_norm"
    assert runtime_cfg["_dom_input_contract"]["working_dom_path"] == manifest.metadata["dom_input_contract"]["working_dom_path"]
    assert runtime_cfg["_input_manifest"]["metadata"]["dom_input_contract"]["dom_id"] == "dom_norm"
    assert Path(runtime_cfg["_dom_input_contract"]["working_dom_path"]).exists()
    assert Path(runtime_cfg["_dom_input_contract"]["valid_mask_path"]).exists()
