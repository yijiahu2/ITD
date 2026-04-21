from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_layer.adapters import build_input_manifest, normalize_agent_runtime_config
from ITD_agent.orchestration.runtime_paths import validate_runtime_cfg


def test_build_input_manifest_supports_dom_dem_chm_and_optional_xiaoban(tmp_path: Path) -> None:
    dom = tmp_path / "dom.tif"
    dem = tmp_path / "dem.tif"
    chm = tmp_path / "chm.tif"
    xiaoban = tmp_path / "xiaoban.shp"
    for path in [dom, dem, chm, xiaoban]:
        path.write_text("stub", encoding="utf-8")

    cfg = {
        "runtime": {"run_name": "test_run"},
        "inputs": {
            "remote_sensing": {
                "images": [{"id": "dom", "path": str(dom), "required": True}],
            },
            "terrain": {
                "dem": [{"id": "dem", "path": str(dem), "required": True}],
            },
            "canopy": {
                "chm": [{"id": "chm", "path": str(chm), "required": True}],
            },
            "industry_vectors": {
                "vectors": [{"id": "xiaoban", "path": str(xiaoban), "required": False}],
            },
            "public_datasets": {"datasets": []},
        },
    }

    manifest = build_input_manifest(cfg)

    assert manifest.input_modalities["image"] is True
    assert manifest.input_modalities["dem"] is True
    assert manifest.input_modalities["chm"] is True
    assert manifest.input_modalities["inventory"] is True
    assert manifest.chm_paths == [str(chm)]


def test_normalize_runtime_config_sets_chm_and_dsm_paths(tmp_path: Path) -> None:
    dom = tmp_path / "dom.tif"
    dem = tmp_path / "dem.tif"
    chm = tmp_path / "chm.tif"
    dsm = tmp_path / "dsm.tif"
    for path in [dom, dem, chm, dsm]:
        path.write_text("stub", encoding="utf-8")

    cfg = {
        "runtime": {
            "run_name": "test_run",
            "conda_sh": "/tmp/conda.sh",
            "conda_env": "test",
            "work_dir": "/tmp",
        },
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": str(dom), "required": True}]},
            "terrain": {"dem": [{"id": "dem", "path": str(dem), "required": True}]},
            "canopy": {"chm": [{"id": "chm", "path": str(chm), "required": True}]},
            "surface": {"dsm": [{"id": "dsm", "path": str(dsm), "required": False}]},
            "industry_vectors": {"vectors": []},
            "survey_data": {"tables": []},
            "public_datasets": {"datasets": []},
        },
        "ITD_agent": {"segmentation_models": {"main_model": {"script": "/tmp/seg.py"}}},
        "outputs": {"root_base_dir": str(tmp_path / "outputs")},
    }

    runtime_cfg, manifest = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["input_image"] == str(dom)
    assert runtime_cfg["dem_tif"] == str(dem)
    assert runtime_cfg["chm_tif"] == str(chm)
    assert runtime_cfg["dsm_tif"] == str(dsm)
    assert manifest.input_modalities["chm"] is True
    assert manifest.input_modalities["dsm"] is True


def test_validate_runtime_cfg_allows_missing_optional_xiaoban_reference() -> None:
    cfg = {
        "input_image": "/tmp/dom.tif",
        "output_dir": "/tmp/out",
        "metrics_json": "/tmp/out/evaluation_metrics.json",
        "details_csv": "/tmp/out/evaluation_details.csv",
        "semantic_prior_script": "/tmp/semantic.py",
        "segmentation_script": "/tmp/seg.py",
        "conda_sh": "/tmp/conda.sh",
        "conda_env": "test",
        "work_dir": "/tmp",
        "diam_list": "160,256,384",
        "tile": 2048,
        "overlap": 256,
        "tile_overlap": 0.35,
        "bsize": 256,
        "augment": True,
        "iou_merge_thr": 0.28,
    }

    validate_runtime_cfg(cfg)
