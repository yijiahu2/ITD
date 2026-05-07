from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_layer.adapters import build_input_manifest, normalize_agent_runtime_config
from ITD_agent.llm_gateway.gateway import build_dynamic_system_prompt
from ITD_agent.planning.scheduler.context_builder import build_scheduler_context


def _touch(path: Path) -> str:
    path.write_text("stub", encoding="utf-8")
    return str(path)


def test_mainline_a_profile_gates_non_dom_inputs(tmp_path: Path) -> None:
    dom = _touch(tmp_path / "dom.tif")
    dem = _touch(tmp_path / "dem.tif")
    chm = _touch(tmp_path / "chm.tif")
    xiaoban = _touch(tmp_path / "xiaoban.shp")
    knowledge = _touch(tmp_path / "rules.md")
    public_dataset = _touch(tmp_path / "public.json")

    cfg = {
        "runtime": {"run_name": "a_profile", "mainline_profile": "A_DOM_ONLY", "work_dir": str(tmp_path)},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": dom, "required": True}]},
            "terrain": {"dem": [{"id": "dem", "path": dem, "required": True}]},
            "canopy": {"chm": [{"id": "chm", "path": chm, "required": True}]},
            "industry_vectors": {"vectors": [{"id": "xiaoban", "path": xiaoban, "required": True}]},
            "domain_knowledge": {"items": [{"id": "rules", "type": "text", "path": knowledge}]},
            "public_datasets": {"datasets": [{"id": "public", "format": "coco", "annotation_path": public_dataset}]},
        },
        "ITD_agent": {"segmentation_models": {"main_model": {"script": "/tmp/seg.py"}}},
        "outputs": {"root_base_dir": str(tmp_path / "outputs")},
    }

    runtime_cfg, manifest = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["mainline_profile"] == "A_DOM_ONLY"
    assert runtime_cfg["_mainline_capabilities"]["allow_dem"] is False
    assert runtime_cfg["_mainline_capabilities"]["allow_public_datasets"] is True
    assert runtime_cfg["_mainline_capabilities"]["allow_memory_context"] is True
    assert runtime_cfg["_mainline_capabilities"]["allow_finetune_pool_context"] is True
    assert runtime_cfg["_mainline_capabilities"]["allow_external_knowledge"] is False
    assert runtime_cfg["input_image"] == dom
    assert "dem_tif" not in runtime_cfg
    assert "chm_tif" not in runtime_cfg
    assert "xiaoban_shp" not in runtime_cfg
    assert "reference_vector_path" not in runtime_cfg
    assert manifest.input_modalities == {
        "image": True,
        "dem": False,
        "chm": False,
        "dsm": False,
        "inventory": False,
        "knowledge": False,
        "public_datasets": True,
    }
    assert manifest.metadata["profile_gate"]["ignored_counts"]["terrain_dem"] == 1
    assert "public_datasets" not in manifest.metadata["profile_gate"].get("ignored_counts", {})


def test_mainline_b_profile_enables_dem_chm_and_shared_learning_context(tmp_path: Path) -> None:
    dom = _touch(tmp_path / "dom.tif")
    dem = _touch(tmp_path / "dem.tif")
    chm = _touch(tmp_path / "chm.tif")
    xiaoban = _touch(tmp_path / "xiaoban.shp")
    knowledge = _touch(tmp_path / "rules.md")

    cfg = {
        "runtime": {"run_name": "b_profile", "mainline_profile": "B_DOM_DEM_CHM_KNOWLEDGE", "work_dir": str(tmp_path)},
        "inputs": {
            "remote_sensing": {"images": [{"id": "dom", "path": dom, "required": True}]},
            "terrain": {"dem": [{"id": "dem", "path": dem, "required": True}]},
            "canopy": {"chm": [{"id": "chm", "path": chm, "required": True}]},
            "industry_vectors": {"vectors": [{"id": "xiaoban", "path": xiaoban, "required": False}]},
            "domain_knowledge": {"items": [{"id": "rules", "type": "text", "path": knowledge}]},
        },
        "ITD_agent": {"segmentation_models": {"main_model": {"script": "/tmp/seg.py"}}},
        "outputs": {"root_base_dir": str(tmp_path / "outputs")},
    }

    runtime_cfg, manifest = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["mainline_profile"] == "B_DOM_DEM_CHM_KNOWLEDGE"
    assert runtime_cfg["_mainline_capabilities"]["allow_chm"] is True
    assert runtime_cfg["_mainline_capabilities"]["allow_public_datasets"] is True
    assert runtime_cfg["_mainline_capabilities"]["allow_memory_context"] is True
    assert runtime_cfg["_mainline_capabilities"]["allow_finetune_pool_context"] is True
    assert runtime_cfg["dem_tif"] == dem
    assert runtime_cfg["chm_tif"] == chm
    assert runtime_cfg["xiaoban_shp"] == xiaoban
    assert runtime_cfg["reference_vector_path"] == xiaoban
    assert runtime_cfg["inventory_vector_path"] == xiaoban
    assert manifest.input_modalities["dem"] is True
    assert manifest.input_modalities["chm"] is True
    assert manifest.input_modalities["inventory"] is True
    assert manifest.input_modalities["knowledge"] is False


def test_scheduler_context_and_system_prompt_are_profile_aware() -> None:
    runtime_cfg = {
        "mainline_profile": "A_DOM_ONLY",
        "_mainline_capabilities": {
            "allow_dem": False,
            "allow_chm": False,
            "allow_external_knowledge": False,
            "allow_public_datasets": True,
            "allow_memory_context": True,
            "allow_finetune_pool_context": True,
        },
        "_input_assessment": {
            "scene_analysis": {
                "terrain_analysis": {"labels": ["steep"], "global_background": {"landform_type": "mountain"}},
            }
        },
        "_data_processing_summary": {
            "metadata": {
                "input_manifest_summary": {
                    "domain_knowledge_items": [{"source_id": "rules", "normalized_type": "rule_knowledge"}],
                    "public_datasets": [{"source_id": "public", "usage_roles": ["benchmark"]}],
                }
            },
        },
    }

    context = build_scheduler_context(runtime_cfg=runtime_cfg)
    prompt = build_dynamic_system_prompt(runtime_cfg=runtime_cfg, task_type="planning")

    assert context["mainline_profile"] == "A_DOM_ONLY"
    assert context["scene_profile"]["terrain_type"] is None
    assert context["scene_profile"]["knowledge_profile_types"] == []
    assert context["terrain_analysis"] == {}
    assert context["knowledge_profiles"] == []
    assert context["public_dataset_profiles"] == [{"source_id": "public", "usage_roles": ["benchmark"]}]
    assert "memory_store_context" in context
    assert "finetune_pool_context" in context
    assert "DOM-only" in prompt
    assert "禁止引用 DEM/CHM" in prompt


def test_b_height_structure_output_extracts_chm_instance_attributes(tmp_path: Path) -> None:
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from output_layer.publisher import build_height_structure_outputs

    crowns = tmp_path / "crowns.gpkg"
    chm = tmp_path / "chm.tif"
    out_vector = tmp_path / "annotated.gpkg"
    out_summary = tmp_path / "height_summary.json"

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:4547")
    gdf.to_file(crowns, driver="GPKG")
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    with rasterio.open(
        chm,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4547",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(data, 1)

    summary = build_height_structure_outputs(
        crowns_src=crowns,
        chm_raster=chm,
        annotated_vector_dst=out_vector,
        summary_dst=out_summary,
    )

    assert summary["available"] is True
    assert summary["height_attributed_count"] == 1
    assert Path(summary["annotated_vector"]).exists()
    annotated = gpd.read_file(summary["annotated_vector"])
    assert "tree_height_p95" in annotated.columns
    assert annotated.loc[0, "tree_height_p95"] > 0


def test_final_report_includes_profile_and_b_height_structure(tmp_path: Path) -> None:
    from output_layer.reporting.experiment_report import build_experiment_report

    summary = {
        "run_name": "b_report",
        "input_manifest": {
            "metadata": {
                "mainline_profile": "B_DOM_DEM_CHM_KNOWLEDGE",
                "input_modalities": {"image": True, "dem": True, "chm": True},
            }
        },
        "final_outputs": {
            "tree_crowns_height_structure_gpkg": "/tmp/tree_crowns_height_structure.gpkg",
            "height_structure_summary_json": "/tmp/height_structure_summary.json",
            "metadata": {
                "mainline_profile": "B_DOM_DEM_CHM_KNOWLEDGE",
                "mainline_capabilities": {"output_height_structure": True},
                "height_structure_summary": {
                    "available": True,
                    "instance_count": 2,
                    "height_attributed_count": 2,
                    "tree_height_p95_mean": 12.5,
                    "tree_height_p95_max": 18.0,
                    "structure_tag_counts": {"low_simple": 2},
                },
            },
        },
        "final_evaluation": {"evaluation_mode": "unavailable", "message": "test"},
    }
    report_path = tmp_path / "report.md"

    build_experiment_report(summary, report_path)
    text = report_path.read_text(encoding="utf-8")

    assert "主线 Profile" in text
    assert "B线高度与结构输出" in text
    assert "B_DOM_DEM_CHM_KNOWLEDGE" in text
