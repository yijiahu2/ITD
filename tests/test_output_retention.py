from __future__ import annotations

from pathlib import Path

from input_layer.adapters import normalize_agent_runtime_config
from ITD_agent.orchestration.output_management import (
    apply_persistent_retention,
    build_retained_summary,
    get_retention_profile,
)


def test_get_retention_profile_prefers_debug_flag() -> None:
    assert get_retention_profile({"cleanup_policy": "minimal", "keep_debug_outputs": True}) == "debug"


def test_get_retention_profile_accepts_minimal() -> None:
    assert get_retention_profile({"cleanup_policy": "minimal"}) == "minimal"


def test_build_retained_summary_drops_heavy_sections_for_minimal() -> None:
    summary = {
        "mode": "ITD_agent_run",
        "run_name": "demo_run",
        "config_path": "/tmp/demo.yaml",
        "run_meta": {
            "run_name": "demo_run",
            "experiment_name": "demo_exp",
            "input_image": "/tmp/demo.tif",
            "output_dir": "/tmp/out",
            "segmentation_script": "seg.py",
            "unused_field": {"huge": True},
        },
        "input_layer": {"remote_sensing_images": [{"id": "img", "path": "/tmp/demo.tif"}]},
        "data_processing": {
            "processing_summary": {
                "metadata": {"image_count": 1},
                "requested_tasks": [{"id": "huge"}],
            },
            "terrain_info": {"dem_tif": "/tmp/dem.tif"},
        },
        "llm_gateway": {
            "main_model_planning_used_llm": True,
            "main_model_gateway_trace": {
                "task_type": "planning",
                "status": "success",
                "provider": "doubao",
                "model": "ep-test",
                "raw_text": "very large",
            },
            "run_retrospective": {
                "task_type": "retrospective",
                "status": "success",
                "provider": "doubao",
                "model": "ep-test",
                "parsed_result": {"actions": ["keep"]},
                "raw_text": "very large",
            },
        },
        "planning_scheduler": {
            "main_model_plan": {
                "generated_config_path": "/tmp/generated.yaml",
                "scheduler_context": {"very": "large"},
                "llm_gateway_result": {
                    "task_type": "planning",
                    "status": "success",
                    "provider": "doubao",
                    "model": "ep-test",
                    "raw_text": "huge",
                },
            },
            "roi_rounds": [{"round_idx": 1, "failure_modes": ["x"]}],
        },
        "segmentation_model": {
            "main_model": {
                "execution_request": {"phase": "main", "runtime_cfg": {"selected_model_name": "main_model"}},
                "execution_result": {"status": "success", "output_paths": {"y_inst_shp": "/tmp/Y_inst.shp"}},
            },
            "roi_rounds": [{"round_idx": 1}],
            "y_inst_shp": "/tmp/Y_inst.shp",
        },
        "metrics": {"score": 0.9},
        "final_evaluation": {"status": "success", "score": 0.9},
        "failure_analysis": {"top_problem_cases": [{"id": 1}]},
        "summary_json": "/tmp/summary.json",
        "metrics_json": "/tmp/metrics.json",
        "details_csv": "/tmp/details.csv",
        "report_md": "/tmp/report.md",
        "final_outputs": {"tree_crowns_shp": "/tmp/final_outputs/tree_crowns.shp"},
        "cleanup": {"removed_files": ["/tmp/a"]},
        "runtime_cleanup": {"removed": True},
    }

    retained = build_retained_summary(summary=summary, runtime_cfg={"cleanup_policy": "minimal"})

    assert retained["run_name"] == "demo_run"
    assert retained["data_processing"]["processing_summary"]["metadata"]["image_count"] == 1
    assert "requested_tasks" not in retained["data_processing"]["processing_summary"]
    assert retained["llm_gateway"]["main_model_gateway_trace"]["model"] == "ep-test"
    assert "raw_text" not in retained["llm_gateway"]["main_model_gateway_trace"]
    assert retained["planning_scheduler"]["main_model_plan"]["llm_gateway_result"]["status"] == "success"


def test_apply_persistent_retention_removes_stage_directories_for_minimal(tmp_path: Path) -> None:
    persistent_root = tmp_path / "run"
    metrics_parent = persistent_root
    for rel in ["input_registry", "planning_scheduler", "data_processing", "roi_refinement"]:
        (persistent_root / rel).mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": "demo_run",
        "summary_json": str(metrics_parent / "ITD_agent_run_summary.json"),
    }
    runtime_cfg = {
        "cleanup_policy": "minimal",
        "output_dir": str(persistent_root),
        "persistent_output_dir": str(persistent_root),
        "metrics_json": str(metrics_parent / "evaluation_metrics.json"),
        "details_csv": str(metrics_parent / "evaluation_details.csv"),
        "run_name": "demo_run",
    }

    retention_info = apply_persistent_retention(summary=summary, runtime_cfg=runtime_cfg)

    assert retention_info["profile"] == "minimal"
    assert not (persistent_root / "input_registry").exists()
    assert not (persistent_root / "planning_scheduler").exists()


def test_normalize_agent_runtime_config_forces_minimal_for_new_style_config(tmp_path: Path) -> None:
    image_path = tmp_path / "demo.tif"
    vector_path = tmp_path / "demo.shp"
    image_path.write_text("rgb", encoding="utf-8")
    vector_path.write_text("vector", encoding="utf-8")
    cfg = {
        "runtime": {"run_name": "demo_run"},
        "inputs": {
            "remote_sensing": {"images": [{"id": "img", "path": str(image_path), "required": True}]},
            "industry_vectors": {
                "vectors": [
                    {
                        "id": "xb",
                        "path": str(vector_path),
                        "key_fields": ["XBH"],
                        "field_mapping": {
                            "xiaoban_id": "XBH",
                            "tree_count": "LMSL",
                            "crown_width": "PJGF",
                            "closure": "YBD",
                            "area_ha": "MJ_hm2",
                        },
                        "required": True,
                    }
                ]
            },
        },
        "outputs": {
            "root_dir": str(tmp_path / "outputs"),
            "cleanup_policy": "debug",
            "temp_runtime": {"enabled": True, "cleanup_after_run": False},
        },
    }

    runtime_cfg, _ = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["cleanup_policy"] == "minimal"
    assert runtime_cfg["keep_debug_outputs"] is False
    assert runtime_cfg["keep_semantic_prior_artifacts"] is False
    assert runtime_cfg["cleanup_temp_runtime"] is True


def test_normalize_agent_runtime_config_forces_minimal_for_legacy_config(tmp_path: Path) -> None:
    cfg = {
        "run_name": "legacy_demo",
        "input_image": str(tmp_path / "demo.tif"),
        "output_dir": str(tmp_path / "legacy_outputs"),
        "cleanup_policy": "debug",
        "keep_debug_outputs": True,
        "keep_semantic_prior_artifacts": True,
    }

    runtime_cfg, _ = normalize_agent_runtime_config(cfg)

    assert runtime_cfg["cleanup_policy"] == "minimal"
    assert runtime_cfg["keep_debug_outputs"] is False
    assert runtime_cfg["keep_semantic_prior_artifacts"] is False
    assert runtime_cfg["cleanup_temp_runtime"] is True
