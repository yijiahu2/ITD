from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis import roi_assessment
from ITD_agent.evaluation_analysis import online_quality_engine
from ITD_agent.evaluation_analysis.metrics_catalog import describe_metric


def test_roi_assessment_does_not_own_extraction_or_llm_decisions() -> None:
    assert not hasattr(roi_assessment, "extract_signal_driven_roi_candidates")
    assert not hasattr(roi_assessment, "request_roi_candidate_selection")
    assert not hasattr(roi_assessment, "request_roi_decision")


def test_online_quality_engine_delegates_output_diagnostics_to_data_processing() -> None:
    assert not hasattr(online_quality_engine, "_load_instances")
    assert not hasattr(online_quality_engine, "_semantic_instance_consistency")
    assert not hasattr(online_quality_engine, "_height_consistency")


def test_metric_catalog_classifies_reference_and_online_metrics() -> None:
    assert describe_metric("tree_count_error_ratio")["category"] == "inventory_count_alignment"
    assert describe_metric("mean_crown_width_error_ratio")["category"] == "crown_boundary_alignment"
    assert describe_metric("semantic_instance_consistency")["category"] == "semantic_instance_alignment"
    assert describe_metric("geometry_plausibility")["category"] == "instance_geometry_plausibility"


def test_build_roi_assessment_uses_precomputed_candidates(tmp_path: Path) -> None:
    details_csv = tmp_path / "details.csv"
    details_csv.write_text(
        "xiaoban_id,pred_tree_count,expected_tree_count,tree_count_error_abs,"
        "pred_mean_crown_width,expected_mean_crown_width,mean_crown_width_error_abs,"
        "pred_cover_ratio,expected_closure,closure_error_abs,"
        "pred_density_trees_per_ha,expected_density,density_error_abs\n"
        "detail_case,98,100,2,4.8,5.0,0.2,0.69,0.72,0.03,480,500,20\n",
        encoding="utf-8",
    )
    cfg = {
        "ITD_agent": {
            "planning": {
                "roi_extraction": {
                    "enabled": True,
                    "use_llm": True,
                    "top_k": 1,
                    "signal_candidate_max_keep": 1,
                }
            }
        }
    }
    metrics = {
        "tree_count_error_ratio": 0.24,
        "mean_crown_width_error_ratio": 0.05,
        "closure_error_abs": 0.03,
    }
    candidates = [
        {"candidate_id": "provided_low", "score": 0.3},
        {"candidate_id": "provided_high", "score": 0.9},
    ]

    assessment = roi_assessment.build_roi_assessment(
        cfg,
        metrics,
        str(details_csv),
        round_idx=0,
        y_inst_tif="/path/that/should/not/be/read.tif",
        candidate_rois=candidates,
    )
    decision = roi_assessment.decide_roi_continuation(cfg, roi_assessment=assessment, metrics=metrics)

    assert assessment["candidate_source"] == "precomputed"
    assert [item["candidate_id"] for item in assessment["candidate_rois"]] == ["provided_high"]
    assert decision["decision_source"] == "heuristic"
    assert "llm_output" not in decision


def test_orchestration_builds_roi_candidates_before_evaluation(monkeypatch) -> None:
    from ITD_agent.orchestration import orchestrator

    captured: dict[str, object] = {}

    def fake_extract_signal_driven_roi_candidates(**kwargs):
        captured.update(kwargs)
        return {
            "selected_candidates": [{"candidate_id": "dp_roi_1", "score": 0.8}],
            "summary_json": "/tmp/roi_summary.json",
        }

    monkeypatch.setattr(orchestrator, "extract_signal_driven_roi_candidates", fake_extract_signal_driven_roi_candidates)

    result = orchestrator._build_roi_candidate_context(
        cfg={
            "output_dir": "/tmp/out",
            "input_image": "/tmp/dom.tif",
            "ITD_agent": {"planning": {"roi_extraction": {"enabled": True, "top_k": 2}}},
        },
        y_inst_tif="/tmp/y_inst.tif",
        m_sem_tif="/tmp/m_sem.tif",
        terrain_info={},
        top_k=2,
        round_idx=0,
    )

    assert captured["y_inst_tif"] == "/tmp/y_inst.tif"
    assert result["candidate_rois"][0]["candidate_id"] == "dp_roi_1"
    assert result["signal_roi_summary"]["summary_json"] == "/tmp/roi_summary.json"


def test_orchestration_rasterizes_instance_vector_when_label_tif_is_missing(monkeypatch) -> None:
    from ITD_agent.orchestration import orchestrator

    captured: dict[str, object] = {}

    def fake_rasterize_instances_to_label_raster(**kwargs):
        captured["rasterize"] = kwargs
        return "/tmp/generated_labels.tif"

    def fake_extract_signal_driven_roi_candidates(**kwargs):
        captured["extract"] = kwargs
        return {"selected_candidates": [{"candidate_id": "vector_roi", "score": 0.7}]}

    monkeypatch.setattr(orchestrator, "rasterize_instances_to_label_raster", fake_rasterize_instances_to_label_raster)
    monkeypatch.setattr(orchestrator, "extract_signal_driven_roi_candidates", fake_extract_signal_driven_roi_candidates)

    result = orchestrator._build_roi_candidate_context(
        cfg={
            "output_dir": "/tmp/out",
            "input_image": "/tmp/dom.tif",
            "ITD_agent": {"planning": {"roi_extraction": {"enabled": True, "top_k": 2}}},
        },
        y_inst_tif=None,
        m_sem_tif="/tmp/m_sem.tif",
        terrain_info={},
        top_k=2,
        round_idx=1,
        inst_shp="/tmp/merged.shp",
    )

    assert captured["rasterize"]["inst_shp"] == "/tmp/merged.shp"
    assert captured["extract"]["y_inst_tif"] == "/tmp/generated_labels.tif"
    assert result["candidate_rois"][0]["candidate_id"] == "vector_roi"
