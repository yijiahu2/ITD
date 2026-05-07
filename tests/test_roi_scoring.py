from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv
from ITD_agent.evaluation_analysis.reference_quality_engine import build_reference_score_breakdown
from ITD_agent.evaluation_analysis.roi_assessment import build_roi_assessment


def test_reference_score_breakdown_prioritizes_crown_when_count_is_stable() -> None:
    metrics = {
        "tree_count_error_ratio": 0.05,
        "mean_crown_width_error_ratio": 0.24,
        "closure_error_abs": 0.08,
        "density_error_abs": 50.0,
        "expected_density": 500.0,
    }
    cfg = {
        "ITD_agent": {
            "planning": {
                "roi_extraction": {
                    "score_weights": {
                        "tree_count_error_ratio": 0.30,
                        "mean_crown_width_error_ratio": 0.40,
                        "closure_error_abs": 0.20,
                        "density_error_ratio": 0.10,
                    }
                }
            }
        }
    }

    breakdown = build_reference_score_breakdown(metrics, cfg=cfg)

    assert breakdown["weights"]["mean_crown_width_error_ratio"] > breakdown["weights"]["tree_count_error_ratio"]
    assert breakdown["normalized_metrics"]["density_error_ratio"] == 0.1
    assert breakdown["metric_groups"]["inventory_count_alignment"]["metrics"][0]["metric"] == "tree_count_error_ratio"
    assert breakdown["metric_groups"]["crown_boundary_alignment"]["metrics"][0]["metric"] == "mean_crown_width_error_ratio"
    assert breakdown["weighted_terms"]["mean_crown_width_error_ratio"]["contribution"] > breakdown["weighted_terms"]["tree_count_error_ratio"]["contribution"]


def test_build_roi_assessment_uses_auto_thresholds_when_config_uses_zero(tmp_path) -> None:
    details_csv = tmp_path / "details.csv"
    details_csv.write_text(
        "xiaoban_id,pred_tree_count,expected_tree_count,tree_count_error_abs,pred_mean_crown_width,expected_mean_crown_width,mean_crown_width_error_abs,pred_cover_ratio,expected_closure,closure_error_abs,pred_density_trees_per_ha,expected_density,density_error_abs\n"
        "1,95,100,5,3.5,5.0,1.5,0.62,0.72,0.10,410,500,90\n",
        encoding="utf-8",
    )
    cfg = {
        "ITD_agent": {
            "planning": {
                "roi_extraction": {
                    "enabled": True,
                    "use_llm": False,
                    "max_rounds": 2,
                    "top_k": 2,
                    "tree_count_error_ratio_thr": 0.0,
                    "mean_crown_width_error_ratio_thr": 0.0,
                    "closure_error_abs_thr": 0.0,
                }
            }
        }
    }
    metrics = {
        "tree_count_error_ratio": 0.05,
        "mean_crown_width_error_ratio": 0.30,
        "closure_error_abs": 0.10,
        "density_error_abs": 90.0,
        "expected_density": 500.0,
    }

    assessment = build_roi_assessment(
        cfg,
        metrics,
        str(details_csv),
        round_idx=0,
    )

    assert "mean_crown_width_error_ratio" in assessment["trigger_metrics"]
    assert "tree_count_error_ratio" not in assessment["trigger_metrics"]
    assert assessment["metric_thresholds"]["tree_count_error_ratio"] > assessment["metric_thresholds"]["mean_crown_width_error_ratio"]
    assert assessment["trigger_details"]["mean_crown_width_error_ratio"]["category"] == "crown_boundary_alignment"


def test_detail_ranker_keeps_boundary_heavy_case_at_top(tmp_path) -> None:
    details_csv = tmp_path / "details.csv"
    details_csv.write_text(
        "xiaoban_id,pred_tree_count,expected_tree_count,tree_count_error_abs,pred_mean_crown_width,expected_mean_crown_width,mean_crown_width_error_abs,pred_cover_ratio,expected_closure,closure_error_abs,pred_density_trees_per_ha,expected_density,density_error_abs\n"
        "count_case,90,100,10,4.8,5.0,0.2,0.69,0.72,0.03,480,500,20\n"
        "boundary_case,98,100,2,3.1,5.2,2.1,0.58,0.74,0.16,430,500,70\n",
        encoding="utf-8",
    )

    summary = summarize_details_csv(str(details_csv), top_k=1)

    assert summary["top_k_reference_units"][0]["reference_unit_id"] == "boundary_case"
