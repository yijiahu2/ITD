from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.expert_model_assessment import evaluate_expert_model_assessment
from ITD_agent.evaluation_analysis.final_assessment import evaluate_reference_quality_result
from ITD_agent.evaluation_analysis.finetune_effect_assessment import compare_finetune_effect
from ITD_agent.evaluation_analysis.roi_assessment import build_roi_assessment


def _write_details(path: Path) -> None:
    path.write_text(
        "xiaoban_id,pred_tree_count,expected_tree_count,tree_count_error_abs,"
        "pred_mean_crown_width,expected_mean_crown_width,mean_crown_width_error_abs,"
        "pred_cover_ratio,expected_closure,closure_error_abs,"
        "pred_density_trees_per_ha,expected_density,density_error_abs\n"
        "1,95,100,5,3.5,5.0,1.5,0.62,0.72,0.10,410,500,90\n",
        encoding="utf-8",
    )


def test_roi_assessment_outputs_flow_decision(tmp_path: Path) -> None:
    details_csv = tmp_path / "details.csv"
    _write_details(details_csv)

    result = build_roi_assessment(
        {"ITD_agent": {"planning": {"roi_extraction": {"enabled": True, "top_k": 1}}}},
        {
            "tree_count_error_ratio": 0.22,
            "mean_crown_width_error_ratio": 0.10,
            "closure_error_abs": 0.02,
            "density_error_abs": 10,
            "expected_density": 100,
        },
        str(details_csv),
        round_idx=0,
        candidate_rois=[{"candidate_id": "roi_1", "score": 0.8}],
    )

    flow = result["flow_decision"]
    assert flow["decision_stage"] == "roi_refinement_decision"
    assert flow["decision_question"] == "是否进入或继续 ROI 局部细化？"
    assert flow["core_metrics"]["continue_refinement"] is True
    assert "trigger_metrics" in flow["core_metrics"]
    assert "trigger_details" in flow["evidence_metrics"]


def test_expert_assessment_outputs_flow_decision(tmp_path: Path) -> None:
    details_csv = tmp_path / "details.csv"
    _write_details(details_csv)

    result = evaluate_expert_model_assessment(
        {"ITD_agent": {"planning": {"roi_extraction": {"enabled": True, "top_k": 1}}}},
        metrics={
            "tree_count_error_ratio": 0.10,
            "mean_crown_width_error_ratio": 0.20,
            "closure_error_abs": 0.05,
            "density_error_abs": 20,
            "expected_density": 100,
        },
        metrics_json=str(tmp_path / "metrics.json"),
        details_csv=str(details_csv),
        round_idx=1,
        previous_score=0.5,
    )

    flow = result["flow_decision"]
    assert flow["decision_stage"] == "expert_model_acceptance"
    assert flow["decision_question"] == "专家模型或局部细化结果是否优于旧结果？"
    assert "current_score" in flow["core_metrics"]
    assert "details_summary" in flow["evidence_metrics"]


def test_final_reference_assessment_outputs_flow_decision() -> None:
    result = evaluate_reference_quality_result(
        {
            "metrics": {
                "tree_count_error_ratio": 0.05,
                "mean_crown_width_error_ratio": 0.15,
                "closure_error_abs": 0.06,
                "density_error_abs": 30,
            }
        }
    )

    flow = result["flow_decision"]
    assert flow["decision_stage"] == "final_result_assessment"
    assert flow["decision_question"] == "最终结果质量如何？"
    assert flow["core_metrics"]["tree_count_error_ratio"] == 0.05
    assert "overall_score" in flow["core_metrics"]
    assert "reference_error_score" in flow["core_metrics"]
    assert "reference_quality_score" in flow["core_metrics"]
    assert "selected_metrics" in flow["evidence_metrics"]
    assert result["decision_flags"]["quality_pass_flag"] is False


def test_finetune_effect_outputs_flow_decision(tmp_path: Path) -> None:
    before_csv = tmp_path / "before.csv"
    after_csv = tmp_path / "after.csv"
    before_csv.write_text(
        "xiaoban_id,tree_count_error_abs,mean_crown_width_error_abs,closure_error_abs,density_error_abs\n"
        "1,10,3,0.2,100\n",
        encoding="utf-8",
    )
    after_csv.write_text(
        "xiaoban_id,tree_count_error_abs,mean_crown_width_error_abs,closure_error_abs,density_error_abs\n"
        "1,4,1,0.1,60\n",
        encoding="utf-8",
    )

    result = compare_finetune_effect(before_csv=str(before_csv), after_csv=str(after_csv), out_dir=str(tmp_path / "out"))

    flow = result["flow_decision"]
    assert flow["decision_stage"] == "finetune_effect_assessment"
    assert flow["decision_question"] == "微调后是否真的提升？"
    assert flow["core_metrics"]["mean_gain_tree_count"] == 6.0
    assert flow["core_metrics"]["accepted_improvement_flag"] is True
    assert "stratified_gain" in flow["evidence_metrics"]
