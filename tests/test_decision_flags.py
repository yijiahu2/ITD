from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.decision_flags import build_decision_flags


def test_build_decision_flags_for_reference_quality() -> None:
    result = {
        "selected_metrics": {
            "tree_count_error_ratio": 0.05,
            "mean_crown_width_error_ratio": 0.10,
            "closure_error_abs": 0.04,
            "density_error_abs": 20.0,
            "expected_density": 500.0,
        },
        "online_quality": {
            "quality_score": 0.10,
            "metrics": {
                "geometry_diagnostics": {
                    "semantic_instance_conflict_flag": False,
                }
            },
        },
        "candidate_roi_count": 0,
    }

    flags = build_decision_flags(result)

    assert flags["overall_score"] is not None
    assert flags["quality_pass_flag"] is True
    assert flags["need_local_refine_flag"] is False
    assert flags["need_manual_review_flag"] is False


def test_build_decision_flags_for_benchmark_with_manual_review() -> None:
    result = {
        "evaluation_mode": "benchmark",
        "ap50": 0.40,
        "ap75": 0.20,
        "f1_score50": 0.50,
        "error_decomposition": {
            "failure_severity": 0.80,
            "failure_pattern_confidence": 0.20,
        },
    }

    flags = build_decision_flags(result)

    assert flags["overall_score"] == pytest.approx(0.355)
    assert flags["quality_pass_flag"] is False
    assert flags["need_param_search_flag"] is True
    assert flags["need_finetune_flag"] is True
    assert flags["need_manual_review_flag"] is True


def test_build_decision_flags_reference_skips_density_when_expected_density_missing() -> None:
    result = {
        "selected_metrics": {
            "tree_count_error_ratio": 0.05,
            "mean_crown_width_error_ratio": 0.10,
            "closure_error_abs": 0.04,
            "density_error_abs": 2000.0,
        },
        "online_quality": {
            "quality_score": 0.10,
            "metrics": {
                "geometry_diagnostics": {
                    "semantic_instance_conflict_flag": False,
                }
            },
        },
    }

    flags = build_decision_flags(result)

    assert flags["overall_score"] is not None
    assert flags["quality_pass_flag"] is True
