from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.parameter_engine.expert_search_space import get_expert_model_search_space


def test_mask_scoring_search_space_has_boundary_specific_params() -> None:
    space = get_expert_model_search_space("mmdet_mask_scoring_rcnn")

    assert space["model_family"] == "mmdet_mask_scoring_rcnn"
    assert space["body_params"]["mask_iou_head"]["fixed"] is True
    assert "boundary_height_alignment_thr" in space["decision_params"]
    assert "boundary_calibration" in space["best_for"]


def test_mask2former_search_space_has_query_params() -> None:
    space = get_expert_model_search_space("mmdet_mask2former")

    assert "num_queries" in space["body_params"]
    assert "topk" in space["decision_params"]
    assert "resize_policy" in space["deployment_params"]
