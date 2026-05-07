from __future__ import annotations

from copy import deepcopy
from typing import Any


_COMMON_DEPLOYMENT_PARAMS = {
    "tile_size": {"type": "int", "values": [1024, 1280, 1536, 1792, 2048]},
    "tile_overlap": {"type": "int", "values": [128, 192, 256, 320, 384]},
    "tile_batch_size": {"type": "int", "values": [1, 2, 4]},
}


_COMMON_DECISION_PARAMS = {
    "score_thr": {"type": "float", "range": [0.10, 0.45]},
    "merge_iou_thr": {"type": "float", "range": [0.25, 0.60]},
    "min_area_px": {"type": "int", "values": [30, 50, 80, 120, 180]},
    "min_sem_overlap_ratio": {"type": "float", "range": [0.0, 0.12]},
    "clip_to_msem": {"type": "bool", "values": [True, False]},
    "height_support_gate": {"type": "float", "range": [0.0, 0.70]},
}


_SPACES: dict[str, dict[str, Any]] = {
    "mmdet_htc": {
        "model_family": "mmdet_htc",
        "body_params": {
            "cascade_stages": {"fixed": 3},
            "roi_head": {"fixed": "htc"},
            "mask_head": {"fixed": "cascade_mask_head"},
        },
        "decision_params": {
            **_COMMON_DECISION_PARAMS,
            "mask_thr_binary": {"type": "float", "range": [0.35, 0.65]},
        },
        "deployment_params": deepcopy(_COMMON_DEPLOYMENT_PARAMS),
        "best_for": ["dense_adhesion", "shadow_topography"],
    },
    "mmdet_cascade_mask_rcnn": {
        "model_family": "mmdet_cascade_mask_rcnn",
        "body_params": {
            "cascade_stages": {"fixed": 3},
            "stage_iou_policy": {"fixed": "progressive_refinement"},
        },
        "decision_params": {
            **_COMMON_DECISION_PARAMS,
            "mask_thr_binary": {"type": "float", "range": [0.40, 0.70]},
        },
        "deployment_params": deepcopy(_COMMON_DEPLOYMENT_PARAMS),
        "best_for": ["large_crown_over_split"],
    },
    "mmdet_mask_scoring_rcnn": {
        "model_family": "mmdet_mask_scoring_rcnn",
        "body_params": {
            "mask_iou_head": {"fixed": True},
            "mask_score_fusion": {"fixed": True},
        },
        "decision_params": {
            **_COMMON_DECISION_PARAMS,
            "mask_thr_binary": {"type": "float", "range": [0.35, 0.70]},
            "mask_quality_thr": {"type": "float", "range": [0.15, 0.60]},
            "boundary_height_alignment_thr": {"type": "float", "range": [0.0, 0.80]},
        },
        "deployment_params": deepcopy(_COMMON_DEPLOYMENT_PARAMS),
        "best_for": ["boundary_calibration", "large_crown_over_split"],
    },
    "mmdet_mask2former": {
        "model_family": "mmdet_mask2former",
        "body_params": {
            "num_queries": {"type": "int", "values": [100, 200, 300]},
            "query_competition": {"fixed": "transformer_decoder"},
            "loss_family": {"fixed": ["loss_cls", "loss_mask", "loss_dice"]},
        },
        "decision_params": {
            **_COMMON_DECISION_PARAMS,
            "mask_thr_binary": {"type": "float", "range": [0.35, 0.65]},
            "topk": {"type": "int", "values": [100, 200, 300]},
        },
        "deployment_params": {
            **deepcopy(_COMMON_DEPLOYMENT_PARAMS),
            "resize_policy": {"type": "enum", "values": ["lsj", "keep_ratio", "fixed_1024"]},
        },
        "best_for": ["shadow_topography", "cross_domain_generalist"],
    },
}


def get_expert_model_search_space(algorithm_name: str) -> dict[str, Any]:
    key = str(algorithm_name or "").strip().lower()
    if key not in _SPACES:
        known = ", ".join(sorted(_SPACES))
        raise KeyError(f"Unknown expert model search space: {algorithm_name}. Known: {known}")
    return deepcopy(_SPACES[key])


def list_expert_model_search_spaces() -> list[str]:
    return sorted(_SPACES)
