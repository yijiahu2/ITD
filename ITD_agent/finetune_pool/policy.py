from __future__ import annotations

from collections import Counter
from typing import Any

from ITD_agent.finetune_pool.clusterer import build_pool_clusters
from ITD_agent.finetune_pool.contracts import FinetuneTriggerSnapshot
from ITD_agent.model_roles import normalize_model_role


DEFAULT_POLICY = {
    "min_failed_roi_samples": 5,
    "min_ready_samples": 5,
    "min_public_dataset_candidates": 0,
}


def load_finetune_pool_policy(runtime_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    itd_cfg = (runtime_cfg or {}).get("ITD_agent") or {}
    planning_cfg = itd_cfg.get("planning") or {}
    finetune_policy = planning_cfg.get("finetune_pool_policy") or itd_cfg.get("finetune_pool_policy") or {}
    for key in DEFAULT_POLICY:
        if key in finetune_policy:
            policy[key] = finetune_policy[key]
    return policy


def infer_failure_category(case: dict[str, Any]) -> str:
    pred_count = case.get("pred_tree_count")
    exp_count = case.get("expected_tree_count")
    crown_abs = float(case.get("mean_crown_width_error_abs") or 0.0)
    closure_abs = float(case.get("closure_error_abs") or 0.0)
    slope_class = str(case.get("slope_class") or "").lower()
    aspect_class = str(case.get("aspect_class") or "").lower()
    landform = str(case.get("landform_type") or "").lower()
    pred_cover = float(case.get("pred_cover_ratio") or 0.0)
    exp_cover = float(case.get("expected_closure") or 0.0)

    if ("north" in aspect_class or "shadow" in aspect_class) and closure_abs >= 0.10:
        return "north_slope_shadow_confused"
    if ("steep" in slope_class or "mountain" in landform or "ridge" in landform) and closure_abs >= 0.12:
        return "terrain_shadow_confused"
    if pred_count is not None and exp_count is not None:
        try:
            if float(pred_count) < float(exp_count) * 0.85:
                if closure_abs >= 0.10:
                    return "count_underestimate_dense"
                return "count_underestimate"
            if float(pred_count) > float(exp_count) * 1.15:
                if crown_abs >= 1.0:
                    return "over_split_large_crown"
                return "count_overestimate"
        except Exception:
            pass
    if exp_cover >= 0.55 and pred_cover < exp_cover and crown_abs >= 1.0:
        return "dense_canopy_adhesion"
    if closure_abs >= 0.12 and crown_abs >= 1.0:
        return "dense_canopy_adhesion"
    if crown_abs >= 1.2 and pred_cover > exp_cover:
        return "boundary_score_mismatch"
    if ("ridge" in landform or "mountain" in landform) and crown_abs >= 1.0:
        return "slope_boundary_distorted"
    if crown_abs >= 1.0:
        return "crown_boundary_fragmented"
    return "crown_overlap_conflict"


def build_finetune_trigger_snapshot(
    *,
    samples: list[dict[str, Any]],
    public_candidates: list[dict[str, Any]],
    run_name: str,
    timestamp: str,
    target_module: str | None = None,
    runtime_cfg: dict[str, Any] | None = None,
) -> FinetuneTriggerSnapshot:
    policy = load_finetune_pool_policy(runtime_cfg)
    clusters = build_pool_clusters(samples)
    source_counts = Counter(str(item.get("source_type") or "unknown") for item in samples)
    ready_counts = Counter(str(item.get("source_type") or "unknown") for item in samples if item.get("ready_for_training"))

    best_cluster = clusters[0] if clusters else None
    trigger_ready = False
    trigger_reason = "当前尚未达到微调触发阈值。"
    recommended_role = None
    recommended_family = None
    recommended_category = None
    if best_cluster:
        recommended_role = normalize_model_role(best_cluster.target_model_role)
        recommended_family = best_cluster.target_expert_family
        recommended_category = best_cluster.failure_category
        required_public_candidates = int(policy["min_public_dataset_candidates"])
        trigger_ready = (
            best_cluster.sample_count >= int(policy["min_failed_roi_samples"])
            and best_cluster.ready_sample_count >= int(policy["min_ready_samples"])
            and len(public_candidates) >= required_public_candidates
        )
        if trigger_ready:
            trigger_reason = (
                f"{normalize_model_role(best_cluster.target_model_role)}/{best_cluster.failure_category} 已累计 "
                f"{best_cluster.sample_count} 个样本，其中 {best_cluster.ready_sample_count} 个可训练。"
            )
            if required_public_candidates <= 0 and not public_candidates:
                trigger_reason += " 当前允许仅依赖本地失败样本触发微调闭环。"
        else:
            trigger_reason = (
                f"{normalize_model_role(best_cluster.target_model_role)}/{best_cluster.failure_category} 当前累计 "
                f"{best_cluster.sample_count} 个样本，尚未达到触发阈值。"
            )

    return FinetuneTriggerSnapshot(
        timestamp=timestamp,
        run_name=run_name,
        trigger_ready=trigger_ready,
        recommended_target_module=target_module,
        recommended_target_model_role=recommended_role,
        recommended_target_expert_family=recommended_family,
        recommended_failure_category=recommended_category,
        trigger_reason=trigger_reason,
        sample_counts=dict(source_counts),
        ready_counts=dict(ready_counts),
        cluster_summaries=[
            {
                "cluster_id": item.cluster_id,
                "target_model_role": normalize_model_role(item.target_model_role),
                "failure_category": item.failure_category,
                "sample_count": item.sample_count,
                "ready_sample_count": item.ready_sample_count,
                "source_types": item.source_types,
                "label_status_breakdown": item.label_status_breakdown,
                "tags": item.tags,
            }
            for item in clusters
        ],
        public_dataset_candidates=public_candidates[:10],
        metadata={"policy": policy},
    )
