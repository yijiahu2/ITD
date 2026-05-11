from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.common.json_store import write_json
from ITD_agent.finetune_pool.query import load_finetune_pool_snapshot, load_public_dataset_candidates, load_recent_finetune_pool_samples
from ITD_agent.finetune_pool.store import DEFAULT_FINETUNE_POOL_ROOT
from ITD_agent.model_roles import EXPERT_MODEL_ROLE, normalize_model_role


def _infer_target_model_role(summary: dict[str, Any], finetune_plan: dict[str, Any], snapshot: dict[str, Any]) -> str:
    return normalize_model_role(
        finetune_plan.get("target_model_role")
        or snapshot.get("recommended_target_model_role")
        or (EXPERT_MODEL_ROLE if ((summary.get("planning_scheduler") or {}).get("roi_rounds") or []) else "main_model")
    )


def _infer_target_expert_family(summary: dict[str, Any], finetune_plan: dict[str, Any], snapshot: dict[str, Any]) -> str:
    return str(
        finetune_plan.get("target_expert_family")
        or snapshot.get("recommended_target_expert_family")
        or "cross_domain_generalist"
    )


def _supports_failure_category(candidate: dict[str, Any], failure_category: str | None) -> bool:
    supported = candidate.get("supported_failure_categories") or []
    if not failure_category:
        return True
    if not supported:
        return True
    return failure_category in supported


def _supports_expert_family(candidate: dict[str, Any], expert_family: str | None) -> bool:
    if not expert_family:
        return True
    candidate_family = str(candidate.get("target_expert_family") or "").strip()
    if not candidate_family:
        return True
    return candidate_family == expert_family


def export_finetune_dataset_bundle(
    *,
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any],
    finetune_plan: dict[str, Any],
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(finetune_pool_root)
    snapshot = load_finetune_pool_snapshot(finetune_pool_root=root)
    all_samples = load_recent_finetune_pool_samples(finetune_pool_root=root, limit=200)
    public_candidates = load_public_dataset_candidates(finetune_pool_root=root, limit=200)

    run_name = str(summary.get("run_name") or runtime_cfg.get("run_name") or "unknown_run")
    target_role = _infer_target_model_role(summary, finetune_plan, snapshot)
    target_expert_family = _infer_target_expert_family(summary, finetune_plan, snapshot)
    failure_category = str(
        finetune_plan.get("failure_category")
        or snapshot.get("recommended_failure_category")
        or ""
    ).strip() or None
    target_module = str(finetune_plan.get("target_module") or "segmentation_model")

    relevant_samples = [
        item
        for item in all_samples
        if normalize_model_role(item.get("target_model_role"), default="main_model") == target_role
        and _supports_expert_family(item, target_expert_family)
    ]
    if failure_category:
        relevant_samples = [
            item
            for item in relevant_samples
            if str(item.get("failure_category") or "") == failure_category or str(item.get("source_type")) == "replay_good_sample"
        ]

    training_ready_samples: list[dict[str, Any]] = []
    weak_supervision_candidates: list[dict[str, Any]] = []
    label_preparation_queue: list[dict[str, Any]] = []
    replay_samples: list[dict[str, Any]] = []
    for item in relevant_samples:
        source_type = str(item.get("source_type") or "unknown")
        label_status = str(item.get("label_status") or "unknown")
        if source_type == "replay_good_sample":
            replay_samples.append(item)
            training_ready_samples.append(item)
            continue
        if label_status in {"manual", "pseudo"} and bool(item.get("ready_for_training")):
            weak_supervision_candidates.append(item)
            training_ready_samples.append(item)
            continue
        label_preparation_queue.append(item)

    selected_public_candidates = [
        item
        for item in public_candidates
        if normalize_model_role(item.get("target_model_role"), default="main_model") == target_role
        and _supports_expert_family(item, target_expert_family)
        and _supports_failure_category(item, failure_category)
    ][:20]

    bundle = {
        "run_name": run_name,
        "target_module": target_module,
        "target_model_role": target_role,
        "target_expert_family": target_expert_family,
        "failure_category": failure_category,
        "supervision_mode": finetune_plan.get("supervision_mode") or "hybrid",
        "selection_summary": {
            "training_ready_sample_count": len(training_ready_samples),
            "weak_supervision_candidate_count": len(weak_supervision_candidates),
            "label_preparation_queue_count": len(label_preparation_queue),
            "replay_sample_count": len(replay_samples),
            "public_dataset_candidate_count": len(selected_public_candidates),
        },
        "training_ready_samples": training_ready_samples,
        "weak_supervision_candidates": weak_supervision_candidates,
        "label_preparation_queue": label_preparation_queue,
        "replay_samples": replay_samples,
        "public_dataset_candidates": selected_public_candidates,
        "next_actions": [
            "优先使用 replay_good_sample 和人工/伪标签已就绪样本构建训练集。",
            "failed_roi_sample 与 hard_case_sample 进入标注/伪标签生成队列，完成标签准备后再纳入训练。",
            "将公开数据集候选与本地样本混合，用于提升泛化并防止灾难性遗忘。",
        ],
        "source_snapshot": snapshot,
    }

    if output_path is None:
        output_path = Path(runtime_cfg.get("output_dir") or ".").resolve() / "finetune" / "finetune_dataset_bundle.json"
    bundle_path = write_json(Path(output_path), bundle)
    return {
        "dataset_bundle_path": bundle_path,
        "selection_summary": bundle["selection_summary"],
        "target_model_role": target_role,
        "target_expert_family": target_expert_family,
        "failure_category": failure_category,
        "supervision_mode": bundle["supervision_mode"],
    }
