from __future__ import annotations

from .contracts import TrainingCandidate, TrainingTriggerContext


def evaluate_dry_run_trigger(candidates: list[TrainingCandidate]) -> dict[str, object]:
    return {
        "trigger_training": False,
        "candidate_count": len(candidates),
        "reason": "Adaptive inference only marks training candidates; training decisions stay in training_loop.",
    }


def evaluate_training_trigger(
    context: TrainingTriggerContext,
    *,
    min_training_ready: int = 100,
    min_replay: int = 30,
    min_public_candidates: int = 0,
    allow_weak_supervision: bool = True,
    training_entry_available: bool = True,
    family_config_available: bool = True,
    max_single_trajectory_ratio: float = 0.5,
) -> dict[str, object]:
    reasons: list[str] = []
    decision = "approve_pilot"
    if context.training_ready_sample_count < min_training_ready:
        decision = "defer"
        reasons.append("training_ready_sample_count_below_threshold")
    if context.replay_sample_count < min_replay:
        decision = "defer"
        reasons.append("replay_sample_count_below_threshold")
    if context.public_dataset_candidate_count < min_public_candidates:
        decision = "defer"
        reasons.append("public_dataset_candidate_count_below_threshold")
    if not allow_weak_supervision and context.weak_supervision_candidate_count > 0:
        decision = "reject"
        reasons.append("weak_supervision_not_allowed")
    if not training_entry_available:
        decision = "reject"
        reasons.append("training_entry_unavailable")
    if not family_config_available:
        decision = "reject"
        reasons.append("target_expert_family_config_unavailable")

    concentration = (context.evidence.get("source_concentration") or {}) if isinstance(context.evidence, dict) else {}
    max_trajectory_ratio = float(concentration.get("max_source_trajectory_ratio") or 0.0)
    if max_trajectory_ratio > max_single_trajectory_ratio and decision == "approve_pilot":
        decision = "need_human_review"
        reasons.append("samples_over_concentrated_in_single_trajectory")

    return {
        "decision": decision,
        "approve_pilot": decision == "approve_pilot",
        "approve_formal": False,
        "reasons": reasons or ["pilot_trigger_conditions_passed"],
        "context": context.to_dict(),
        "thresholds": {
            "min_training_ready": min_training_ready,
            "min_replay": min_replay,
            "min_public_candidates": min_public_candidates,
            "allow_weak_supervision": allow_weak_supervision,
            "max_single_trajectory_ratio": max_single_trajectory_ratio,
        },
    }
