from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ReviewWriteAction(str, Enum):
    WRITE_MEMORY = "write_memory"
    WRITE_SKILL_DRAFT = "write_skill_draft"
    WRITE_SKILL_ACTIVE_POLICY = "write_skill_active_policy"
    WRITE_FINETUNE_SAMPLE = "write_finetune_sample"
    EXPORT_FINETUNE_BUNDLE = "export_finetune_bundle"
    START_TRAINING_JOB = "start_training_job"
    UPDATE_MODEL_WEIGHT = "update_model_weight"
    PROMOTE_MODEL = "promote_model"
    UPDATE_ROUTING_POLICY = "update_routing_policy"
    START_DISTILLATION_JOB = "start_distillation_job"


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORBIDDEN_FLAGS = {
    "allow_training_trigger": "Review cannot start training jobs.",
    "allow_weight_update": "Review cannot update model weights.",
    "allow_model_promotion": "Review cannot promote models.",
    "allow_active_skill_policy": "Review cannot activate hard skill policies.",
    "allow_routing_policy_update": "Review cannot update active routing policy.",
    "allow_expert_to_main_distillation": "Review cannot start expert-to-main distillation.",
}


ACTION_FLAG = {
    ReviewWriteAction.WRITE_MEMORY: "allow_memory_write",
    ReviewWriteAction.WRITE_SKILL_DRAFT: "allow_skill_draft_write",
    ReviewWriteAction.WRITE_SKILL_ACTIVE_POLICY: "allow_active_skill_policy",
    ReviewWriteAction.WRITE_FINETUNE_SAMPLE: "allow_finetune_sample_write",
    ReviewWriteAction.EXPORT_FINETUNE_BUNDLE: "allow_finetune_bundle_export",
    ReviewWriteAction.START_TRAINING_JOB: "allow_training_trigger",
    ReviewWriteAction.UPDATE_MODEL_WEIGHT: "allow_weight_update",
    ReviewWriteAction.PROMOTE_MODEL: "allow_model_promotion",
    ReviewWriteAction.UPDATE_ROUTING_POLICY: "allow_routing_policy_update",
    ReviewWriteAction.START_DISTILLATION_JOB: "allow_expert_to_main_distillation",
}


def assert_review_guardrails(cfg: dict[str, Any]) -> None:
    guardrails = cfg.get("guardrails") or {}
    for key, message in FORBIDDEN_FLAGS.items():
        if bool(guardrails.get(key, False)):
            raise ValueError(f"{message} Set `{key}` to false for review.")


def check_write_action(action: ReviewWriteAction | str, cfg: dict[str, Any]) -> GuardrailResult:
    action_enum = ReviewWriteAction(action)
    flag = ACTION_FLAG[action_enum]
    guardrails = cfg.get("guardrails") or {}
    default_allowed = action_enum in {
        ReviewWriteAction.WRITE_MEMORY,
        ReviewWriteAction.WRITE_SKILL_DRAFT,
        ReviewWriteAction.WRITE_FINETUNE_SAMPLE,
        ReviewWriteAction.EXPORT_FINETUNE_BUNDLE,
    }
    allowed = bool(guardrails.get(flag, default_allowed))
    if action_enum in {ReviewWriteAction.WRITE_SKILL_ACTIVE_POLICY, ReviewWriteAction.START_TRAINING_JOB, ReviewWriteAction.UPDATE_MODEL_WEIGHT, ReviewWriteAction.PROMOTE_MODEL, ReviewWriteAction.UPDATE_ROUTING_POLICY, ReviewWriteAction.START_DISTILLATION_JOB}:
        allowed = False
    return GuardrailResult(
        allowed=allowed,
        action=action_enum.value,
        reason="allowed_by_review_guardrail" if allowed else f"blocked_by_review_guardrail:{flag}",
    )
