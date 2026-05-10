from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class V2WriteAction(str, Enum):
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
    "allow_training_trigger": "V2 cannot start training jobs.",
    "allow_weight_update": "V2 cannot update model weights.",
    "allow_model_promotion": "V2 cannot promote models.",
    "allow_active_skill_policy": "V2 cannot activate hard skill policies.",
    "allow_routing_policy_update": "V2 cannot update active routing policy.",
    "allow_expert_to_main_distillation": "V2 cannot start expert-to-main distillation.",
}


ACTION_FLAG = {
    V2WriteAction.WRITE_MEMORY: "allow_memory_write",
    V2WriteAction.WRITE_SKILL_DRAFT: "allow_skill_draft_write",
    V2WriteAction.WRITE_SKILL_ACTIVE_POLICY: "allow_active_skill_policy",
    V2WriteAction.WRITE_FINETUNE_SAMPLE: "allow_finetune_sample_write",
    V2WriteAction.EXPORT_FINETUNE_BUNDLE: "allow_finetune_bundle_export",
    V2WriteAction.START_TRAINING_JOB: "allow_training_trigger",
    V2WriteAction.UPDATE_MODEL_WEIGHT: "allow_weight_update",
    V2WriteAction.PROMOTE_MODEL: "allow_model_promotion",
    V2WriteAction.UPDATE_ROUTING_POLICY: "allow_routing_policy_update",
    V2WriteAction.START_DISTILLATION_JOB: "allow_expert_to_main_distillation",
}


def assert_v2_guardrails(cfg: dict[str, Any]) -> None:
    guardrails = cfg.get("guardrails") or {}
    for key, message in FORBIDDEN_FLAGS.items():
        if bool(guardrails.get(key, False)):
            raise ValueError(f"{message} Set `{key}` to false for V2.")


def check_write_action(action: V2WriteAction | str, cfg: dict[str, Any]) -> GuardrailResult:
    action_enum = V2WriteAction(action)
    flag = ACTION_FLAG[action_enum]
    guardrails = cfg.get("guardrails") or {}
    default_allowed = action_enum in {
        V2WriteAction.WRITE_MEMORY,
        V2WriteAction.WRITE_SKILL_DRAFT,
        V2WriteAction.WRITE_FINETUNE_SAMPLE,
        V2WriteAction.EXPORT_FINETUNE_BUNDLE,
    }
    allowed = bool(guardrails.get(flag, default_allowed))
    if action_enum in {V2WriteAction.WRITE_SKILL_ACTIVE_POLICY, V2WriteAction.START_TRAINING_JOB, V2WriteAction.UPDATE_MODEL_WEIGHT, V2WriteAction.PROMOTE_MODEL, V2WriteAction.UPDATE_ROUTING_POLICY, V2WriteAction.START_DISTILLATION_JOB}:
        allowed = False
    return GuardrailResult(
        allowed=allowed,
        action=action_enum.value,
        reason="allowed_by_v2_guardrail" if allowed else f"blocked_by_v2_guardrail:{flag}",
    )
