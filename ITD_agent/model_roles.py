from __future__ import annotations

from typing import Any


MAIN_MODEL_ROLE = "main_model"
EXPERT_MODEL_ROLE = "expert_model"
LEGACY_EXPERT_MODEL_ROLES = {"expert_model", "child", "expert", "sub"}


def normalize_model_role(value: Any, default: str = MAIN_MODEL_ROLE) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if normalized in {MAIN_MODEL_ROLE, "main", "primary"}:
        return MAIN_MODEL_ROLE
    if normalized in {EXPERT_MODEL_ROLE, "expert_model", *LEGACY_EXPERT_MODEL_ROLES}:
        return EXPERT_MODEL_ROLE
    return normalized


def is_expert_model_role(value: Any) -> bool:
    return normalize_model_role(value) == EXPERT_MODEL_ROLE
