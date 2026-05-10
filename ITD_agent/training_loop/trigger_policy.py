from __future__ import annotations

from .contracts import TrainingCandidate


def evaluate_dry_run_trigger(candidates: list[TrainingCandidate]) -> dict[str, object]:
    return {
        "trigger_training": False,
        "candidate_count": len(candidates),
        "reason": "V1 only supports dry-run training candidate intake.",
    }
