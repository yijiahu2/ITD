from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_training_evolution_feedback(
    *,
    training_summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir) / "evolution_feedback"
    root.mkdir(parents=True, exist_ok=True)

    promotion = training_summary.get("promotion_decision") or {}
    sample_quality = training_summary.get("sample_quality_report") or {}
    model_version = training_summary.get("model_version") or {}

    feedback = {
        "feedback_type": "training_evolution_feedback",
        "training_status": (training_summary.get("training_result") or {}).get("status"),
        "promotion_decision": promotion.get("decision"),
        "model_version_id": model_version.get("model_version_id"),
        "accepted_samples": sample_quality.get("accepted_count"),
        "recommended_memory_write": True,
        "recommended_skill_update": promotion.get("decision") in {"promote_to_shadow"},
        "recommended_routing_review": promotion.get("decision") in {"promote_to_shadow", "keep_candidate"},
    }

    path = root / "training_evolution_feedback.json"
    path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"feedback_path": str(path), "feedback": feedback}
