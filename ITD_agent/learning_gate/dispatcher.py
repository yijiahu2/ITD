from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import write_json
from ITD_agent.learning_gate.evidence_gate import decide_learning_event


def dispatch_learning_events(
    *,
    events: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    decisions = [decide_learning_event(event) for event in events]
    report = {
        "event_count": len(events),
        "decisions": decisions,
        "allow_write_memory": bool(cfg.get("allow_write_memory", False)),
        "allow_write_skill_draft": bool(cfg.get("allow_write_skill_draft", False)),
        "allow_write_finetune_pool": bool(cfg.get("allow_write_finetune_pool", False)),
    }
    report_path = write_json(Path(output_dir) / "learning_gate" / "dispatch_report.json", report)
    return {"event_count": len(events), "report_path": report_path, "report": report}
