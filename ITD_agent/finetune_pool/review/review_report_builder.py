from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import write_csv, write_json


def build_review_report(
    *,
    review_run_id: str,
    source_run_id: str,
    output_dir: Path,
    integrity_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    asset_counts: dict[str, int],
) -> dict[str, Any]:
    decision_counts = Counter(str(item.get("decision")) for item in decisions)
    candidate_counts = Counter(str(item.get("candidate_type")) for item in decisions)
    guardrail_blocks = sum(1 for item in decisions if item.get("candidate_type") == "guardrail_probe" and item.get("decision") == "reject")
    invalid_count = sum(1 for row in integrity_rows if not row.get("valid"))
    missing_artifact_count = sum(len(row.get("missing_artifacts") or []) for row in integrity_rows)
    report = {
        "review_run_id": review_run_id,
        "source_run_id": source_run_id,
        "output_dir": str(output_dir),
        "trajectory_count": len(integrity_rows),
        "invalid_trajectories": invalid_count,
        "missing_artifacts": missing_artifact_count,
        "decision_counts": dict(decision_counts),
        "candidate_counts": dict(candidate_counts),
        "guardrail_blocks": guardrail_blocks,
        "asset_counts": asset_counts,
    }
    write_json(output_dir / "reports" / "review_summary.json", report)
    write_csv(output_dir / "reports" / "review_summary.csv", [_flatten_report(report)])
    write_json(output_dir / "reports" / "asset_summary.json", asset_counts)
    write_json(output_dir / "reports" / "error_summary.json", {"invalid_trajectories": invalid_count, "missing_artifacts": missing_artifact_count})
    return report


def _flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in report.items() if not isinstance(v, dict)}
    for prefix in ["decision_counts", "candidate_counts", "asset_counts"]:
        for key, value in (report.get(prefix) or {}).items():
            flat[f"{prefix}.{key}"] = value
    return flat
