from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_json, write_jsonl


def evaluate_replay_guard(*, evaluation: dict[str, Any], cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    guard_cfg = cfg.get("replay_guard") or {}
    delta = evaluation.get("delta") or {}
    metric_delta = delta.get("delta") if isinstance(delta, dict) else {}
    failures: list[dict[str, Any]] = []
    replay_count = int(((cfg.get("_v3_internal") or {}).get("replay_sample_count") or 0))
    if bool(guard_cfg.get("require_evaluated_delta", True)) and (not isinstance(delta, dict) or delta.get("status") != "computed"):
        failures.append({"check": "evaluated_delta", "reason": "before_after_metric_delta_not_computed"})
    if bool(guard_cfg.get("require_replay_samples", True)) and replay_count <= 0:
        failures.append({"check": "replay_samples", "reason": "no_replay_samples_available"})
    _check_drop(failures, metric_delta, "ap_50_95", guard_cfg.get("max_ap_drop", 0.01))
    _check_drop(failures, metric_delta, "ap50", guard_cfg.get("max_ap50_drop", 0.015))
    _check_drop(failures, metric_delta, "recall", guard_cfg.get("max_recall_drop", 0.02))
    _check_drop(failures, metric_delta, "precision", guard_cfg.get("max_precision_drop", 0.02))
    report = {
        "passed": not failures,
        "decision": "pass" if not failures else "fail",
        "failures": failures,
        "thresholds": guard_cfg,
        "evaluation_status": (evaluation.get("candidate") or {}).get("status"),
        "replay_sample_count": replay_count,
    }
    root = Path(output_dir) / "replay_guard"
    write_json(root / "replay_guard_report.json", report)
    write_jsonl(root / "replay_failed_cases.jsonl", failures)
    return report


def _check_drop(failures: list[dict[str, Any]], delta: Any, key: str, threshold: Any) -> None:
    if not isinstance(delta, dict) or delta.get(key) is None:
        return
    try:
        value = float(delta[key])
        limit = float(threshold)
    except (TypeError, ValueError):
        return
    if value < -limit:
        failures.append({"metric": key, "delta": value, "max_allowed_drop": limit})
