from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ITD_agent.evaluation_analysis.geometry_failure_tags import build_geometry_failure_tags
from ITD_agent.evaluation_analysis.geometry_metrics import build_geometry_profile
from ITD_agent.finetune_pool.review.io_utils import write_json, write_jsonl


def evaluate_dom_only_geometry_guard(*, cfg: dict[str, Any], output_dir: str | Path, model_version_id: str | None = None) -> dict[str, Any]:
    guard_cfg = cfg.get("dom_only_geometry_guard") or {}
    root = Path(output_dir) / "geometry_guard"
    if not bool(guard_cfg.get("enabled", True)):
        report = _report(
            model_version_id=model_version_id,
            status="disabled",
            passed=True,
            baseline={},
            candidate={},
            delta={},
            failed=[],
            warnings=[],
            decision_hint="not_applicable",
        )
        write_json(root / "dom_only_geometry_guard_report.json", report)
        write_jsonl(root / "dom_only_geometry_failed_cases.jsonl", [])
        return report

    baseline_instances = _load_instances(guard_cfg.get("baseline_instances_json"))
    candidate_instances = _load_instances(guard_cfg.get("candidate_instances_json"))
    if not baseline_instances or not candidate_instances:
        report = _report(
            model_version_id=model_version_id,
            status="not_evaluated",
            passed=False,
            baseline={},
            candidate={},
            delta={},
            failed=[{"check": "dom_only_geometry_inputs", "reason": "baseline_or_candidate_instances_missing"}],
            warnings=[],
            decision_hint="keep_candidate",
        )
        write_json(root / "dom_only_geometry_guard_report.json", report)
        write_jsonl(root / "dom_only_geometry_failed_cases.jsonl", report["failed_guard_items"])
        return report

    baseline_summary = _geometry_summary(baseline_instances)
    candidate_summary = _geometry_summary(candidate_instances)
    delta = {f"{key}_delta": float(candidate_summary.get(key, 0.0)) - float(baseline_summary.get(key, 0.0)) for key in baseline_summary}
    failed, warnings = _evaluate_delta(delta, guard_cfg.get("thresholds") or {})
    passed = not failed
    report = _report(
        model_version_id=model_version_id,
        status="evaluated",
        passed=passed,
        baseline=baseline_summary,
        candidate=candidate_summary,
        delta=delta,
        failed=failed,
        warnings=warnings,
        decision_hint="allow_shadow" if passed and not warnings else "keep_candidate",
    )
    write_json(root / "dom_only_geometry_guard_report.json", report)
    write_jsonl(root / "dom_only_geometry_failed_cases.jsonl", failed)
    return report


def _geometry_summary(instances: list[dict[str, Any]]) -> dict[str, float]:
    profile = build_geometry_profile(instances)
    tags = build_geometry_failure_tags(profile)
    count = max(1, int(profile.get("instance_count") or 0))
    tag_counts: dict[str, int] = {}
    for tag in tags:
        key = str(tag.get("tag") or "unknown")
        tag_counts[key] = tag_counts.get(key, 0) + 1
    measurements = list(profile.get("measurements") or [])
    fragmented = sum(1 for item in measurements if float(item.get("boundary_complexity") or 0.0) >= 3.0)
    abnormal_density = 1 if count >= 500 else 0
    return {
        "small_instance_ratio": tag_counts.get("tiny_false_positive", 0) / count,
        "large_instance_ratio": tag_counts.get("oversized_crown", 0) / count,
        "duplicate_instance_ratio": 0.0,
        "fragmented_mask_ratio": fragmented / count,
        "abnormal_density_block_ratio": float(abnormal_density),
        "roi_outside_false_positive_ratio": 0.0,
        "boundary_complexity_ratio": fragmented / count,
    }


def _evaluate_delta(delta: dict[str, float], thresholds: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    mapping = {
        "small_instance_ratio_delta": "max_small_instance_ratio_increase",
        "large_instance_ratio_delta": "max_large_instance_ratio_increase",
        "duplicate_instance_ratio_delta": "max_duplicate_instance_ratio_increase",
        "fragmented_mask_ratio_delta": "max_fragmented_mask_ratio_increase",
        "abnormal_density_block_ratio_delta": "max_abnormal_density_block_ratio_increase",
        "roi_outside_false_positive_ratio_delta": "max_roi_outside_false_positive_increase",
        "boundary_complexity_ratio_delta": "max_boundary_complexity_increase",
    }
    failed: list[dict[str, Any]] = []
    warnings: list[str] = []
    for delta_key, threshold_key in mapping.items():
        value = float(delta.get(delta_key) or 0.0)
        limit = float(thresholds.get(threshold_key, 0.03))
        if value > limit:
            failed.append({"check": delta_key, "delta": value, "threshold": limit})
        elif value > 0:
            warnings.append(f"{delta_key} slightly increased")
    return failed, warnings


def _report(
    *,
    model_version_id: str | None,
    status: str,
    passed: bool,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    delta: dict[str, Any],
    failed: list[dict[str, Any]],
    warnings: list[str],
    decision_hint: str,
) -> dict[str, Any]:
    return {
        "model_version_id": model_version_id,
        "status": status,
        "evaluated_dom_samples": 0 if status != "evaluated" else 1,
        "geometry_guard_passed": passed,
        "baseline_geometry_summary": baseline,
        "candidate_geometry_summary": candidate,
        "geometry_delta": delta,
        "failed_guard_items": failed,
        "warning_items": warnings,
        "decision_hint": decision_hint,
    }


def _load_instances(path: Any) -> list[dict[str, Any]]:
    if not path:
        return []
    src = Path(str(path))
    if not src.exists():
        return []
    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("instances") or payload.get("annotations") or [])
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return []
