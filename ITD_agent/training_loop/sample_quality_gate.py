from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.io_utils import write_json


def apply_sample_quality_gate(
    *,
    samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    target: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for sample in samples:
        result = _evaluate_sample(sample, cfg=cfg, target=target)
        item = {**sample, "quality_gate": result}
        if result["decision"] == "accept":
            accepted.append(item)
        elif result["decision"] == "defer":
            deferred.append(item)
        else:
            rejected.append(item)

    report = {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "deferred_count": len(deferred),
        "accepted_by_failure_category": dict(Counter(str(item.get("failure_category") or "unknown") for item in accepted)),
        "rejected_reasons": dict(Counter(reason for item in rejected for reason in (item.get("quality_gate") or {}).get("reasons", []))),
        "deferred_reasons": dict(Counter(reason for item in deferred for reason in (item.get("quality_gate") or {}).get("reasons", []))),
    }
    out = Path(output_dir)
    write_json(out / "finetune_bundle" / "sample_quality_report.json", report)
    _write_rejected_csv(out / "finetune_bundle" / "rejected_samples.csv", rejected)
    return {
        "accepted_samples": accepted,
        "rejected_samples": rejected,
        "deferred_samples": deferred,
        "sample_quality_report": report,
    }


def _evaluate_sample(sample: dict[str, Any], *, cfg: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    label_status = str(sample.get("label_status") or "")
    artifact_refs = dict(sample.get("artifact_refs") or {})
    metadata = dict(sample.get("metadata") or {})
    roi = dict(metadata.get("roi") or {})
    target_role = str(target.get("target_model_role") or "")
    target_family = target.get("target_expert_family")
    failure_category = target.get("failure_category")

    if str(sample.get("target_model_role") or "") != target_role:
        reasons.append("target_model_role_mismatch")
    if target_family and str(sample.get("target_expert_family") or "") != str(target_family):
        reasons.append("target_expert_family_mismatch")
    if failure_category and str(sample.get("failure_category") or "") != str(failure_category):
        reasons.append("failure_category_mismatch")
    if not bool(sample.get("ready_for_training")):
        reasons.append("not_ready_for_training")
    if label_status == "manual" and not bool(cfg.get("allow_manual_labels", True)):
        reasons.append("manual_label_not_allowed")
    if label_status == "pseudo" and not bool(cfg.get("allow_pseudo_labels", True)):
        reasons.append("pseudo_label_not_allowed")
    if label_status not in {"manual", "pseudo"}:
        reasons.append("unsupported_label_status")
    if bool(cfg.get("reject_missing_artifacts", True)):
        for key in ["image", "gt_mask"]:
            path = artifact_refs.get(key)
            if not path or not Path(str(path)).exists():
                reasons.append(f"missing_{key}_artifact")
    bbox = roi.get("bbox_px") or []
    if bool(cfg.get("reject_empty_annotations", True)) and not _valid_bbox(bbox):
        reasons.append("invalid_or_empty_bbox")
    if bool(cfg.get("reject_invalid_masks", True)) and label_status not in {"manual", "pseudo"}:
        reasons.append("invalid_mask_label_status")
    if str(sample.get("source_type") or "") == "replay_good_sample":
        return {"decision": "defer", "reasons": ["replay_sample_reserved_for_replay_split"]}
    return {"decision": "reject" if reasons else "accept", "reasons": reasons or ["sample_quality_gate_passed"]}


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _write_rejected_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "failure_category", "target_model_role", "target_expert_family", "reasons"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            gate = row.get("quality_gate") or {}
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "failure_category": row.get("failure_category"),
                    "target_model_role": row.get("target_model_role"),
                    "target_expert_family": row.get("target_expert_family"),
                    "reasons": ";".join(gate.get("reasons") or []),
                }
            )
