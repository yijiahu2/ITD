from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_json


def build_expert_to_main_distillation_manifest(
    *,
    distillation_candidates: list[dict[str, Any]],
    output_dir: str | Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    enabled = bool((cfg.get("distillation") or {}).get("enabled", True))
    root = Path(output_dir) / "distillation"
    rows = []
    if enabled:
        min_tier = str((cfg.get("distillation") or {}).get("pseudo_label_quality_min") or "silver")
        allowed_tiers = _allowed_tiers(min_tier)
        for item in distillation_candidates:
            tier = str(item.get("quality_tier") or "silver")
            if tier not in allowed_tiers:
                continue
            if str(item.get("status") or "") not in {"candidate_only", "approved", "accepted"}:
                continue
            rows.append(
                {
                    "distillation_candidate_id": item.get("distillation_candidate_id"),
                    "source_run_id": item.get("source_run_id"),
                    "source_trajectory_id": item.get("source_trajectory_id"),
                    "source_roi_id": item.get("source_roi_id"),
                    "source_expert_model": item.get("expert_model"),
                    "quality_tier": tier,
                    "label_source": "gt_or_v2_pseudo_label",
                    "status": "manifest_only",
                }
            )
    csv_path = root / "main_model_distillation_manifest.csv"
    root.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else [
            "distillation_candidate_id",
            "source_run_id",
            "source_trajectory_id",
            "source_roi_id",
            "source_expert_model",
            "quality_tier",
            "label_source",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path = write_json(root / "main_model_distillation_manifest.json", rows)
    report = {
        "enabled": enabled,
        "candidate_count": len(distillation_candidates),
        "manifest_count": len(rows),
        "manifest_csv": str(csv_path),
        "manifest_json": json_path,
        "run_distillation_training": False,
    }
    write_json(root / "distillation_report.json", report)
    return report


def _allowed_tiers(min_tier: str) -> set[str]:
    if min_tier == "gold":
        return {"gold"}
    return {"gold", "silver"}
