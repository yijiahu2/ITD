from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _expert_family_prior(row: dict[str, Any]) -> list[str]:
    forest_type = str(row.get("forest_type_en") or "").lower()
    domain = str(row.get("forest_domain") or "").lower()
    resolution_min = float(row.get("resolution_cm_min") or 0.0)
    priors: list[str] = []
    if "evergreen" in forest_type or "rainforest" in forest_type or "moist" in forest_type:
        priors.append("dense_adhesion")
    if "montane" in forest_type:
        priors.append("shadow_topography")
    if "savanna" in forest_type or "urban" in domain:
        priors.append("large_crown_over_split")
    if resolution_min <= 3.0:
        priors.append("boundary_calibration")
    if not priors:
        priors.append("cross_domain_generalist")
    return list(dict.fromkeys(priors))


def _parameter_prior(row: dict[str, Any]) -> dict[str, Any]:
    resolution_cm = float(row.get("resolution_cm_min") or row.get("resolution_cm_max") or 10.0)
    if resolution_cm <= 3.0:
        return {"tile_size": 1536, "tile_overlap": 256, "score_thr": 0.20, "min_area_px": 80}
    if resolution_cm <= 5.0:
        return {"tile_size": 1536, "tile_overlap": 256, "score_thr": 0.18, "min_area_px": 60}
    return {"tile_size": 1280, "tile_overlap": 192, "score_thr": 0.16, "min_area_px": 40}


def build_public_dataset_knowledge_index(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets") or {}
    rows = []
    for key, raw in datasets.items():
        row = dict(raw or {})
        row["metadata_key"] = key
        row["recommended_expert_families"] = _expert_family_prior(row)
        row["parameter_prior"] = _parameter_prior(row)
        rows.append(row)

    by_domain: dict[str, list[str]] = {}
    for row in rows:
        domain = str(row.get("forest_type_en") or "unknown")
        by_domain.setdefault(domain, []).append(str(row.get("dataset_key") or row.get("metadata_key")))

    return {
        "source_metadata": str(path),
        "dataset_count": len(rows),
        "datasets": rows,
        "by_forest_type": by_domain,
        "usage": {
            "online_prompt_policy": "use_as_compressed_prior_digest_only",
            "training_policy": "use_for_expert_model_training_and_benchmark",
            "parameter_policy": "use_for_initial_template_prior_not_final_online_decision",
        },
    }
