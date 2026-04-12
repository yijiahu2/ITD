from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, load_json


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _default_wrapper(expert_family: str) -> dict[str, Any]:
    family = str(expert_family or "cross_domain_generalist").strip().lower()
    if family in {"dense_adhesion", "shadow_topography", "cross_domain_generalist"}:
        return {"type": "RepeatDataset", "times": 2 if family != "cross_domain_generalist" else 1}
    return {"type": "ClassBalancedDataset", "oversample_thr": 0.001}


def build_training_injection_manifest(
    *,
    cfg: dict[str, Any],
    dataset_summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    expert_strategy = dict(cfg.get("expert_training_strategy") or {})
    knowledge_strategy = dict(cfg.get("knowledge_injection_strategy") or {})
    target_expert_family = str(
        cfg.get("target_expert_family")
        or expert_strategy.get("target_expert_family")
        or knowledge_strategy.get("target_expert_family")
        or "cross_domain_generalist"
    )
    bundle_path = cfg.get("finetune_dataset_bundle_path") or cfg.get("dataset_bundle_path")
    bundle = load_json(bundle_path) if bundle_path and Path(str(bundle_path)).exists() else {}
    manifest = {
        "target_model_role": str(cfg.get("target_model_role") or "main_model"),
        "target_expert_family": target_expert_family,
        "segmentation_algorithm": str(cfg.get("segmentation_algorithm") or expert_strategy.get("segmentation_algorithm") or ""),
        "dataset_wrapper": expert_strategy.get("dataset_wrapper") or _default_wrapper(target_expert_family),
        "curriculum_mode": str(expert_strategy.get("curriculum_mode") or knowledge_strategy.get("curriculum_mode") or "mixed_domain"),
        "prior_axes": _normalize_list(expert_strategy.get("prior_axes") or knowledge_strategy.get("prior_axes")),
        "replay_ratio": float(expert_strategy.get("replay_ratio") or knowledge_strategy.get("replay_ratio") or 0.0),
        "hard_case_ratio": float(expert_strategy.get("hard_case_ratio") or knowledge_strategy.get("hard_case_ratio") or 0.0),
        "bundle_selection_summary": bundle.get("selection_summary") or {},
        "dataset_counts": dict((dataset_summary or {}).get("counts") or {}),
        "knowledge_injection_strategy": knowledge_strategy,
    }
    manifest_path = Path(output_dir) / "expert_injection_manifest.json"
    dump_json(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
