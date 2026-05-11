from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ITD_agent.evolution.review.io_utils import load_structured, write_json
from ITD_agent.training_loop.contracts import TrainingPlan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_training_plan(
    *,
    cfg: dict[str, Any],
    trigger_context: dict[str, Any],
    family_cfg: dict[str, Any],
    dataset_bundle: dict[str, Any],
    output_dir: str | Path,
    training_mode: str = "pilot",
) -> TrainingPlan:
    out_root = Path(output_dir)
    job_id = _job_id(training_mode, str(family_cfg.get("algorithm_name") or trigger_context.get("target_model_id")))
    job_dir = out_root / "training_jobs" / f"{training_mode}_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    algorithm_name = str(family_cfg.get("algorithm_name") or trigger_context.get("target_model_id"))
    generated_config = job_dir / "generated_config.yaml"
    trainer_module = _trainer_module(algorithm_name)
    execute_training = bool((cfg.get("runner") or {}).get("execute_training", False))
    build_only = not execute_training or bool(((cfg.get("training") or {}).get(training_mode) or {}).get("build_only", False))
    command = [sys.executable, "-u", "-m", trainer_module, "--config", str(generated_config)]
    if build_only:
        command.append("--build-only")
    config_payload = _generated_training_config(
        cfg=cfg,
        trigger_context=trigger_context,
        family_cfg=family_cfg,
        dataset_bundle=dataset_bundle,
        job_dir=job_dir,
        training_mode=training_mode,
    )
    generated_config.write_text(yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    plan = TrainingPlan(
        training_job_id=job_id,
        training_mode=training_mode,
        target_model_role=str(trigger_context.get("target_model_role")),
        target_model_id=str(trigger_context.get("target_model_id")),
        algorithm_name=algorithm_name,
        target_expert_family=trigger_context.get("target_expert_family"),
        failure_category=trigger_context.get("failure_category"),
        source_config_path=str(family_cfg.get("source_config_path") or ""),
        generated_config_path=str(generated_config),
        output_dir=str(job_dir),
        command=command,
        expected_checkpoint_glob=str(job_dir / "**" / "*.pth"),
        metadata={
            "build_only": build_only,
            "dataset_bundle_dir": dataset_bundle.get("dataset_bundle_dir"),
            "family_training_defaults": family_cfg.get("training_defaults") or {},
        },
    )
    write_json(job_dir / "training_plan.json", plan.to_dict())
    command_text = " ".join(shlex.quote(part) for part in command)
    (job_dir / "command.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(PROJECT_ROOT))}",
                "export PYTHONNOUSERSITE=1",
                f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT))}:${{PYTHONPATH:-}}",
                command_text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return plan


def _generated_training_config(
    *,
    cfg: dict[str, Any],
    trigger_context: dict[str, Any],
    family_cfg: dict[str, Any],
    dataset_bundle: dict[str, Any],
    job_dir: Path,
    training_mode: str,
) -> dict[str, Any]:
    defaults = dict(family_cfg.get("training_defaults") or {})
    mode_cfg = dict(((cfg.get("training") or {}).get(training_mode) or {}))
    epochs = int(mode_cfg.get("override_epochs") or defaults.get("epochs") or 1)
    dataset_dir = Path(str(dataset_bundle["dataset_bundle_dir"]))
    annotations = dict(dataset_bundle.get("annotation_paths") or {})
    source_config_path = str(family_cfg.get("source_config_path") or "")
    template_payload = load_structured(source_config_path) if source_config_path and Path(source_config_path).exists() else {}
    payload = {
        **template_payload,
        "output_dir": str(job_dir),
        "run_name": str((cfg.get("source") or {}).get("run_id") or "v3_training"),
        "segmentation_algorithm": family_cfg.get("algorithm_name"),
        "segmentation_train_out_dirname": "segmentation_training",
        "segmentation_dataset_dirname": "external_segmentation_dataset",
        "public_dataset_root": str(dataset_dir),
        "public_dataset_annotation_files_by_role": {
            "train": annotations.get("train"),
            "val": annotations.get("val"),
            "test": annotations.get("test"),
        },
        "segmentation_class_names": ["crown"],
        "segmentation_num_classes": 1,
        "segmentation_train_epochs": epochs,
        "segmentation_train_batch_size": int(defaults.get("batch_size") or 1),
        "segmentation_train_num_workers": int(defaults.get("num_workers") or 0),
        "segmentation_train_lr": float(defaults.get("lr") or 0.0001),
        "segmentation_train_weight_decay": float(defaults.get("weight_decay") or 0.0001),
        "segmentation_eval_after_train": bool((cfg.get("evaluation") or {}).get("run_coco_eval", False)),
        "target_model_role": trigger_context.get("target_model_role"),
        "target_expert_family": trigger_context.get("target_expert_family"),
        "finetune_dataset_bundle_path": str(Path(cfg["_v3_internal"]["finetune_dataset_bundle_path"])),
        "expert_training_strategy": {
            "target_expert_family": trigger_context.get("target_expert_family"),
            "segmentation_algorithm": family_cfg.get("algorithm_name"),
            "dataset_wrapper": defaults.get("dataset_wrapper") or {},
            "curriculum_mode": defaults.get("curriculum_mode"),
            "prior_axes": defaults.get("prior_axes") or [],
            "replay_ratio": defaults.get("replay_ratio") or 0.0,
            "hard_case_ratio": defaults.get("hard_case_ratio") or 0.0,
        },
        "v3_metadata": {
            "trigger_context": trigger_context,
            "family_config": family_cfg,
            "dataset_card_path": dataset_bundle.get("dataset_card_path"),
            "source_config_path": source_config_path,
        },
    }
    return {key: value for key, value in payload.items() if value is not None}


def _trainer_module(algorithm_name: str) -> str:
    if str(algorithm_name) == "maskdino_official":
        return "ITD_agent.segmentation.model_training.train_maskdino_instance"
    return "ITD_agent.segmentation.model_training.train_mmdet_instance"


def _job_id(mode: str, algorithm_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_algo = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in algorithm_name)
    return f"{stamp}_{safe_algo}_{mode}"
