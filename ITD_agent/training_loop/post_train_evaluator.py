from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ITD_agent.evaluation_analysis.finetune_effect_assessment import compare_finetune_effect
from ITD_agent.finetune_pool.review.io_utils import write_json
from ITD_agent.training_loop.contracts import TrainingRunResult


def run_post_train_evaluation(
    *,
    cfg: dict[str, Any],
    training_result: TrainingRunResult,
    output_dir: str | Path,
) -> dict[str, Any]:
    eval_dir = Path(output_dir) / "evaluation"
    eval_cfg = cfg.get("evaluation") or {}
    test_summary = _run_candidate_eval_if_requested(cfg=cfg, training_result=training_result)
    baseline = {"status": "not_provided"}
    candidate = {
        "status": training_result.status,
        "training_job_id": training_result.training_job_id,
        "best_checkpoint_path": training_result.best_checkpoint_path,
        "test_summary": test_summary,
    }
    delta = {"status": "not_computed", "reason": "no before/after evaluation artifacts configured"}
    baseline_metrics = _load_metrics(eval_cfg.get("baseline_metrics_json"))
    candidate_metrics = _load_metrics(eval_cfg.get("candidate_metrics_json")) or _metrics_from_training_artifacts(training_result)
    if baseline_metrics and candidate_metrics:
        baseline = {"status": "provided", "metrics": baseline_metrics, "source": eval_cfg.get("baseline_metrics_json")}
        candidate = {**candidate, "metrics": candidate_metrics}
        delta = {
            "status": "computed",
            "evaluation_mode": "metric_json_delta",
            "delta": _metric_delta(baseline_metrics, candidate_metrics),
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": candidate_metrics,
        }
    before_csv = eval_cfg.get("before_csv")
    after_csv = eval_cfg.get("after_csv")
    if before_csv and after_csv:
        effect = compare_finetune_effect(
            before_csv=str(before_csv),
            after_csv=str(after_csv),
            out_dir=str(eval_dir / "finetune_effect"),
            config_path=eval_cfg.get("config_path"),
        )
        baseline = {"source_csv": before_csv}
        candidate = {**candidate, "source_csv": after_csv}
        delta = effect
    else:
        effect = {"status": "skipped", "reason": "before_csv/after_csv not configured"}
    paths = {
        "baseline_eval": write_json(eval_dir / "baseline_eval.json", baseline),
        "candidate_eval": write_json(eval_dir / "candidate_eval.json", candidate),
        "delta_eval": write_json(eval_dir / "delta_eval.json", delta),
        "error_type_delta": write_json(eval_dir / "error_type_delta.json", {"status": "skipped", "reason": "no candidate predictions configured"}),
        "geometry_delta": write_json(eval_dir / "geometry_delta.json", {"status": "skipped", "reason": "no geometry eval artifacts configured"}),
        "finetune_effect_assessment": write_json(eval_dir / "finetune_effect_assessment.json", effect),
    }
    return {"baseline": baseline, "candidate": candidate, "delta": delta, "effect": effect, "paths": paths}


def _run_candidate_eval_if_requested(*, cfg: dict[str, Any], training_result: TrainingRunResult) -> dict[str, Any] | None:
    eval_cfg = cfg.get("evaluation") or {}
    if not bool(eval_cfg.get("run_coco_eval", False)):
        return None
    if training_result.status not in {"completed", "recovered_ckpt"} or not training_result.best_checkpoint_path:
        return {"status": "skipped", "reason": "training_checkpoint_unavailable"}
    command = _candidate_eval_command(training_result)
    if not command:
        return {"status": "skipped", "reason": "unsupported_training_command"}
    result = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]))
    return {"status": "completed" if result.returncode == 0 else "failed", "returncode": int(result.returncode), "command": command}


def _candidate_eval_command(training_result: TrainingRunResult) -> list[str] | None:
    if "--config" not in training_result.command:
        return None
    config_path = training_result.command[training_result.command.index("--config") + 1]
    module = "ITD_agent.segmentation.model_training.test_mmdet_instance"
    if any("train_maskdino_instance" in part for part in training_result.command):
        module = "ITD_agent.segmentation.model_training.test_maskdino_instance"
    return [
        sys.executable,
        "-u",
        "-m",
        module,
        "--config",
        str(config_path),
        "--checkpoint",
        str(training_result.best_checkpoint_path),
    ]


def _load_metrics(path_value: Any) -> dict[str, float]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return _extract_metric_dict(payload)
    return {}


def _metrics_from_training_artifacts(training_result: TrainingRunResult) -> dict[str, float]:
    roots = [Path(training_result.training_metrics_path).parent] if training_result.training_metrics_path else []
    if training_result.best_checkpoint_path:
        roots.append(Path(training_result.best_checkpoint_path).parent)
    for root in roots:
        metrics = _find_metrics_under(root)
        if metrics:
            return metrics
    return {}


def _find_metrics_under(root: Path) -> dict[str, float]:
    if not root.exists():
        return {}
    for path in sorted(root.rglob("*.json")):
        if path.stat().st_size > 10_000_000:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        metrics = _extract_metric_dict(payload)
        if metrics:
            return metrics
    return {}


def _extract_metric_dict(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload,
        payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        payload.get("metric") if isinstance(payload.get("metric"), dict) else {},
        payload.get("candidate_metrics") if isinstance(payload.get("candidate_metrics"), dict) else {},
    ]
    aliases = {
        "coco/segm_mAP": "ap_50_95",
        "segm_mAP": "ap_50_95",
        "bbox_mAP": "bbox_ap",
        "coco/segm_mAP_50": "ap50",
        "segm_mAP_50": "ap50",
        "coco/segm_mAP_75": "ap75",
        "segm_mAP_75": "ap75",
    }
    out: dict[str, float] = {}
    for candidate in candidates:
        for raw_key, raw_value in candidate.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key not in {"ap_50_95", "bbox_ap", "ap50", "ap75", "precision", "recall", "target_error_rate", "target_error_delta"}:
                continue
            try:
                out[key] = float(raw_value)
            except (TypeError, ValueError):
                continue
    return out


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    keys = sorted(set(before) | set(after))
    delta: dict[str, float | None] = {}
    for key in keys:
        if before.get(key) is None or after.get(key) is None:
            delta[key] = None
            continue
        delta[key] = float(after[key]) - float(before[key])
    if "target_error_rate" in before and "target_error_rate" in after:
        delta["target_error_delta"] = float(after["target_error_rate"]) - float(before["target_error_rate"])
    return delta
