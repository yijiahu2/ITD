from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ITD_agent.finetune_pool.review.io_utils import load_structured, write_csv, write_json
from ITD_agent.finetune_pool.dataset_exporter import export_finetune_dataset_bundle
from ITD_agent.training_loop.dom_only_geometry_guard import evaluate_dom_only_geometry_guard
from ITD_agent.training_loop.contracts import TrainingTriggerContext
from ITD_agent.training_loop.expert_to_main_distill import build_expert_to_main_distillation_manifest
from ITD_agent.training_loop.family_config_resolver import resolve_family_training_config
from ITD_agent.training_loop.model_capability_profile import build_model_capability_profile
from ITD_agent.training_loop.model_promotion import decide_model_promotion, status_for_registry
from ITD_agent.training_loop.model_registry import build_model_version_id, register_model_version
from ITD_agent.training_loop.post_train_evaluator import run_post_train_evaluation
from ITD_agent.training_loop.replay_guard import evaluate_replay_guard
from ITD_agent.training_loop.routing_candidate_builder import build_routing_update_candidate
from ITD_agent.training_loop.sample_quality_gate import apply_sample_quality_gate
from ITD_agent.training_loop.trainer_runner import run_training_plan
from ITD_agent.training_loop.training_feedback_writer import write_training_feedback_candidates
from ITD_agent.training_loop.training_bundle_materializer import materialize_training_dataset_bundle
from ITD_agent.training_loop.training_plan_builder import build_training_plan
from ITD_agent.training_loop.trigger_policy import evaluate_training_trigger
from ITD_agent.training_loop.review_asset_loader import load_review_assets


def run_training_loop(config_path: str) -> dict[str, Any]:
    cfg = load_structured(config_path)
    _assert_v3_guardrails(cfg)
    output_dir = Path(str((cfg.get("runner") or {}).get("output_dir") or Path("outputs") / "controlled_training"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshots(cfg, config_path, output_dir)

    target = dict(cfg.get("target") or {})
    source = dict(cfg.get("source") or {})
    review_asset_dir = _resolve_review_asset_dir(source)
    review_assets = load_review_assets(review_asset_dir=review_asset_dir, target_cfg=target, output_dir=output_dir)
    finetune_bundle_result = _export_finetune_bundle(cfg=cfg, review_assets=review_assets, target=target, output_dir=output_dir)
    finetune_bundle = json.loads(Path(finetune_bundle_result["dataset_bundle_path"]).read_text(encoding="utf-8"))

    family_cfg = resolve_family_training_config(
        taxonomy_path=(cfg.get("expert_taxonomy") or {}).get("path", "configs/expert_taxonomy/expert_families.yaml"),
        target_expert_family=target.get("target_expert_family"),
        target_model_id=str(target.get("target_model_id") or ""),
        failure_category=target.get("failure_category"),
    )
    quality = apply_sample_quality_gate(
        samples=list(finetune_bundle.get("training_ready_samples") or []),
        cfg=cfg.get("quality_gate") or {},
        target=target,
        output_dir=output_dir,
    )
    replay_samples = list(finetune_bundle.get("replay_samples") or [])
    trigger_context = _build_trigger_context(
        cfg=cfg,
        target=target,
        review_assets=review_assets,
        finetune_bundle_result=finetune_bundle_result,
        accepted_count=len(quality["accepted_samples"]),
        replay_count=len(replay_samples),
    )
    trigger_decision = evaluate_training_trigger(
        trigger_context,
        min_training_ready=int((cfg.get("quality_gate") or {}).get("min_training_ready_samples", 100)),
        min_replay=int((cfg.get("quality_gate") or {}).get("min_replay_samples", 30)),
        min_public_candidates=int((cfg.get("quality_gate") or {}).get("min_public_candidates", 0)),
        allow_weak_supervision=bool((cfg.get("quality_gate") or {}).get("allow_pseudo_labels", True)),
        training_entry_available=bool(family_cfg.get("training_entry_available")),
        family_config_available=bool(family_cfg.get("available")),
        max_single_trajectory_ratio=float((cfg.get("quality_gate") or {}).get("max_single_trajectory_ratio", 1.0)),
    )
    write_json(output_dir / "trigger" / "trigger_context.json", trigger_context.to_dict())
    write_json(output_dir / "trigger" / "trigger_decision.json", trigger_decision)
    write_json(output_dir / "trigger" / "trigger_report.json", {"family_config": family_cfg, "decision": trigger_decision})

    dataset_bundle = materialize_training_dataset_bundle(
        accepted_samples=quality["accepted_samples"],
        replay_samples=replay_samples,
        output_dir=output_dir,
        dataset_cfg=cfg.get("dataset") or {},
        family_cfg=family_cfg,
    )
    cfg["_v3_internal"] = {
        "finetune_dataset_bundle_path": finetune_bundle_result["dataset_bundle_path"],
        "replay_sample_count": len(replay_samples),
    }

    plan = None
    training_result = None
    pilot_plan = None
    pilot_training_result = None
    formal_plan = None
    formal_training_result = None
    evaluation = {"candidate": {"status": "skipped"}, "delta": {}}
    replay_guard_report = {"passed": False, "decision": "skipped", "reason": "training_not_triggered"}
    dom_geometry_guard_report = {"geometry_guard_passed": False, "status": "skipped", "reason": "training_not_triggered"}
    promotion_decision = {"decision": "not_applicable", "reason": "training_not_triggered"}
    capability_profile = None
    routing_candidate_report = None
    feedback_report = None
    model_record = None
    if trigger_decision["decision"] == "approve_pilot":
        plan = build_training_plan(
            cfg=cfg,
            trigger_context=trigger_context.to_dict(),
            family_cfg=family_cfg,
            dataset_bundle=dataset_bundle,
            output_dir=output_dir,
            training_mode="pilot",
        )
        pilot_plan = plan
        training_result = run_training_plan(plan, execute=bool((cfg.get("runner") or {}).get("execute_training", False)))
        pilot_training_result = training_result
        evaluation = run_post_train_evaluation(cfg=cfg, training_result=training_result, output_dir=output_dir)
        replay_guard_report = evaluate_replay_guard(evaluation=evaluation, cfg=cfg, output_dir=output_dir)
        model_version_id = build_model_version_id(plan)
        dom_geometry_guard_report = evaluate_dom_only_geometry_guard(cfg=cfg, output_dir=output_dir, model_version_id=model_version_id)
        capability_profile = build_model_capability_profile(
            plan=plan,
            training_result=training_result,
            evaluation=evaluation,
            replay_guard_report=replay_guard_report,
            dom_geometry_guard_report=dom_geometry_guard_report,
            family_cfg=family_cfg,
            output_dir=output_dir,
            model_version_id=model_version_id,
        )
        promotion_decision = decide_model_promotion(
            cfg=cfg,
            training_result=training_result,
            replay_guard_report=replay_guard_report,
            dom_geometry_guard_report=dom_geometry_guard_report,
            capability_profile=capability_profile,
            evaluation=evaluation,
            output_dir=output_dir,
        )
        if _should_run_formal_training(cfg, training_result, replay_guard_report, dom_geometry_guard_report):
            plan = build_training_plan(
                cfg=cfg,
                trigger_context=trigger_context.to_dict(),
                family_cfg=family_cfg,
                dataset_bundle=dataset_bundle,
                output_dir=output_dir,
                training_mode="formal",
            )
            formal_plan = plan
            training_result = run_training_plan(plan, execute=bool((cfg.get("runner") or {}).get("execute_training", False)))
            formal_training_result = training_result
            evaluation = run_post_train_evaluation(cfg=cfg, training_result=training_result, output_dir=output_dir)
            replay_guard_report = evaluate_replay_guard(evaluation=evaluation, cfg=cfg, output_dir=output_dir)
            model_version_id = build_model_version_id(plan)
            dom_geometry_guard_report = evaluate_dom_only_geometry_guard(cfg=cfg, output_dir=output_dir, model_version_id=model_version_id)
            capability_profile = build_model_capability_profile(
                plan=plan,
                training_result=training_result,
                evaluation=evaluation,
                replay_guard_report=replay_guard_report,
                dom_geometry_guard_report=dom_geometry_guard_report,
                family_cfg=family_cfg,
                output_dir=output_dir,
                model_version_id=model_version_id,
            )
            promotion_decision = decide_model_promotion(
                cfg=cfg,
                training_result=training_result,
                replay_guard_report=replay_guard_report,
                dom_geometry_guard_report=dom_geometry_guard_report,
                capability_profile=capability_profile,
                evaluation=evaluation,
                output_dir=output_dir,
            )
        if bool((cfg.get("promotion") or {}).get("register_candidate", True)) and training_result.best_checkpoint_path and training_result.status in {"completed", "recovered_ckpt"}:
            model_record = register_model_version(
                plan=plan,
                result=training_result,
                status=status_for_registry(promotion_decision),
                metrics_summary=evaluation.get("candidate") or {},
                replay_guard_summary=replay_guard_report,
                output_dir=output_dir,
                evidence={
                    "dataset_card_path": dataset_bundle.get("dataset_card_path"),
                    "training_plan_path": str(Path(plan.output_dir) / "training_plan.json"),
                    "generated_config_path": plan.generated_config_path,
                    "replay_guard_report_path": str(output_dir / "replay_guard" / "replay_guard_report.json"),
                    "capability_profile_path": capability_profile.get("profile_path"),
                },
            )
        else:
            write_json(
                output_dir / "model_registry" / "registration_skipped.json",
                {
                    "reason": "no_completed_training_checkpoint",
                    "training_status": training_result.status,
                    "best_checkpoint_path": training_result.best_checkpoint_path,
                },
            )
        routing_candidate_report = build_routing_update_candidate(
            cfg=cfg,
            capability_profile=capability_profile,
            promotion_decision=promotion_decision,
            replay_guard_report=replay_guard_report,
            dom_geometry_guard_report=dom_geometry_guard_report,
            family_cfg=family_cfg,
            output_dir=output_dir,
        )
        feedback_report = write_training_feedback_candidates(
            cfg=cfg,
            plan=plan,
            training_result=training_result,
            sample_quality_report=quality["sample_quality_report"],
            dataset_card=dataset_bundle["dataset_card"],
            replay_guard_report=replay_guard_report,
            dom_geometry_guard_report=dom_geometry_guard_report,
            promotion_decision=promotion_decision,
            capability_profile=capability_profile,
            output_dir=output_dir,
        )

    distillation_report = build_expert_to_main_distillation_manifest(
        distillation_candidates=list(review_assets.get("distillation_candidates") or []),
        output_dir=output_dir,
        cfg=cfg,
    )
    summary = {
        "version": "training_loop",
        "mode": "controlled_training",
        "output_dir": str(output_dir),
        "source_run_id": source.get("run_id"),
        "review_asset_dir": str(review_asset_dir),
        "finetune_pool_dir": source.get("finetune_pool_dir"),
        "trigger_decision": trigger_decision,
        "family_config": family_cfg,
        "sample_quality_report": quality["sample_quality_report"],
        "dataset_card": dataset_bundle["dataset_card"],
        "training_plan": plan.to_dict() if plan else None,
        "training_result": training_result.to_dict() if training_result else None,
        "pilot_training_plan": pilot_plan.to_dict() if pilot_plan else None,
        "pilot_training_result": pilot_training_result.to_dict() if pilot_training_result else None,
        "formal_training_plan": formal_plan.to_dict() if formal_plan else None,
        "formal_training_result": formal_training_result.to_dict() if formal_training_result else None,
        "replay_guard": replay_guard_report,
        "dom_only_geometry_guard": dom_geometry_guard_report,
        "capability_profile": capability_profile,
        "promotion_decision": promotion_decision,
        "model_version": model_record.to_dict() if model_record else None,
        "distillation_report": distillation_report,
        "routing_candidate_report": routing_candidate_report,
        "training_feedback_report": feedback_report,
    }
    write_json(output_dir / "reports" / "training_summary.json", summary)
    write_csv(output_dir / "reports" / "training_summary.csv", [_flatten_summary(summary)])
    return summary


def _export_finetune_bundle(*, cfg: dict[str, Any], review_assets: dict[str, Any], target: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = export_finetune_dataset_bundle(
        summary={"run_name": (cfg.get("source") or {}).get("run_id") or "training_loop"},
        runtime_cfg={"output_dir": str(output_dir)},
        finetune_plan={
            "target_model_role": target.get("target_model_role"),
            "target_expert_family": target.get("target_expert_family"),
            "failure_category": target.get("failure_category"),
            "target_module": "segmentation_model",
        },
        finetune_pool_root=review_assets["imported_finetune_pool_root"],
        output_path=output_dir / "finetune_bundle" / "finetune_dataset_bundle.json",
    )
    return result


def _build_trigger_context(
    *,
    cfg: dict[str, Any],
    target: dict[str, Any],
    review_assets: dict[str, Any],
    finetune_bundle_result: dict[str, Any],
    accepted_count: int,
    replay_count: int,
) -> TrainingTriggerContext:
    imported = list(review_assets.get("imported_finetune_samples") or [])
    trajectory_counts = Counter(str((item.get("metadata") or {}).get("source_trajectory_id") or "unknown") for item in imported)
    max_ratio = (max(trajectory_counts.values()) / len(imported)) if imported else 0.0
    selection = dict(finetune_bundle_result.get("selection_summary") or {})
    return TrainingTriggerContext(
        source_run_id=str((cfg.get("source") or {}).get("run_id") or ""),
        source_review_asset_dir=str(_resolve_review_asset_dir(cfg.get("source") or {})),
        target_model_role=str(target.get("target_model_role") or "expert_model"),
        target_model_id=str(target.get("target_model_id") or ""),
        target_expert_family=target.get("target_expert_family"),
        failure_category=target.get("failure_category"),
        training_ready_sample_count=accepted_count,
        weak_supervision_candidate_count=int(selection.get("weak_supervision_candidate_count") or 0),
        replay_sample_count=replay_count,
        public_dataset_candidate_count=int(selection.get("public_dataset_candidate_count") or 0),
        dataset_bundle_path=finetune_bundle_result.get("dataset_bundle_path"),
        evidence={
            "selection_summary": selection,
            "source_concentration": {
                "source_trajectory_count": len(trajectory_counts),
                "max_source_trajectory_ratio": max_ratio,
            },
        },
    )


def _resolve_review_asset_dir(source: dict[str, Any]) -> Path:
    for key in ("review_asset_dir", "finetune_pool_dir"):
        value = source.get(key)
        if value:
            candidate = Path(str(value))
            if key == "finetune_pool_dir" and candidate.name == "finetune_pool":
                return candidate.parent
            return candidate
    raise KeyError("source.review_asset_dir or source.finetune_pool_dir is required")


def _assert_v3_guardrails(cfg: dict[str, Any]) -> None:
    guardrails = cfg.get("guardrails") or {}
    if bool(guardrails.get("allow_active_model_replace", False)):
        raise ValueError("training_loop cannot replace active model automatically")
    if bool(guardrails.get("allow_active_routing_policy_update", False)):
        raise ValueError("training_loop cannot update active route_map automatically")
    if bool(guardrails.get("allow_active_skill_policy", False)):
        raise ValueError("training_loop cannot activate hard skill policy automatically")
    if bool(guardrails.get("allow_llm_direct_training_decision", False)):
        raise ValueError("training_loop cannot let LLM directly decide training")
    if bool(guardrails.get("allow_llm_direct_model_promotion", False)):
        raise ValueError("training_loop cannot let LLM directly promote model")


def _write_config_snapshots(cfg: dict[str, Any], config_path: str, output_dir: Path) -> None:
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "training_config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    normalized = {**cfg, "runner": {**(cfg.get("runner") or {}), "output_dir": str(output_dir)}}
    (config_dir / "normalized_training_config.yaml").write_text(yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": summary.get("output_dir"),
        "source_run_id": summary.get("source_run_id"),
        "trigger_decision": (summary.get("trigger_decision") or {}).get("decision"),
        "accepted_samples": (summary.get("sample_quality_report") or {}).get("accepted_count"),
        "training_status": (summary.get("training_result") or {}).get("status"),
        "promotion_decision": (summary.get("promotion_decision") or {}).get("decision"),
        "model_version_id": (summary.get("model_version") or {}).get("model_version_id"),
        "capability_profile_path": (summary.get("capability_profile") or {}).get("profile_path"),
        "geometry_guard_passed": (summary.get("dom_only_geometry_guard") or {}).get("geometry_guard_passed"),
        "distillation_manifest_count": (summary.get("distillation_report") or {}).get("manifest_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_training_loop(args.config), indent=2, ensure_ascii=False))


def _should_run_formal_training(
    cfg: dict[str, Any],
    pilot_result: Any,
    replay_guard_report: dict[str, Any],
    dom_geometry_guard_report: dict[str, Any],
) -> bool:
    formal_cfg = dict(((cfg.get("training") or {}).get("formal") or {}))
    if not bool(formal_cfg.get("enabled", False)):
        return False
    if bool(formal_cfg.get("require_pilot_pass", True)) and getattr(pilot_result, "status", None) not in {"completed", "recovered_ckpt"}:
        return False
    if bool(formal_cfg.get("require_pilot_replay_guard_pass", True)) and not replay_guard_report.get("passed"):
        return False
    if bool(formal_cfg.get("require_pilot_dom_geometry_guard_pass", True)) and not dom_geometry_guard_report.get("geometry_guard_passed"):
        return False
    return True


if __name__ == "__main__":
    main()
