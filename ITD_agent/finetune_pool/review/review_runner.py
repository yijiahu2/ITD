from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.review.asset_writers import (
    write_distillation_candidate,
    write_finetune_sample,
    write_memory_record,
    write_review_event,
    write_review_run_record,
    write_routing_candidate,
    write_skill_record,
)
from ITD_agent.finetune_pool.review.finetune_bundle_exporter import export_finetune_bundle
from ITD_agent.finetune_pool.review.io_utils import load_structured, write_csv, write_json, write_jsonl
from ITD_agent.finetune_pool.review.review_context_builder import ReviewContext, build_llm_review_context, build_review_context
from ITD_agent.finetune_pool.review.review_guardrails import ReviewWriteAction, assert_review_guardrails, check_write_action
from ITD_agent.finetune_pool.review.review_report_builder import build_review_report
from ITD_agent.finetune_pool.review.reviewers import DistillationReviewer, FinetuneReviewer, MemoryReviewer, RoutingReviewer
from ITD_agent.finetune_pool.review.skill_reviewer import build_class_level_skill_records
from ITD_agent.finetune_pool.review.trajectory_compressor import compress_trajectory_for_review
from ITD_agent.finetune_pool.review.trajectory_integrity_validator import validate_trajectory_integrity
from ITD_agent.finetune_pool.review.trajectory_reader import list_v1_trajectories, read_trajectory
from ITD_agent.evolution.state.db import connect_state_db


def run_review_stage(config_path: str) -> dict[str, Any]:
    cfg = load_structured(config_path)
    assert_review_guardrails(cfg)
    source_cfg = cfg.get("source") or {}
    output_cfg = cfg.get("output") or {}
    source_run_id = str(source_cfg.get("run_id") or "")
    db_path = Path(str(source_cfg.get("state_db_path") or Path(source_cfg.get("artifact_root", ".")) / "state.sqlite"))
    artifact_root = Path(str(source_cfg.get("artifact_root") or db_path.parent))
    output_dir = Path(str(output_cfg.get("output_dir") or artifact_root / "review"))
    output_dir.mkdir(parents=True, exist_ok=True)
    review_run_id = f"review_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"

    write_json(output_dir / "config" / "review_config.json", cfg)
    write_json(output_dir / "config" / "normalized_review_config.json", _normalize_config(cfg, db_path, artifact_root, output_dir))

    refs = list_v1_trajectories(db_path=db_path, run_id=source_run_id or None, artifact_root=artifact_root)
    all_decisions: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    contexts: list[ReviewContext] = []
    asset_counts = {
        "memory_records": 0,
        "skill_records": 0,
        "finetune_samples": 0,
        "routing_candidates": 0,
        "distillation_candidates": 0,
    }

    with connect_state_db(db_path) as conn:
        write_review_run_record(
            conn=conn,
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            config_path=config_path,
            output_dir=str(output_dir),
            status="running",
        )
        source_image_root = _resolve_source_image_root(conn, source_run_id)
        for ref in refs:
            trajectory = read_trajectory(ref)
            artifacts = _resolve_artifacts_for_trajectory(conn, ref.trajectory_id)
            integrity = validate_trajectory_integrity(
                trajectory=trajectory,
                artifacts=artifacts,
                cfg={**(cfg.get("integrity") or {}), **(cfg.get("error_recovery") or {})},
            )
            integrity_row = integrity.to_dict()
            integrity_row["trajectory_path"] = str(ref.path)
            integrity_rows.append(integrity_row)
            if not integrity.valid:
                write_review_event(
                    conn=conn,
                    review_run_id=review_run_id,
                    source_run_id=source_run_id,
                    review_type="integrity",
                    decision={
                        "candidate_id": ref.trajectory_id,
                        "candidate_type": "trajectory",
                        "trajectory_id": ref.trajectory_id,
                        "decision": "reject",
                        "reason": "invalid_v1_trajectory",
                        "evidence_refs": integrity_row,
                    },
                    error={"missing_fields": integrity.missing_fields},
                )
                continue

            summary, review_context_payload = compress_trajectory_for_review(trajectory)
            write_json(output_dir / "compressed_trajectories" / f"{ref.trajectory_id}.summary.json", summary)
            write_json(output_dir / "compressed_trajectories" / f"{ref.trajectory_id}.review_context.json", review_context_payload)
            context = build_review_context(trajectory=trajectory, trajectory_summary=summary, artifact_refs=artifacts)
            contexts.append(context)
            _write_split_contexts(output_dir, context, cfg)

            decisions = _review_one_context(context=context, cfg=cfg)
            for decision_obj in decisions:
                decision = decision_obj.to_dict()
                all_decisions.append(decision)
                guardrail = _guardrail_for_decision(decision, cfg)
                write_review_event(
                    conn=conn,
                    review_run_id=review_run_id,
                    source_run_id=source_run_id,
                    decision=decision,
                    review_type=str(decision.get("target_asset_type") or decision.get("candidate_type")),
                    guardrail_result=guardrail.to_dict(),
                )
                if decision.get("decision") != "approve" or not decision.get("safe_to_write") or not guardrail.allowed:
                    continue
                target = decision.get("target_asset_type")
                if target == "memory":
                    write_memory_record(conn=conn, output_dir=output_dir, decision=decision)
                    asset_counts["memory_records"] += 1
                elif target == "finetune_sample":
                    max_total = int((cfg.get("finetune_pool") or {}).get("max_total_samples", 0) or 0)
                    if max_total and asset_counts["finetune_samples"] >= max_total:
                        continue
                    source_image = _resolve_source_image(source_image_root, trajectory)
                    write_finetune_sample(conn=conn, output_dir=output_dir, decision=decision, source_image_path=source_image)
                    asset_counts["finetune_samples"] += 1
                elif target == "routing_candidate":
                    write_routing_candidate(conn=conn, output_dir=output_dir, decision=decision)
                    asset_counts["routing_candidates"] += 1
                elif target == "distillation_candidate":
                    write_distillation_candidate(conn=conn, output_dir=output_dir, decision=decision)
                    asset_counts["distillation_candidates"] += 1

        for record in build_class_level_skill_records(review_run_id=review_run_id, source_run_id=source_run_id, contexts=contexts, cfg=cfg):
            guardrail = check_write_action(ReviewWriteAction.WRITE_SKILL_DRAFT, cfg)
            decision = {
                "candidate_id": record["skill_id"],
                "candidate_type": "skill_candidate",
                "trajectory_id": ",".join(record.get("source_trajectory_ids") or []),
                "decision": "approve" if guardrail.allowed else "reject",
                "reason": "class_level_skill_draft_from_v2_evidence",
                "evidence_refs": record.get("evidence_summary") or {},
                "target_asset_type": "skill_draft",
                "safe_to_write": guardrail.allowed,
            }
            all_decisions.append(decision)
            write_review_event(
                conn=conn,
                review_run_id=review_run_id,
                source_run_id=source_run_id,
                decision=decision,
                review_type="skill_draft",
                guardrail_result=guardrail.to_dict(),
            )
            if guardrail.allowed:
                write_skill_record(conn=conn, output_dir=output_dir, record=record)
                asset_counts["skill_records"] += 1

        all_decisions.extend(_record_guardrail_block_checks(conn=conn, review_run_id=review_run_id, source_run_id=source_run_id, cfg=cfg))
        _write_finetune_manifests(output_dir)
        if bool((cfg.get("finetune_pool") or {}).get("export_coco_bundle", True)):
            export_finetune_bundle(review_output_dir=output_dir, out_dir=output_dir / "finetune_pool" / "coco_export_bundle")
        write_json(output_dir / "integrity" / "integrity_report.json", {"trajectories": integrity_rows})
        write_jsonl(output_dir / "integrity" / "invalid_trajectories.jsonl", [row for row in integrity_rows if not row.get("valid")])
        write_json(output_dir / "compressed_trajectories" / "compression_metrics.json", _compression_metrics(contexts))
        report = build_review_report(
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            output_dir=output_dir,
            integrity_rows=integrity_rows,
            decisions=all_decisions,
            asset_counts=asset_counts,
        )
        write_review_run_record(
            conn=conn,
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            config_path=config_path,
            output_dir=str(output_dir),
            status="completed",
            summary=report,
        )
    return report


def _review_one_context(*, context: ReviewContext, cfg: dict[str, Any]) -> list[Any]:
    decisions: list[Any] = []
    if bool((cfg.get("memory_review") or {}).get("enabled", True)):
        memory_reviewer = MemoryReviewer()
        decisions.extend(memory_reviewer.review_many([*context.memory_candidates, *memory_reviewer.synthesize_candidates(context)], context, cfg))
    if bool((cfg.get("finetune_pool") or {}).get("enabled", True)):
        decisions.extend(FinetuneReviewer().review_many(context.training_candidates, context, cfg))
    if bool((cfg.get("routing_review") or {}).get("enabled", True)):
        decisions.extend(RoutingReviewer().review_many(context.routing_update_candidates, context, cfg))
    if bool((cfg.get("distillation_review") or {}).get("enabled", True)):
        decisions.extend(DistillationReviewer().review_many(context.distillation_candidates, context, cfg))
    return decisions


def _guardrail_for_decision(decision: dict[str, Any], cfg: dict[str, Any]):
    target = decision.get("target_asset_type")
    action_by_target = {
        "memory": ReviewWriteAction.WRITE_MEMORY,
        "skill_draft": ReviewWriteAction.WRITE_SKILL_DRAFT,
        "finetune_sample": ReviewWriteAction.WRITE_FINETUNE_SAMPLE,
        "routing_candidate": ReviewWriteAction.UPDATE_ROUTING_POLICY if bool((cfg.get("routing_review") or {}).get("allow_routing_policy_update", False)) else ReviewWriteAction.WRITE_MEMORY,
        "distillation_candidate": ReviewWriteAction.START_DISTILLATION_JOB if bool((cfg.get("distillation_review") or {}).get("allow_distillation_job", False)) else ReviewWriteAction.WRITE_MEMORY,
    }
    return check_write_action(action_by_target.get(target, ReviewWriteAction.WRITE_MEMORY), cfg)


def _record_guardrail_block_checks(*, conn: sqlite3.Connection, review_run_id: str, source_run_id: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for action in [
        ReviewWriteAction.START_TRAINING_JOB,
        ReviewWriteAction.UPDATE_MODEL_WEIGHT,
        ReviewWriteAction.PROMOTE_MODEL,
        ReviewWriteAction.UPDATE_ROUTING_POLICY,
        ReviewWriteAction.WRITE_SKILL_ACTIVE_POLICY,
        ReviewWriteAction.START_DISTILLATION_JOB,
    ]:
        result = check_write_action(action, cfg)
        decision = {
            "candidate_id": f"guardrail_{action.value}",
            "candidate_type": "guardrail_probe",
            "trajectory_id": None,
            "decision": "approve" if result.allowed else "reject",
            "reason": result.reason,
            "evidence_refs": {"action": action.value},
            "target_asset_type": "guardrail",
            "safe_to_write": False,
        }
        decisions.append(decision)
        write_review_event(
            conn=conn,
            review_run_id=review_run_id,
            source_run_id=source_run_id,
            decision=decision,
            review_type="guardrail",
            guardrail_result=result.to_dict(),
        )
    return decisions


def _write_split_contexts(output_dir: Path, context: ReviewContext, cfg: dict[str, Any]) -> None:
    for name in ["memory", "skill", "finetune", "routing", "distillation"]:
        payload = build_llm_review_context(context, name, cfg)
        write_json(output_dir / "review_contexts" / f"{context.trajectory_id}.{name}_context.json", payload)


def _compression_metrics(contexts: list[ReviewContext]) -> dict[str, Any]:
    return {
        "trajectory_count": len(contexts),
        "total_training_candidates": sum(len(item.training_candidates) for item in contexts),
        "total_distillation_candidates": sum(len(item.distillation_candidates) for item in contexts),
        "large_instance_payloads_removed": True,
        "pending_candidates_embedded_in_llm_context": False,
        "roi_maps_embedded_in_llm_context": False,
        "llm_context_style": "per_reviewer_compact_summary_with_selected_candidate_refs",
    }


def _normalize_config(cfg: dict[str, Any], db_path: Path, artifact_root: Path, output_dir: Path) -> dict[str, Any]:
    normalized = dict(cfg)
    normalized["source"] = {**(cfg.get("source") or {}), "state_db_path": str(db_path), "artifact_root": str(artifact_root)}
    normalized["output"] = {**(cfg.get("output") or {}), "output_dir": str(output_dir)}
    return normalized


def _resolve_source_image_root(conn: sqlite3.Connection, source_run_id: str) -> Path | None:
    row = conn.execute("SELECT config_path FROM runs WHERE run_id = ? LIMIT 1", (source_run_id,)).fetchone()
    if not row or not row[0]:
        return None
    cfg_path = Path(str(row[0]))
    if not cfg_path.exists():
        return None
    try:
        cfg = load_structured(cfg_path)
    except Exception:
        return None
    input_cfg = dict(cfg.get("input") or {})
    if not input_cfg.get("image_root"):
        try:
            from ITD_agent.evolution.real_inference_adapter import derive_dataset_input

            input_cfg = derive_dataset_input(input_cfg)
        except Exception:
            pass
    image_root = input_cfg.get("image_root")
    return Path(str(image_root)) if image_root else None


def _resolve_source_image(image_root: Path | None, trajectory: dict[str, Any]) -> Path | None:
    image_path = str((trajectory.get("input_snapshot") or {}).get("image_path") or "")
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path
    if image_root:
        candidate = image_root / image_path
        if candidate.exists():
            return candidate
    return None


def _write_finetune_manifests(output_dir: Path) -> None:
    samples_path = output_dir / "finetune_pool" / "samples.jsonl"
    rows: list[dict[str, Any]] = []
    if samples_path.exists():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = json.loads(line)
                rows.append({k: sample.get(k) for k in ["sample_id", "source_trajectory_id", "source_roi_id", "sample_type", "target_error_type", "quality_score", "metadata_path"]})
    write_json(output_dir / "finetune_pool" / "manifest.json", rows)
    write_csv(output_dir / "finetune_pool" / "manifest.csv", rows)


def _resolve_artifacts_for_trajectory(conn: sqlite3.Connection, trajectory_id: str) -> dict[str, dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT artifact_id, artifact_type, path, format, metadata_json, sha256
        FROM artifacts
        WHERE trajectory_id = ?
        ORDER BY artifact_id
        """,
        (trajectory_id,),
    ).fetchall()
    artifacts: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = Path(row["path"])
        artifacts[str(row["artifact_type"])] = {
            "artifact_id": row["artifact_id"],
            "path": str(path),
            "exists": path.exists(),
            "format": row["format"],
            "sha256": row["sha256"],
        }
    return artifacts
