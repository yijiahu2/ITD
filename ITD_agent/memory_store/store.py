from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ITD_agent.common.json_store import append_jsonl, replace_jsonl, write_json
from ITD_agent.common.scene_profile import extract_input_profile, scene_profile_from_summary
from ITD_agent.common.values import safe_float
from ITD_agent.memory_store.compact import compact_memory_record, compact_planning_summary
from ITD_agent.memory_store.contracts import (
    ExecutionTraceMemory,
    FailurePatternMemory,
    RunRetrospectiveMemory,
    SuccessfulStrategyMemory,
)
from ITD_agent.memory_store.query import (
    DEFAULT_MEMORY_ROOT,
    load_recent_execution_traces,
    load_recent_failure_patterns,
    load_recent_run_retrospectives,
    load_recent_success_strategies,
)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_memory_id(prefix: str, run_name: str | None) -> str:
    return f"{prefix}:{run_name or 'unknown'}:{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _score_metrics(metrics: dict[str, Any]) -> float | None:
    tree = safe_float(metrics.get("tree_count_error_ratio"))
    crown = safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure = safe_float(metrics.get("closure_error_abs"))
    density = safe_float(metrics.get("density_error_abs"))
    if tree is None or crown is None or closure is None:
        return None
    return tree + crown + closure + (density or 0.0) / 1000.0


def _artifact_refs(summary: dict[str, Any]) -> dict[str, Any]:
    final_outputs = summary.get("final_outputs") or {}
    return {
        "summary_json": summary.get("summary_json"),
        "metrics_json": summary.get("metrics_json"),
        "details_csv": summary.get("details_csv"),
        "report_md": summary.get("report_md"),
        "report_json": summary.get("report_json"),
        "tree_crowns_shp": final_outputs.get("tree_crowns_shp") or summary.get("tree_crowns_shp"),
        "tree_points_shp": final_outputs.get("tree_points_shp") or summary.get("tree_points_shp"),
        "segmentation_visualization_png": final_outputs.get("segmentation_visualization_png") or summary.get("segmentation_visualization_png"),
    }


def _extract_input_profile(input_manifest: dict[str, Any]) -> dict[str, Any]:
    return extract_input_profile(input_manifest)


def _extract_scene_profile(summary: dict[str, Any], input_manifest: dict[str, Any]) -> dict[str, Any]:
    return scene_profile_from_summary(summary, input_manifest)


def _planning_summary(summary: dict[str, Any]) -> dict[str, Any]:
    planning = summary.get("planning_scheduler") or {}
    return compact_planning_summary(
        {
            "main_model_plan": planning.get("main_model_plan") or {},
            "roi_round_count": len(planning.get("roi_rounds") or []),
            "roi_rounds": planning.get("roi_rounds") or [],
            "refinement_review": planning.get("refinement_review") or {},
            "finetune_recommendation": planning.get("finetune_recommendation") or {},
            "finetune_training_plan": planning.get("finetune_training_plan") or {},
        }
    )


def _segmentation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    segmentation = summary.get("segmentation_model") or {}
    payload = {
        "memory_type": "execution_trace",
        "segmentation_summary": {
            "main_model": segmentation.get("main_model") or {},
            "roi_round_count": len(segmentation.get("roi_rounds") or []),
            "y_inst_shp": segmentation.get("y_inst_shp"),
            "tree_crowns_shp": segmentation.get("tree_crowns_shp"),
            "tree_points_shp": segmentation.get("tree_points_shp"),
        },
    }
    return compact_memory_record(payload).get("segmentation_summary") or {}


def _evaluation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    final_eval = summary.get("final_evaluation") or {}
    payload = {
        "memory_type": "execution_trace",
        "evaluation_summary": {
            "metrics": summary.get("metrics") or {},
            "final_evaluation": final_eval,
            "failure_analysis": summary.get("failure_analysis") or {},
        },
    }
    return compact_memory_record(payload).get("evaluation_summary") or {}


def _update_indexes(*, memory_root: str | Path, payload: dict[str, Any]) -> None:
    root = Path(memory_root)
    index_root = root / "index"
    scene_index_path = index_root / "by_scene.json"
    tag_index_path = index_root / "by_tag.json"
    scene_index = json.loads(scene_index_path.read_text(encoding="utf-8")) if scene_index_path.exists() else {}
    tag_index = json.loads(tag_index_path.read_text(encoding="utf-8")) if tag_index_path.exists() else {}

    scene = payload.get("scene_profile") or {}
    scene_key = "|".join(
        [
            str(scene.get("forest_type") or ""),
            str(scene.get("terrain_type") or ""),
            str(scene.get("image_resolution_m") or ""),
        ]
    )
    memory_ref = {
        "memory_id": payload.get("memory_id"),
        "run_name": payload.get("run_name"),
        "memory_type": payload.get("memory_type"),
        "timestamp": payload.get("timestamp"),
    }
    scene_index.setdefault(scene_key, [])
    scene_index[scene_key].append(memory_ref)
    for tag in payload.get("tags") or []:
        tag_index.setdefault(str(tag), [])
        tag_index[str(tag)].append(memory_ref)

    write_json(scene_index_path, scene_index)
    write_json(tag_index_path, tag_index)


def _trim_artifact_refs(artifact_refs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "summary_json",
        "metrics_json",
        "details_csv",
        "report_md",
        "report_json",
        "tree_crowns_shp",
        "tree_points_shp",
        "segmentation_visualization_png",
    }
    return {key: value for key, value in (artifact_refs or {}).items() if key in allowed}


def compact_memory_store_records(
    *,
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    remove_legacy_duplicates: bool = True,
) -> dict[str, Any]:
    root = Path(memory_root)
    records_root = root / "records"
    result: dict[str, Any] = {"memory_root": str(root), "compacted_files": {}, "removed_files": []}
    if not records_root.exists():
        return result

    file_names = [
        "execution_trace.jsonl",
        "failure_pattern.jsonl",
        "run_retrospective.jsonl",
        "successful_strategy.jsonl",
        "successful_strategies.jsonl",
    ]
    for name in file_names:
        path = records_root / name
        if not path.exists():
            continue
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(compact_memory_record(json.loads(line)))
                except Exception:
                    continue
        before_size = path.stat().st_size if path.exists() else 0
        replace_jsonl(path, rows)
        after_size = path.stat().st_size if path.exists() else 0
        result["compacted_files"][name] = {
            "row_count": len(rows),
            "size_before": before_size,
            "size_after": after_size,
        }

    if remove_legacy_duplicates:
        for name in ["execution_log.jsonl"]:
            path = records_root / name
            if path.exists():
                path.unlink()
                result["removed_files"].append(str(path))
    return result


def rebuild_memory_indexes(
    *,
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
) -> dict[str, Any]:
    root = Path(memory_root)
    records_root = root / "records"
    index_root = root / "index"
    scene_index_path = index_root / "by_scene.json"
    tag_index_path = index_root / "by_tag.json"
    scene_index: dict[str, list[dict[str, Any]]] = {}
    tag_index: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()

    file_names = [
        "execution_trace.jsonl",
        "failure_pattern.jsonl",
        "run_retrospective.jsonl",
        "successful_strategy.jsonl",
        "successful_strategies.jsonl",
    ]
    indexed_count = 0
    for name in file_names:
        path = records_root / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = compact_memory_record(json.loads(line))
                except Exception:
                    continue
                memory_id = str(payload.get("memory_id") or "")
                dedupe_key = memory_id or json.dumps(payload, ensure_ascii=False, sort_keys=True)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                indexed_count += 1

                scene = payload.get("scene_profile") or {}
                scene_key = "|".join(
                    [
                        str(scene.get("forest_type") or ""),
                        str(scene.get("terrain_type") or ""),
                        str(scene.get("image_resolution_m") or ""),
                    ]
                )
                memory_ref = {
                    "memory_id": payload.get("memory_id"),
                    "run_name": payload.get("run_name"),
                    "memory_type": payload.get("memory_type"),
                    "timestamp": payload.get("timestamp"),
                }
                scene_index.setdefault(scene_key, [])
                scene_index[scene_key].append(memory_ref)
                for tag in payload.get("tags") or []:
                    if not tag:
                        continue
                    tag_index.setdefault(str(tag), [])
                    tag_index[str(tag)].append(memory_ref)

    write_json(scene_index_path, scene_index)
    write_json(tag_index_path, tag_index)
    return {
        "memory_root": str(root),
        "indexed_record_count": indexed_count,
        "scene_key_count": len(scene_index),
        "tag_key_count": len(tag_index),
        "scene_index_path": str(scene_index_path),
        "tag_index_path": str(tag_index_path),
    }


def record_execution(
    *,
    summary: dict[str, Any],
    input_manifest: dict[str, Any],
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
) -> dict[str, Any]:
    root = Path(memory_root)
    scene_profile = _extract_scene_profile(summary, input_manifest)
    payload = ExecutionTraceMemory(
        memory_id=_make_memory_id("execution_trace", summary.get("run_name")),
        memory_type="execution_trace",
        timestamp=_timestamp(),
        run_name=str(summary.get("run_name") or ""),
        mode=str(summary.get("mode") or "single_experiment"),
        scene_profile=scene_profile,
        input_profile=_extract_input_profile(input_manifest),
        planning_summary=_planning_summary(summary),
        segmentation_summary=_segmentation_summary(summary),
        evaluation_summary=_evaluation_summary(summary),
        artifact_refs=_artifact_refs(summary),
        tags=scene_profile.get("tags") or [],
    ).to_dict()
    payload["artifact_refs"] = _trim_artifact_refs(payload.get("artifact_refs") or {})
    payload = compact_memory_record(payload)
    new_path = append_jsonl(root / "records" / "execution_trace.jsonl", payload)
    _update_indexes(memory_root=root, payload=payload)
    return {
        "execution_trace_log": new_path,
        "memory_root": str(root),
    }


def record_success_strategy(
    *,
    summary: dict[str, Any],
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    thresholds = thresholds or {
        "tree_count_error_ratio": 0.20,
        "mean_crown_width_error_ratio": 0.25,
        "closure_error_abs": 0.12,
    }
    metrics = summary.get("metrics") or {}
    tree = safe_float(metrics.get("tree_count_error_ratio"))
    crown = safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure = safe_float(metrics.get("closure_error_abs"))
    if tree is None or crown is None or closure is None:
        return None
    if not (
        tree <= thresholds["tree_count_error_ratio"]
        and crown <= thresholds["mean_crown_width_error_ratio"]
        and closure <= thresholds["closure_error_abs"]
    ):
        return None

    scene_profile = _extract_scene_profile(summary, summary.get("input_manifest") or {})
    retrospective = (((summary.get("llm_gateway") or {}).get("run_retrospective") or {}).get("parsed_result") or {})
    payload = SuccessfulStrategyMemory(
        memory_id=_make_memory_id("successful_strategy", summary.get("run_name")),
        memory_type="successful_strategy",
        timestamp=_timestamp(),
        run_name=str(summary.get("run_name") or ""),
        scene_profile=scene_profile,
        metrics=metrics,
        score=_score_metrics(metrics),
        strategy_summary=_planning_summary(summary),
        llm_success_strategies=[str(item) for item in (retrospective.get("success_strategies") or [])],
        artifact_refs=_artifact_refs(summary),
        tags=scene_profile.get("tags") or [],
    ).to_dict()
    payload["artifact_refs"] = _trim_artifact_refs(payload.get("artifact_refs") or {})
    payload = compact_memory_record(payload)
    new_path = append_jsonl(Path(memory_root) / "records" / "successful_strategy.jsonl", payload)
    _update_indexes(memory_root=memory_root, payload=payload)
    return {"successful_strategy_log": new_path}


def record_failure_pattern(
    *,
    summary: dict[str, Any],
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    thresholds = thresholds or {
        "tree_count_error_ratio": 0.22,
        "mean_crown_width_error_ratio": 0.25,
        "closure_error_abs": 0.12,
        "ap50_min": 0.60,
    }
    metrics = summary.get("metrics") or {}
    final_eval = summary.get("final_evaluation") or {}
    benchmark_metrics = final_eval.get("metrics") or {}
    retrospective = (((summary.get("llm_gateway") or {}).get("run_retrospective") or {}).get("parsed_result") or {})
    tree = safe_float(metrics.get("tree_count_error_ratio"))
    crown = safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure = safe_float(metrics.get("closure_error_abs"))
    ap50 = safe_float(benchmark_metrics.get("ap50"))
    should_record = bool(
        retrospective.get("failure_modes")
        or (tree is not None and tree >= thresholds["tree_count_error_ratio"])
        or (crown is not None and crown >= thresholds["mean_crown_width_error_ratio"])
        or (closure is not None and closure >= thresholds["closure_error_abs"])
        or (ap50 is not None and ap50 < thresholds["ap50_min"])
    )
    if not should_record:
        return None

    scene_profile = _extract_scene_profile(summary, summary.get("input_manifest") or {})
    finetune_recommendation = ((summary.get("planning_scheduler") or {}).get("finetune_recommendation") or {})
    recommended_actions = []
    if finetune_recommendation.get("should_recommend"):
        recommended_actions.append(str(finetune_recommendation.get("reason") or "建议累计样本后触发微调。"))
    if retrospective.get("training_recommendation", {}).get("reason"):
        recommended_actions.append(str(retrospective["training_recommendation"]["reason"]))
    payload = FailurePatternMemory(
        memory_id=_make_memory_id("failure_pattern", summary.get("run_name")),
        memory_type="failure_pattern",
        timestamp=_timestamp(),
        run_name=str(summary.get("run_name") or ""),
        scene_profile=scene_profile,
        failure_summary=summary.get("failure_analysis") or {},
        failure_modes=[str(item) for item in (retrospective.get("failure_modes") or [])],
        trigger_mode=str(finetune_recommendation.get("trigger_mode") or ""),
        recommended_actions=recommended_actions,
        artifact_refs=_artifact_refs(summary),
        tags=scene_profile.get("tags") or [],
    ).to_dict()
    payload["artifact_refs"] = _trim_artifact_refs(payload.get("artifact_refs") or {})
    payload = compact_memory_record(payload)
    path = append_jsonl(Path(memory_root) / "records" / "failure_pattern.jsonl", payload)
    _update_indexes(memory_root=memory_root, payload=payload)
    return {"failure_pattern_log": path}


def record_run_retrospective(
    *,
    summary: dict[str, Any],
    memory_root: str | Path = DEFAULT_MEMORY_ROOT,
) -> dict[str, Any] | None:
    retrospective = (summary.get("llm_gateway") or {}).get("run_retrospective") or {}
    if not retrospective:
        return None
    scene_profile = _extract_scene_profile(summary, summary.get("input_manifest") or {})
    payload = RunRetrospectiveMemory(
        memory_id=_make_memory_id("run_retrospective", summary.get("run_name")),
        memory_type="run_retrospective",
        timestamp=_timestamp(),
        run_name=str(summary.get("run_name") or ""),
        scene_profile=scene_profile,
        llm_gateway_result=retrospective,
        parsed_result=(retrospective.get("parsed_result") or {}) if isinstance(retrospective, dict) else {},
        tags=scene_profile.get("tags") or [],
    ).to_dict()
    payload = compact_memory_record(payload)
    path = append_jsonl(Path(memory_root) / "records" / "run_retrospective.jsonl", payload)
    _update_indexes(memory_root=memory_root, payload=payload)
    return {"run_retrospective_log": path}


__all__ = [
    "DEFAULT_MEMORY_ROOT",
    "compact_memory_store_records",
    "rebuild_memory_indexes",
    "load_recent_execution_traces",
    "load_recent_failure_patterns",
    "load_recent_run_retrospectives",
    "load_recent_success_strategies",
    "record_execution",
    "record_failure_pattern",
    "record_run_retrospective",
    "record_success_strategy",
]
