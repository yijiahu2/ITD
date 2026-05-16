from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("annotations") or payload.get("predictions") or []
    return [dict(item) for item in payload]


def _write_json(path: str | Path, payload: Any) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = [float(value) for value in a[:4]]
    bx, by, bw, bh = [float(value) for value in b[:4]]
    ax2 = ax + max(aw, 0.0)
    ay2 = ay + max(ah, 0.0)
    bx2 = bx + max(bw, 0.0)
    by2 = by + max(bh, 0.0)
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter) / float(max(union, 1e-9))


def _merge_expert_additions(
    *,
    main_predictions: list[dict[str, Any]],
    expert_predictions: list[dict[str, Any]],
    min_overlap_to_skip: float = 0.5,
) -> list[dict[str, Any]]:
    fused = [dict(item, fusion_source="main_model") for item in main_predictions]
    for expert in expert_predictions:
        bbox = list(expert.get("bbox") or [0, 0, 0, 0])
        overlaps = [
            _bbox_iou(bbox, list(item.get("bbox") or [0, 0, 0, 0]))
            for item in fused
            if str(item.get("image_id")) == str(expert.get("image_id"))
        ]
        if not overlaps or max(overlaps) < min_overlap_to_skip:
            fused.append(dict(expert, fusion_source="expert_model_addition"))
    return fused


def _expert_replace_only_covered_images(
    *,
    main_predictions: list[dict[str, Any]],
    expert_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expert_image_ids = {str(item.get("image_id")) for item in expert_predictions}
    retained_main = [
        dict(item, fusion_source="main_model_uncovered_by_expert")
        for item in main_predictions
        if str(item.get("image_id")) not in expert_image_ids
    ]
    return [*retained_main, *[dict(item, fusion_source="expert_model") for item in expert_predictions]]


def fuse_coco_predictions(
    *,
    main_prediction_path: str | Path,
    expert_prediction_path: str | Path | None,
    dominant_error_type: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    main_predictions = _load_predictions(main_prediction_path)
    expert_predictions = _load_predictions(expert_prediction_path) if expert_prediction_path and Path(expert_prediction_path).exists() else []
    decision_log: list[dict[str, Any]] = []
    error_type = str(dominant_error_type or "boundary_quality")
    fused: list[dict[str, Any]]

    if not expert_predictions:
        fused = [dict(item, fusion_source="main_model") for item in main_predictions]
        decision_log.append({"strategy": "main_only", "reason": "expert_prediction_missing_or_empty"})
    elif error_type in {"under_segmentation", "missed_crown_recall", "boundary_quality"}:
        fused = _merge_expert_additions(main_predictions=main_predictions, expert_predictions=expert_predictions)
        decision_log.append({"strategy": "main_baseline_plus_expert_additions", "dominant_error_type": error_type})
    elif error_type == "false_positive_cleanup":
        fused = _expert_replace_only_covered_images(main_predictions=main_predictions, expert_predictions=expert_predictions)
        decision_log.append({"strategy": "expert_replace_only_covered_images", "dominant_error_type": error_type})
    elif error_type == "over_segmentation":
        fused = _merge_expert_additions(
            main_predictions=main_predictions,
            expert_predictions=expert_predictions,
            min_overlap_to_skip=0.2,
        )
        decision_log.append({"strategy": "main_stable_plus_expert_low_overlap", "dominant_error_type": error_type})
    else:
        fused = _merge_expert_additions(main_predictions=main_predictions, expert_predictions=expert_predictions)
        decision_log.append({"strategy": "main_baseline_plus_expert_additions_default", "dominant_error_type": error_type})

    out_dir = Path(output_dir)
    prediction_path = out_dir / "fused_prediction_coco.json"
    summary_path = out_dir / "fusion_summary.json"
    _write_json(prediction_path, fused)
    summary = {
        "dominant_error_type": error_type,
        "main_count": len(main_predictions),
        "expert_count": len(expert_predictions),
        "fused_count": len(fused),
        "decision_log": decision_log,
        "fused_prediction_coco": str(prediction_path),
    }
    _write_json(summary_path, summary)
    return {**summary, "fused_prediction_path": str(prediction_path), "summary_path": str(summary_path)}
