from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from output_layer.reporting.final_result_evaluator import evaluate_final_result


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        if value is None:
            return "-"
        return str(value)


def _build_benchmark_lines(result: dict[str, Any]) -> list[str]:
    crown_area = result.get("crown_area_iou_0_50") or {}
    lines = [
        "## 最终融合结果质量",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Precision (IoU=0.50) | {_fmt(result.get('precision50_percent'))}% |",
        f"| Recall (IoU=0.50) | {_fmt(result.get('recall50_percent'))}% |",
        f"| AP50 | {_fmt(result.get('ap50'))} |",
        f"| AP75 | {_fmt(result.get('ap75'))} |",
        f"| MAE（匹配树冠面积, IoU=0.50） | {_fmt(result.get('mae'))} |",
        f"| RMSE（匹配树冠面积, IoU=0.50） | {_fmt(result.get('rmse'))} |",
        f"| RMSE%（匹配树冠面积, IoU=0.50） | {_fmt(result.get('rmse_percent'))}% |",
        f"| R2（匹配树冠面积, IoU=0.50） | {_fmt(result.get('r2'))} |",
        f"| Precision (IoU=0.75) | {_fmt(result.get('precision75_percent'))}% |",
        f"| Recall (IoU=0.75) | {_fmt(result.get('recall75_percent'))}% |",
        "",
        "## 评估说明",
        "",
        f"- 评估模式: `benchmark`",
        f"- 预测树冠数量: `{result.get('num_predictions')}`",
        f"- 真值树冠数量: `{result.get('num_ground_truth')}`",
        f"- 预测得分字段: `{result.get('score_field') or 'constant_one'}`",
        f"- 真值文件: `{result.get('ground_truth_file')}`",
        f"- IoU=0.50 匹配树冠数量: `{crown_area.get('num_matched_crowns')}`",
    ]
    return lines


def _build_inventory_lines(result: dict[str, Any]) -> list[str]:
    metrics = result.get("selected_metrics") or {}
    rows = [
        ("预测树木数量", metrics.get("pred_tree_count")),
        ("期望树木数量", metrics.get("expected_tree_count")),
        ("树木数量误差率", metrics.get("tree_count_error_ratio")),
        ("预测平均冠幅(m)", metrics.get("pred_mean_crown_width")),
        ("期望平均冠幅(m)", metrics.get("expected_mean_crown_width")),
        ("冠幅误差率", metrics.get("mean_crown_width_error_ratio")),
        ("预测郁闭度", metrics.get("pred_cover_ratio")),
        ("期望郁闭度", metrics.get("expected_closure")),
        ("郁闭度误差绝对值", metrics.get("closure_error_abs")),
        ("预测密度(株/ha)", metrics.get("pred_density_trees_per_ha")),
        ("期望密度(株/ha)", metrics.get("expected_density")),
        ("密度误差绝对值", metrics.get("density_error_abs")),
    ]
    lines = [
        "## 最终融合结果质量",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## 评估说明",
            "",
            "- 评估模式: `reference_quality`",
            "- 当前任务缺少可用的标准真值树冠矢量，因此未计算 AP50/AP75/R2。",
            "- 报告仅保留融合后最终结果与参考约束数据之间的质量指标。",
        ]
    )
    return lines


def _build_area_consistency_lines(result: dict[str, Any]) -> list[str]:
    area = result.get("area_consistency") or {}
    if not area.get("available"):
        return []
    rows = [
        ("DOM面积(m²)", area.get("patch_area_m2")),
        ("语义树冠面积(m²)", area.get("semantic_area")),
        ("实例union面积(m²)", area.get("instance_union_area")),
        ("语义覆盖率", area.get("semantic_cover_ratio")),
        ("实例覆盖率", area.get("instance_cover_ratio")),
        ("覆盖率差值绝对值", area.get("cover_ratio_delta_abs")),
        ("语义-实例 IoU", area.get("overlap_iou")),
        ("实例/语义面积比", area.get("coverage_ratio")),
        ("语义召回率", area.get("semantic_recall")),
        ("实例泄漏率", area.get("instance_leakage")),
        ("语义缺口率", area.get("semantic_gap")),
    ]
    lines = [
        "",
        "## 冠层面积一致性",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {_fmt(value)} |")
    return lines


def _build_geometry_diagnostics_lines(result: dict[str, Any]) -> list[str]:
    geometry = result.get("geometry_diagnostics") or {}
    if not geometry.get("available"):
        return []
    rows = [
        ("实例数", geometry.get("instance_count"), 0),
        ("union树冠面积(m²)", geometry.get("union_area_m2"), 4),
        ("面积和/union比", geometry.get("sum_to_union_ratio"), 4),
        ("平均面积(m²)", geometry.get("mean_area_m2"), 4),
        ("面积中位数(m²)", geometry.get("median_area_m2"), 4),
        ("平均等效冠幅(m)", geometry.get("mean_equivalent_crown_width_m"), 4),
        ("碎片率(<4m²)", geometry.get("small_fragment_ratio_lt_4m2"), 4),
        ("碎片率(<6m²)", geometry.get("small_fragment_ratio_lt_6m2"), 4),
        ("最大单块占比", geometry.get("max_instance_area_share"), 4),
        ("前5大单块占比", geometry.get("top5_instance_area_share"), 4),
        ("边界接触率", geometry.get("edge_touch_ratio"), 4),
        ("几何重叠对数", geometry.get("overlap_pair_count"), 0),
    ]
    lines = [
        "",
        "## 几何健康度",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for name, value, digits in rows:
        lines.append(f"| {name} | {_fmt(value, digits=digits)} |")
    return lines


def _build_unavailable_lines(result: dict[str, Any]) -> list[str]:
    return [
        "## 最终融合结果质量",
        "",
        "- 无法生成 benchmark 最终评估结果。",
        f"- 原因: {result.get('message') or '未知'}",
    ]


def _build_report_payload(
    summary: dict[str, Any],
    runtime_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = summary.get("final_evaluation")
    if not isinstance(result, dict) or not result:
        result = evaluate_final_result(summary, runtime_cfg=runtime_cfg)
    return {
        "run_name": summary.get("run_name") or summary.get("mode"),
        "tree_crowns_shp": (
            summary.get("tree_crowns_shp")
            or summary.get("merged_inst_shp")
            or (summary.get("segmentation_model") or {}).get("tree_crowns_shp")
            or (summary.get("segmentation_model") or {}).get("y_inst_shp")
        ),
        "tree_points_shp": summary.get("tree_points_shp") or (summary.get("segmentation_model") or {}).get("tree_points_shp"),
        "segmentation_visualization_png": (
            summary.get("segmentation_visualization_png")
            or summary.get("tree_crowns_preview_png")
            or (summary.get("final_outputs") or {}).get("segmentation_visualization_png")
        ),
        "final_evaluation": result,
    }


def build_experiment_report(
    summary: dict[str, Any],
    report_path: str | Path,
    *,
    runtime_cfg: dict[str, Any] | None = None,
    report_json_path: str | Path | None = None,
) -> str:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_report_payload(summary, runtime_cfg=runtime_cfg)
    result = payload["final_evaluation"]
    summary["final_evaluation"] = result

    lines = [f"# 最终评估报告：{payload.get('run_name')}", ""]
    mode = result.get("evaluation_mode")
    if mode == "benchmark":
        lines.extend(_build_benchmark_lines(result))
    elif mode in {"inventory_consistency", "reference_quality"}:
        lines.extend(_build_inventory_lines(result))
        lines.extend(_build_area_consistency_lines(result))
        lines.extend(_build_geometry_diagnostics_lines(result))
    else:
        lines.extend(_build_unavailable_lines(result))

    lines.extend(["", "## 最终交付物", ""])
    for label, path in [
        ("树冠掩码 SHP", payload.get("tree_crowns_shp")),
        ("树木定位点 SHP", payload.get("tree_points_shp")),
        ("融合可视化 PNG", payload.get("segmentation_visualization_png")),
    ]:
        if path:
            lines.append(f"- {label}: `{path}`")
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    if report_json_path:
        report_json_path = Path(report_json_path)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["report_json"] = str(report_json_path)
    return str(report_path)
