from __future__ import annotations

import json
from typing import Any

from .retrospective_input import build_run_retrospective_input


def _build_planning_prompt(
    *,
    planning_stage: str,
    template_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
) -> str:
    stage_label = "主模型" if planning_stage == "main_model" else "子模型"
    return f"""
你是 ITD_agent 的 LLM网关，当前负责 {stage_label} 配置决策。
你的任务不是重写整份配置，而是基于配置模板、评估分析结果、记忆上下文和处理中上下文，产出结构化参数更新建议。

配置模板：
{json.dumps(template_cfg, ensure_ascii=False, indent=2)}

调度上下文：
{json.dumps(scheduler_context, ensure_ascii=False, indent=2)}

要求：
1. 只输出需要更新的参数。
2. 重点考虑主/子模型运行配置、ROI 规则、子模型调用规则、微调训练配置、知识嵌入规则和中间数据处理规则。
3. 若上下文中存在影像纹理指标或纹理标签，例如 contrast、entropy、asm、energy、correlation、homogeneity、texture_complex、texture_smooth，应将其纳入场景判断与参数决策。
4. 若上下文中存在影像质量指标或质量标签，例如 blur、overexposed_ratio、underexposed_ratio、shadow_ratio_estimate、stripe_noise_score、color_cast_score、blur_high、shadow_heavy、stripe_noise、color_cast，也必须纳入参数决策。
5. 若当前主模型是 legacy_cellpose_sam / Cellpose-SAM，必须优先判断并输出这些核心参数是否需要更新：diam_list、tile、overlap、tile_overlap、augment、iou_merge_thr、bsize。
6. 对 Cellpose-SAM 的参数判断必须结合影像分辨率、先验平均冠幅、郁闭度、密度、纹理复杂度、边缘强度、模糊、阴影、曝光异常、条带噪声以及色偏风险，不能只做泛化描述。
7. 若近期成功策略可复用，优先复用。
8. 若近期失败模式反复出现，给出进入微调池与记忆库的建议。
9. 只输出 JSON。

输出格式：
{{
  "parameter_updates": {{}},
  "roi_refine_plan": {{
    "max_rounds": 2,
    "top_k": 3,
    "buffer_m": 5.0,
    "strategy_mode": "auto",
    "selection_rules": ["..."],
    "stop_rules": ["..."]
  }},
  "child_model_call_plan": {{
    "preferred_child_model": "可选",
    "candidate_models": ["..."],
    "routing_rules": ["..."],
    "escalation_rules": ["..."]
  }},
  "finetune_training_plan": {{
    "should_prepare": false,
    "target_module": "segmentation_model",
    "trigger_mode": "defer_until_pool_threshold",
    "train_mode": "head_only",
    "freeze_backbone": false,
    "epochs": 4,
    "batch_size": 1,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "data_selection_rule": "..."
  }},
  "knowledge_embedding_plan": {{
    "backbone_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "neck_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "initial_prediction_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "head_rules": [{{"rule": "...", "condition": "...", "parameter_hint": "..."}}],
    "config_hints": {{}}
  }},
  "reason": "简短说明",
  "memory_rule": "成功策略写入记忆库规则",
  "finetune_rule": "失败样本写入微调池规则"
}}
"""


def _build_roi_decision_prompt(
    *,
    roi_assessment: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    return f"""
你是 ITD_agent 的 LLM网关，当前负责 ROI 细化决策。
请基于 ROI 评估和当前指标，判断是否继续调用子模型细化。

ROI评估：
{json.dumps(roi_assessment, ensure_ascii=False, indent=2)}

当前指标：
{json.dumps(metrics, ensure_ascii=False, indent=2)}

只输出 JSON：
{{
  "continue_refinement": true,
  "reason": "简短说明",
  "preferred_child_model": "可选，若不需要可省略"
}}
    """


def _build_roi_candidate_selection_prompt(
    *,
    candidate_rois: list[dict[str, Any]],
    metrics: dict[str, Any],
    scene_analysis: dict[str, Any] | None = None,
) -> str:
    return f"""
你是 ITD_agent 的 LLM网关，当前负责 ROI 候选区域排序。
请基于候选区域的纹理、阴影、地形复杂度、实例边界异常、语义先验覆盖情况，以及当前场景分析和全局指标，
从这些“已生成的候选区域”中选出最值得优先细化的 ROI。你不能虚构新的几何区域，只能在候选列表中排序和筛选。

场景分析：
{json.dumps(scene_analysis or {}, ensure_ascii=False, indent=2)}

当前全局指标：
{json.dumps(metrics, ensure_ascii=False, indent=2)}

候选 ROI：
{json.dumps(candidate_rois, ensure_ascii=False, indent=2)}

只输出 JSON：
{{
  "selected_candidate_ids": ["signal_roi_00_01", "signal_roi_00_02"],
  "reason": "简短说明",
  "discarded_candidate_ids": ["signal_roi_00_03"]
}}
"""


def _build_retrospective_prompt(
    *,
    run_summary: dict[str, Any],
    memory_context: list[dict[str, Any]] | None,
    finetune_context: list[dict[str, Any]] | None,
) -> str:
    retrospective_input = build_run_retrospective_input(
        run_summary=run_summary,
        memory_context=memory_context,
        finetune_context=finetune_context,
    )
    return f"""
你是 ITD_agent 的 LLM网关，当前负责本轮运行复盘与知识更新建议。

请严格基于下面这个“模板化复盘输入”进行总结，不要自行假设不存在的上下文。

模板化复盘输入：
{json.dumps(retrospective_input, ensure_ascii=False, indent=2)}

请输出 JSON，总结：
1. 本轮成功策略
2. 本轮主要失败模式
3. 是否建议写入记忆库
4. 是否建议进入微调池
5. 是否建议触发微调训练

输出格式：
{{
  "success_strategies": ["..."],
  "failure_modes": ["..."],
  "memory_update": {{
    "should_record": true,
    "reason": "..."
  }},
  "finetune_pool_update": {{
    "should_enqueue": true,
    "reason": "..."
  }},
  "training_recommendation": {{
    "should_trigger": false,
    "target_module": "segmentation_model",
    "reason": "..."
  }}
}}
"""
