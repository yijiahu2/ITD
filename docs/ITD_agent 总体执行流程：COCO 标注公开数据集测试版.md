# ITD_agent 总体执行流程：COCO 标注公开数据集测试版

```text
0. 系统启动与配置加载
  ↓
1. COCO 数据集输入
  ↓
2. input_layer 输入校验与 InputManifest 构建
  ↓
3. data_processing 构建样本上下文与影像画像
  ↓
4. Experience Retrieval：检索历史经验、Skill、模型能力画像
  ↓
5. Context Injection：把经验注入本轮调度上下文
  ↓
6. planning/scheduler 生成主模型推理计划
  ↓
7. segmentation 执行主模型推理
  ↓
8. evaluation_analysis 评估主模型结果
  ↓
9. evaluation_analysis 分解四类错误
  ↓
10. evaluation_analysis 计算几何指标与失败标签
  ↓
11. 主模型决策 Main Decision
      ├── accept_main → 进入最终结果冻结
      ├── retry_main_plan → 修复主模型计划 → 回到主模型推理
      ├── escalate_expert → 进入专家模型循环
      └── record_failure → 冻结失败结果 / 保留主模型结果
  ↓
12. evolution/roi 构建 ROI candidates
  ↓
13. evolution/roi 对 ROI 分级、聚类
  ↓
14. evolution/expert 构建 ExpertTask
  ↓
15. planning/scheduler + evolution/expert 执行专家路由
  ↓
16. segmentation 执行专家模型推理 / mock / replay
  ↓
17. evaluation_analysis 评估专家结果
  ↓
18. 专家结果决策 Expert Decision
      ├── accept → 局部融合
      ├── partial_accept → 部分融合
      ├── reject → 回滚主模型结果
      ├── retry_expert_plan → 修复 ExpertTask → 回到专家推理
      ├── try_next_expert → 回到专家路由
      └── record_uncertain → 不融合，只记录
  ↓
19. evolution/fusion 融合后重新评估
      ├── 融合优于主模型 → 接受融合结果
      └── 融合退化 → 回滚主模型结果
  ↓
20. 最终结果冻结
      ├── 前台结果输出线
      │     ↓
      │   output_layer 输出results/ masks/ visualization/ report/
      │
      └── 旁路自进化线
            ↓
          evolution/trajectory 写入 Inference Evolution Trajectory
            ↓
          state 写入 SQLite 状态库与 Artifact Registry
            ↓
          review 生成 pending review candidates
            ↓
          memory_store / skill registry / finetune_pool / training_loop 候选写入
            ↓
          后台审查
            ↓
          轻量经验进化闭环 + 模型权重更新闭环
            ↓
          更新经验、Skill、路由策略、训练样本池、模型版本
            ↓
          下一轮推理前被重新检索和使用
```

------

# 0. 系统启动与配置加载

入口命令可以是：

```bash
itd-agent evolve-infer --config configs/examples/itd_agent_evolve_coco.yaml
```

或者暂时用项目已有入口包装：

```bash
python -m ITD_agent.orchestration.evolve_infer_runner --config configs/examples/itd_agent_evolve_coco.yaml
```

当前仓库已有主运行入口是 `python -m ITD_agent.orchestration.orchestrator --config <config>`，同时也有脚本 wrapper；后续 evolve-infer 应该作为新的专用入口，而不是继续把所有逻辑塞进旧 `orchestrator.py`。

配置加载后，系统需要解析：

```text
run 配置
dataset 配置
mainline_profile
main_model 配置
expert_models 配置
roi_policy
evaluation 配置
fusion 配置
trajectory 配置
state/artifact 配置
training_loop 配置
```

配置示意：

```yaml
run:
  mode: supervised_coco_evolve
  experiment_name: itd_agent_coco_evolve
  run_name: coco_main_expert_loop
  output_dir: outputs/evolve_runs/coco_main_expert_loop

mainline:
  profile: A_DOM_ONLY

dataset:
  mode: supervised_coco
  image_dir: /path/to/images
  annotation_json: /path/to/annotations/instances_tree_val.json
  category_name: tree

main_model:
  model_id: legacy_cellpose_sam

expert_models:
  enabled: true
  execution_mode: real   # real | mock | replay

state:
  sqlite_path: outputs/runtime_state/itd_agent_state.db
  artifact_root: outputs/evolve_runs/coco_main_expert_loop
```

------

# 1. COCO 数据集输入

输入内容：

```text
images/
annotations/instances_xxx.json
```

COCO JSON 中至少应包含：

```text
images
annotations
categories
bbox
segmentation
area
image_id
category_id
```

对于单木树冠实例分割任务，理想情况是：

```text
一个 annotation = 一个树冠实例 mask
```

这一步系统只读取数据，不做推理、不做训练。

------

# 2. input_layer：输入校验与 InputManifest 构建

`input_layer` 负责输入合法性和输入清单。仓库 `codemap.md` 明确指出，`input_layer/` 负责构建 `InputManifest`、校验输入路径和 schema、准备输入 workspace 并写入 input registry。

检查内容包括：

```text
COCO annotation_json 是否存在
image_dir 是否存在
COCO JSON 是否能解析
images 中登记的文件是否真实存在
annotation 是否有 segmentation
segmentation 是否有效
category 是否匹配 tree
image width/height 是否与真实图像一致
是否有空标注图像
是否有非法 polygon / RLE
是否存在重复 image_id / annotation_id
```

输出：

```json
{
  "dataset_mode": "supervised_coco",
  "mainline_profile": "A_DOM_ONLY",
  "has_gt": true,
  "image_count": 100,
  "annotation_count": 3500,
  "category_count": 1,
  "input_modalities": {
    "dom": true,
    "coco_gt": true,
    "dem": false,
    "chm": false,
    "inventory": false
  }
}
```

关键原则：

```text
COCO GT 可以进入 evaluation_analysis；
COCO GT 不能进入 main_model inference；
COCO GT 不能泄漏给主模型推理计划；
COCO GT 不能作为专家模型真实推理输入，除非是 mock/oracle 模式并明确标记。
```

------

# 3. data_processing：样本上下文与影像画像构建

`data_processing` 负责输入画像、影像质量、public dataset 摘要、ROI 数据准备和实例后处理等。仓库 codemap 已经把它定义为数据处理层，而不是评估或决策层。

对每张 COCO 图像构建：

```json
{
  "sample_id": "Stadtwald_57_3328",
  "image_id": 123,
  "image_path": ".../Stadtwald_57_3328.tif",
  "width": 1024,
  "height": 1024,
  "gt_instance_count": 35,
  "has_gt": true,
  "dataset_mode": "supervised_coco"
}
```

同时提取 DOM-only 可用画像：

```text
图像尺寸
通道数
亮度
对比度
模糊度
纹理复杂度
树冠尺度先验，如果可由 GT 统计
样本来源
图像质量标签
```

注意：主线 A 暂不使用 DEM、CHM、DSM、小班、清查数据。

------

# 4. Experience Retrieval：推理前经验检索

这是之前流程容易漏掉的关键步骤。

沉淀经验不是存完就结束，它必须在下一轮推理前被检索和使用。

在每次新运行开始前，系统需要基于当前样本画像检索：

```text
相似历史 trajectory
历史 failure patterns
历史 successful strategies
可用 skills
模型能力画像
专家路由历史
训练后模型表现记录
replay 退化记录
```

形成：

```json
{
  "matched_memories": [],
  "matched_skills": [],
  "similar_trajectories": [],
  "model_capability_hints": [],
  "routing_hints": [],
  "parameter_hints": [],
  "safety_guards": []
}
```

示例：

```json
{
  "matched_memory": {
    "failure_family": "small_crown_recall",
    "scene_tags": ["low_contrast", "dense_crown"],
    "recommended_action": "lower_score_threshold_then_check_fp",
    "usefulness_score": 0.84
  }
}
```

------

# 5. Context Injection：经验注入调度上下文

检索到的经验不能直接替代客观评估，只能作为受控上下文进入调度。

注入位置包括：

| 注入对象                 | 经验作用                    |
| ------------------------ | --------------------------- |
| `main_plan_builder`      | 给主模型参数建议            |
| `main_plan_repair`       | 判断是否先修主模型计划      |
| `roi_status_assigner`    | 调整 ROI 严重度和专家资格   |
| `expert_router`          | 调整专家选择和 safety guard |
| `expert_result_reviewer` | 调整专家接受/拒绝检查项     |
| `training_loop`          | 判断样本是否有训练价值      |
| `trajectory`             | 记录本轮使用了哪些历史经验  |

关键原则：

```text
经验只提供 hint；
硬裁决仍由 COCO GT 指标、几何指标和规则阈值完成。
```

------

# 6. planning/scheduler：主模型推理计划生成

`planning/scheduler` 根据：

```text
InputManifest
sample context
data_processing summary
ExperienceContext
模型注册信息
主线 A 限制
```

生成主模型计划。

示例：

```json
{
  "stage": "main_model_plan",
  "model_role": "main_model",
  "model_id": "legacy_cellpose_sam",
  "input_image": ".../Stadtwald_57_3328.tif",
  "tile_size": 1024,
  "runtime_params": {
    "score_threshold": 0.35,
    "mask_threshold": 0.5,
    "merge_iou_threshold": 0.3
  },
  "experience_hints_used": [
    "mem_low_contrast_small_crown_001"
  ],
  "fallback_policy": {
    "enable_main_retry": true,
    "enable_expert_escalation": true
  }
}
```

注意：

```text
GT 不进入 main_plan。
```

------

# 7. segmentation：主模型推理

调用 `ITD_agent.segmentation.executor.execute_segmentation_model` 或新 adapter。当前仓库中 `segmentation/executor.py` 已经承担模型执行入口，并通过 `SegmentationExecutionRequest` 和 `SegmentationExecutionResult` 记录请求与结果，应继续保留和扩展。

主模型执行：

```text
legacy_cellpose_sam
  ↓
输入 DOM image / tile
  ↓
输出实例 mask / vector / prediction json
```

如果图像是 1024×1024：

```text
一张 COCO image = 一个推理 tile
```

如果图像大于 1024：

```text
滑窗切 tile
tile 内推理
tile 间融合
恢复为 image 级结果
```

输出：

```text
main_prediction.json
main_instance_mask.tif
main_instances.gpkg
main_execution_result.json
```

统一结果结构：

```json
{
  "instance_id": "pred_001",
  "bbox_px": [x1, y1, x2, y2],
  "mask_ref": "...",
  "polygon": "...",
  "score": 0.87,
  "area_px": 1234,
  "model_id": "legacy_cellpose_sam"
}
```

------

# 8. evaluation_analysis：主模型 COCO 评估

主模型推理完成后，才允许读取 GT 进行评估。

评估内容：

```text
AP
AP50
AP75
Precision
Recall
F1
mask IoU
boundary IoU
centroid distance error
area error ratio
diameter error ratio
```

输出：

```json
{
  "main_model_metrics": {
    "ap": 0.42,
    "ap50": 0.71,
    "precision": 0.76,
    "recall": 0.68,
    "f1": 0.72
  }
}
```

当前仓库 `evaluation_analysis/evaluator.py` 已经提供 main-model、ROI、expert/child-model 和 finetune-effect 评估 facade，后续应扩展而不是绕开。

------

# 9. evaluation_analysis：四类错误分解

这是 COCO 测试最关键环节。

系统构建：

```text
pred_instances × gt_instances → IoU matrix
```

然后分解四类错误。

## 9.1 漏检 false_negative

```text
GT 有树冠，但没有任何 pred 与其成功匹配。
```

记录：

```json
{
  "error_type": "false_negative",
  "gt_id": "gt_031",
  "matched_pred_ids": [],
  "bbox_px": [...]
}
```

## 9.2 误检 false_positive

```text
pred 有实例，但没有匹配任何 GT。
```

记录：

```json
{
  "error_type": "false_positive",
  "pred_id": "pred_012",
  "matched_gt_ids": [],
  "bbox_px": [...]
}
```

## 9.3 欠分割 under_segmentation

```text
一个 pred 覆盖多个 GT，多个树冠被合并。
```

记录：

```json
{
  "error_type": "under_segmentation",
  "pred_id": "pred_021",
  "affected_gt_ids": ["gt_011", "gt_012", "gt_013"]
}
```

## 9.4 过分割 over_segmentation

```text
多个 pred 对应同一个 GT，一个树冠被拆碎。
```

记录：

```json
{
  "error_type": "over_segmentation",
  "gt_id": "gt_044",
  "affected_pred_ids": ["pred_031", "pred_032"]
}
```

输出：

```text
coco_error_decomposition.json
```

------

# 10. evaluation_analysis：几何质量评估

COCO 阶段，错误类型由 GT 决定；几何指标用于解释、打分和后续训练样本筛选。

计算指标：

```text
area
equivalent_diameter
axis_ratio
compactness
circularity
solidity
boundary_complexity
hole_count
hole_area_ratio
nearest_neighbor_distance
overlap_ratio
local_density
```

生成标签：

```text
tiny_false_positive
oversized_crown
elongated_false_positive
fragmented_boundary
merged_crowns
over_split_crown
duplicate_detection
missing_regular_spacing
unstable_edge_mask
shape_domain_shift
```

示例：

```json
{
  "pred_021": {
    "area": 8420,
    "equivalent_diameter": 103.5,
    "boundary_complexity": 4.8,
    "failure_tags": [
      "oversized_crown",
      "merged_crowns"
    ]
  }
}
```

------

# 11. 主模型循环：Main Loop

主模型不是只跑一次。主模型阶段必须有受控循环。

```text
MAIN_PLAN
  ↓
MAIN_INFER
  ↓
MAIN_EVAL
  ↓
GEOMETRY_REVIEW
  ↓
MAIN_DECISION
      ├── accept_main
      ├── retry_main_plan → MAIN_PLAN_REPAIR → MAIN_INFER
      ├── escalate_expert
      └── record_failure
```

## 11.1 accept_main

如果主模型结果已达标：

```text
直接进入最终结果冻结。
```

## 11.2 retry_main_plan

如果问题更像计划/参数/后处理错误：

```text
修复主模型计划，再重跑主模型。
```

典型情况：

```text
tile 边缘效应
postprocess threshold 问题
scale parameter 问题
全局漏检
全局误检
全局小碎片
score threshold 过高/过低
mask threshold 不合理
```

## 11.3 escalate_expert

如果是局部、明确、专家可能修正的问题：

```text
进入专家模型循环。
```

## 11.4 record_failure

如果证据不足或自动修正风险过高：

```text
保留主模型结果或冻结失败结果，并记录 trajectory。
```

护栏：

```yaml
adaptive_inference:
  max_main_retries: 1
  min_improvement_epsilon: 0.01
```

------

# 12. evolution/roi：ROI candidate 构建

当主模型决策为 `escalate_expert` 时，进入 ROI 流程。

ROI 不是简单裁图，而是：

```text
主模型失败区域的结构化证据。
```

每个四类错误都生成 ROI candidate：

```json
{
  "roi_id": "roi_001",
  "source": "coco_gt",
  "level1_error_type": "under_segmentation",
  "level2_problem_tags": [
    "merged_crowns",
    "oversized_crown"
  ],
  "reason_tags": [
    "crown_overlap",
    "stand_density"
  ],
  "failure_family": "crown_split",
  "bbox_px": [120, 260, 520, 690],
  "affected_pred_ids": ["pred_021"],
  "affected_gt_ids": ["gt_011", "gt_012", "gt_013"],
  "severity_score": 0.82,
  "confidence_level": "confirmed"
}
```

原则：

```text
所有错误都记录为 ROI candidate；
轻微错误不删除，只降级。
```

------

# 13. evolution/roi：ROI 分级

ROI 分级包括：

```text
record_only
monitor
actionable
```

同时有三个布尔字段：

```text
expert_eligible
training_eligible
distill_eligible
```

示例：

```json
{
  "roi_id": "roi_001",
  "review_status": "actionable",
  "expert_eligible": true,
  "training_eligible": true,
  "distill_eligible": false
}
```

注意：

```text
expert_eligible、training_eligible、distill_eligible 不互斥。
```

一个 ROI 可以同时：

```text
交给专家修正
进入训练候选
进入蒸馏候选
进入失败记忆候选
```

------

# 14. evolution/roi：ROI 聚类

专家模型输入通常是：

```text
1024×1024 expert tile
```

所以不能每个小 ROI 单独跑专家。

聚类规则：

```text
同一 image / tile 内
同一 failure_family
达到触发条件
→ 构建 ExpertTask
```

如果 COCO 图像本身是 1024×1024：

```text
一张 image = 一个 expert tile
```

如果图像更大：

```text
以高 severity ROI 为 anchor
裁出 1024×1024 expert tile
聚合同 tile 内同类 ROI
```

------

# 15. evolution/expert：ExpertTask 构建

ExpertTask 是专家模型执行单位。

```json
{
  "expert_task_id": "expert_task_001",
  "trajectory_id": "traj_xxx",
  "image_id": "123",
  "failure_family": "crown_split",
  "level1_error_type": "under_segmentation",
  "roi_ids": ["roi_001", "roi_004"],
  "expert_model": "htc",
  "tile_window_px": [0, 0, 1024, 1024],
  "input_tile_path": ".../expert_tile.tif",
  "valid_mask_path": ".../valid_mask.tif",
  "fusion_scope": "roi_masks_plus_buffer"
}
```

------

# 16. 专家模型路由

路由由：

```text
planning/scheduler/expert_routing_policy.py
evolution/expert/expert_router.py
```

共同完成。

初始规则：

```text
欠分割 → HTC
过分割 → Mask2Former
误检 → Cascade Mask R-CNN
漏检 → MaskDINO
```

但是最终路由应同时参考：

```text
错误类型
failure_family
ROI 几何标签
历史专家成功率
模型能力画像
ExperienceContext
安全 guard
```

当前规则只是初始人工先验，不是永久定论。

------

# 17. segmentation：专家模型推理

专家执行模式：

```text
real
mock
replay
```

## real

调用真实专家模型。

## mock

用于跑通闭环，例如：

```text
use_gt_or_perturbed_gt
simulate_improvement
simulate_failure
```

必须标记：

```text
oracle_mock = true
```

## replay

读取已有专家预测结果，用于测试 ROI、评估、融合和 trajectory 逻辑。

专家输入：

```text
完整 1024×1024 expert tile
```

专家输出：

```text
expert_prediction.json
expert_prediction.tif
expert_instances.gpkg
```

------

# 18. evaluation_analysis：专家结果评估

专家结果不能无条件接受。

比较对象：

```text
主模型在 ROI 内的结果
专家模型在 ROI 内的结果
COCO GT
```

ROI 内评估：

```text
mask_iou
boundary_iou
false_positive_count
false_negative_count
under_segmentation_count
over_segmentation_count
area_error_ratio
diameter_error_ratio
```

ROI 外安全检查：

```text
是否新增大量 FP
是否破坏原本 TP
是否引入异常小碎片
是否造成边界破碎
是否让整体指标退化
```

输出：

```json
{
  "expert_task_id": "expert_task_001",
  "decision": "accept",
  "improvement": {
    "roi_iou_gain": 0.12,
    "under_segmentation_count_drop": 2
  },
  "safety_check": {
    "new_fp_outside_roi": 0,
    "destroyed_tp_outside_roi": 0
  }
}
```

------

# 19. 专家模型循环：Expert Loop

专家循环也不是单次线性流程。

```text
ROI_BUILD
  ↓
ROI_STATUS_ASSIGN
  ↓
ROI_CLUSTER
  ↓
EXPERT_TASK_BUILD
  ↓
EXPERT_ROUTE
  ↓
EXPERT_INFER
  ↓
EXPERT_EVAL
  ↓
EXPERT_DECISION
      ├── accept → local_fusion
      ├── partial_accept → partial_fusion
      ├── reject → rollback_to_main
      ├── retry_expert_plan → EXPERT_TASK_BUILD
      ├── try_next_expert → EXPERT_ROUTE
      └── record_uncertain → keep_main
```

护栏：

```yaml
adaptive_inference:
  max_expert_rounds: 1
  max_next_expert_trials: 0
```

也就是说，设计允许循环，但运行必须受控。

------

# 20. evolution/fusion：融合或回滚

根据专家审查结果执行融合。

## accept

```text
ROI mask + buffer 内采用专家结果；
ROI 外保留主模型。
```

## partial_accept

```text
只融合 accepted_roi_ids；
其余 ROI 保留主模型。
```

## reject

```text
完全回滚；
保留主模型结果。
```

## record_uncertain

```text
不融合；
只记录。
```

融合后必须重新评估：

```text
fused_result_eval > main_result_eval + min_improvement_epsilon
```

如果融合后退化：

```text
回滚主模型结果。
```

------

# 21. 最终结果冻结

最终结果来源可能是：

```text
main_only
main_retried
expert_fused
partial_expert_fused
rollback_to_main
failure_recorded
```

冻结结果包括：

```text
final_prediction.tif
final_instances.gpkg
final_metrics.json
final_summary.json
```

这一步之后分两条线：

```text
前台结果输出线
旁路自进化线
```

------

# 22. output_layer：前台结果输出线

最终结果冻结后，立即进入 output_layer。

输出：

```text
final_prediction.tif
final_instances.gpkg
final_visualization.png
final_metrics.json
final_report.json / html / md
```

报告应包括：

```text
主模型指标
专家介入后指标
四类错误数量
ROI 数量
ExpertTask 数量
专家 accept / partial_accept / reject 统计
融合后提升
回滚原因
失败样本列表
训练候选数量
trajectory 路径
artifact 路径
```

示例：

```json
{
  "run_id": "run_xxx",
  "dataset_mode": "supervised_coco",
  "main_model": {
    "ap50": 0.71,
    "precision": 0.76,
    "recall": 0.68
  },
  "final_result": {
    "ap50": 0.75,
    "precision": 0.78,
    "recall": 0.72
  },
  "error_decomposition": {
    "false_positive": 120,
    "false_negative": 180,
    "under_segmentation": 43,
    "over_segmentation": 59
  },
  "roi_summary": {
    "roi_candidates": 402,
    "expert_tasks": 75,
    "accepted": 41,
    "rejected": 22,
    "partial_accept": 12
  }
}
```

------

# 23. evolution/trajectory：旁路自进化线开始

最终结果冻结后，同时写 trajectory。

Trajectory 记录完整证据链：

```json
{
  "trajectory_id": "traj_xxx",
  "run_id": "run_xxx",
  "image_id": "xxx",
  "mode": "supervised_coco",
  "mainline_profile": "A_DOM_ONLY",

  "input_snapshot": {},
  "experience_context_used": {},
  "main_model_stage": {},
  "main_eval_stage": {},
  "geometry_review_stage": {},
  "main_decision_stage": {},
  "roi_stage": {},
  "expert_task_stage": {},
  "expert_review_stage": {},
  "fusion_stage": {},
  "final_result": {},

  "pending_review_candidates": {
    "memory_candidates": [],
    "skill_candidates": [],
    "training_candidates": [],
    "distillation_candidates": [],
    "routing_update_candidates": []
  },

  "review_status": "pending"
}
```

注意：

```text
Trajectory 是经验和训练闭环的证据资产；
不是最终结果输出的阻塞前置条件。
```

------

# 24. state：SQLite 状态库与 Artifact Registry

系统写入：

```text
runs
trajectories
roi_candidates
expert_tasks
expert_reviews
fusion_events
training_candidates
review_events
artifacts
experience_usage_events
model_routing_events
```

SQLite 只存：

```text
状态
索引
路径
摘要
JSON metadata
hash
```

不存：

```text
大影像
大矢量
大 mask
```

大文件存文件系统：

```text
GeoTIFF / COG
GPKG / GeoJSON
JSON / CSV / Parquet
PNG
```

------

# 25. 后台审查：Background Review

Trajectory 写出后，进入后台审查。

后台审查包括五类：

```text
Memory Review
Skill Review
Training Sample Review
Routing Review
Distillation Review
```

## 25.1 Memory Review

判断是否写长期经验：

```text
同类失败是否重复出现
专家修正是否稳定有效
是否代表典型失败模式
是否值得下次检索使用
```

## 25.2 Skill Review

判断是否形成可复用规则：

```text
某类 ROI 规则是否反复有效
某专家在某 failure family 是否稳定成功
某类错误是否应优先修主模型计划
```

## 25.3 Training Sample Review

判断样本是否进入训练池：

```text
错误类型是否明确
GT 是否可靠
专家修正是否通过
是否有训练价值
是否不是孤立噪声
```

## 25.4 Routing Review

判断专家路由是否要调整：

```text
当前专家是否选对
同类 ROI 是否有更优专家
是否需要增加 safety guard
```

## 25.5 Distillation Review

判断专家成功样本是否反哺主模型：

```text
专家 accept / partial_accept
COCO GT 证明更好
ROI 外无副作用
样本质量达到 gold/silver
```

------

# 26. 轻量经验进化闭环

后台审查通过后，进入轻量闭环：

```text
trajectory
  ↓
memory/skill/routing candidates
  ↓
审查通过
  ↓
memory_store / skill_registry / routing_policy
  ↓
下一轮 Experience Retrieval
  ↓
影响主模型计划、ROI 分级、专家路由、专家审查
```

经验使用必须记录：

```json
{
  "experience_usage_id": "use_xxx",
  "source_memory_id": "mem_xxx",
  "used_in": "expert_router",
  "expected_effect": "improve small crown recall",
  "actual_effect": {
    "recall_gain": 0.04,
    "fp_increase": 0.01
  },
  "result": "effective"
}
```

如果有效：

```text
提高 usefulness_score
强化 skill
提高路由权重
```

如果无效：

```text
降低权重
标记 stale
降级或禁用 skill
```

------

# 27. 模型权重更新闭环

这是更重的闭环，不能省略。

完整流程：

```text
Trajectory
  ↓
Training Review
  ↓
Finetune Pool
  ↓
样本质量筛选
  ↓
Dataset Packager
  ↓
Replay Guard 构建
  ↓
Pilot Training
  ↓
Pilot Evaluation
  ↓
Formal Training
  ↓
New Checkpoint
  ↓
COCO Benchmark
  ↓
Replay Regression
  ↓
Geometry Review
  ↓
Model Promotion
  ↓
Update Model Registry / Model Profiles
  ↓
下一轮推理使用新模型
```

------

## 27.1 样本来源

进入训练闭环的样本包括：

```text
主模型失败 ROI
专家成功修正 ROI
专家失败 ROI
replay good samples
public COCO GT samples
人工确认样本
```

## 27.2 样本用途

| 来源                | 用途                     |
| ------------------- | ------------------------ |
| 主模型漏检样本      | 提升召回                 |
| 主模型误检样本      | 降低 FP                  |
| 欠分割样本          | 提升粘连树冠拆分         |
| 过分割样本          | 提升单冠完整性           |
| 专家成功样本        | 专家能力蒸馏给主模型     |
| 专家失败样本        | 更新专家模型或路由负样本 |
| replay good samples | 防止遗忘                 |

------

## 27.3 training_loop

`training_loop/` 负责：

```text
sample_intake
dataset_packager
trigger_policy
training_plan_builder
pilot_trainer
formal_trainer
replay_guard
post_train_evaluator
model_promotion
expert_to_main_distill
```

训练触发条件示例：

```text
同类失败 ROI ≥ 30
gold/silver 样本 ≥ 100
replay good samples ≥ 50
最近 N 次运行中重复出现 ≥ 3 次
样本质量通过率 ≥ 80%
训练资源允许
人工或规则审批通过
```

------

## 27.4 模型晋级

新 checkpoint 不能直接替换主模型。

流程：

```text
candidate
  ↓
shadow
  ↓
active
  ↓
specialized
```

如果失败：

```text
rejected / deprecated / retired
```

主模型晋级看：

```text
COCO AP / AP50 / AP75
Precision / Recall / F1
四类错误数量变化
专家介入频率是否下降
几何异常是否下降
replay 是否不退化
推理耗时是否可接受
```

专家模型晋级看：

```text
目标 failure family 修正成功率
accept rate
partial_accept rate
reject rate
ROI 内改善幅度
ROI 外副作用
replay 安全性
是否优于现有专家
```

------

# 28. 模型权重更新后如何进入下一轮

晋级成功后更新：

```text
segmentation/model_registry
segmentation/model_profiles
planning/scheduler/expert_routing_policy
Experience Retrieval index
model capability profile
```

示例：

```json
{
  "model_id": "main_model_v002",
  "base_model": "legacy_cellpose_sam",
  "checkpoint": "outputs/training_runs/run_xxx/checkpoints/best.pt",
  "role": "main_model",
  "status": "active",
  "improvement_summary": {
    "ap50_gain": 0.04,
    "false_negative_drop": 0.12,
    "expert_call_rate_drop": 0.18
  }
}
```

下一轮运行时：

```text
InputManifest
  ↓
Scene Profile
  ↓
Experience Retrieval
  ↓
Model Registry 查询 active model
  ↓
Scheduler 选择 main_model_v002 或对应专家模型
  ↓
进入新一轮主模型/专家模型循环
```

这才是模型权重更新闭环真正闭合。

------

# 29. 所有模块串联总图

```text
┌──────────────────────────────────────────────┐
│ 0. CLI / orchestration                        │
│ itd-agent evolve-infer                         │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 1. input_layer                                │
│ COCO 校验 / InputManifest / A_DOM_ONLY profile │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 2. data_processing                            │
│ 样本上下文 / 影像画像 / public dataset summary │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 3. memory_store + skill + model_profiles      │
│ Experience Retrieval                          │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 4. planning/scheduler                         │
│ Context Injection / 主模型计划 / 路由策略       │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 5. segmentation                               │
│ 主模型推理                                    │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 6. evaluation_analysis                        │
│ COCO 评估 / 四类错误 / 几何评估                │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 7. Main Loop Decision                         │
│ accept / retry / expert / failure             │
└──────────────────────────────────────────────┘
         ↓ retry                 ↓ expert
   planning/scheduler        ┌──────────────────────────────┐
   main_plan_repair          │ 8. evolution/roi              │
         ↓                   │ ROI candidate / status/cluster│
   segmentation              └──────────────────────────────┘
   rerun main                             ↓
                              ┌──────────────────────────────┐
                              │ 9. evolution/expert           │
                              │ ExpertTask / routing / runner │
                              └──────────────────────────────┘
                                           ↓
                              ┌──────────────────────────────┐
                              │ 10. segmentation              │
                              │ 专家模型推理 / mock / replay  │
                              └──────────────────────────────┘
                                           ↓
                              ┌──────────────────────────────┐
                              │ 11. evaluation_analysis       │
                              │ 专家结果评估 / ROI 外安全检查 │
                              └──────────────────────────────┘
                                           ↓
                              ┌──────────────────────────────┐
                              │ 12. evolution/fusion          │
                              │ 融合 / 部分融合 / 回滚         │
                              └──────────────────────────────┘
                                           ↓
┌──────────────────────────────────────────────┐
│ 13. final result freeze                       │
│ main_only / expert_fused / rollback           │
└──────────────────────────────────────────────┘
        ↓ 前台输出线                         ↓ 旁路进化线
┌──────────────────────┐          ┌──────────────────────────────┐
│ 14. output_layer      │          │ 15. evolution/trajectory      │
│ final report/results  │          │ trajectory.json               │
└──────────────────────┘          └──────────────────────────────┘
                                                ↓
                                  ┌──────────────────────────────┐
                                  │ 16. state                    │
                                  │ SQLite + Artifact Registry   │
                                  └──────────────────────────────┘
                                                ↓
                                  ┌──────────────────────────────┐
                                  │ 17. background review         │
                                  │ memory/skill/train/routing   │
                                  └──────────────────────────────┘
                                      ↓                    ↓
                         ┌──────────────────────┐   ┌──────────────────────┐
                         │ 18A. light evolution │   │ 18B. training_loop    │
                         │ memory/skill/routing │   │ weight update loop    │
                         └──────────────────────┘   └──────────────────────┘
                                      ↓                    ↓
                         ┌──────────────────────┐   ┌──────────────────────┐
                         │ 下一轮检索使用经验     │   │ 新模型晋级并注册       │
                         └──────────────────────┘   └──────────────────────┘
                                      ↓                    ↓
                                  ┌──────────────────────────────┐
                                  │ 19. 下一轮推理前再次使用       │
                                  │ Experience + New Model        │
                                  └──────────────────────────────┘
```

------

# 最终一句话总结

**ITD_agent 用 COCO 标注公开数据集测试时，完整流程不是一条线性推理链，而是“前台主模型—专家模型受控推理闭环 + 旁路 trajectory 经验沉淀闭环 + 模型权重更新闭环”的组合：输入经 input_layer 和 data_processing 构建上下文后，先检索历史经验并注入 scheduler，再由主模型循环完成推理、评估、重试或升级专家；专家循环围绕 ROI 进行任务构建、路由、推理、评估、融合或回滚；最终结果冻结后立即由 output_layer 输出，同时 trajectory 写入 state 和 artifact registry，后台审查把经验流向 memory/skill/finetune_pool/training_loop，训练闭环再通过样本筛选、pilot、正式训练、replay 验证和模型晋级更新模型权重，使下一轮推理同时使用沉淀经验和新模型版本。**





## 推荐给 Codex 的完整指令

```
请基于当前仓库代码，对 ITD_agent 的执行流程进行一次“最小侵入式架构对齐改造”，目标是让项目的总体执行流程逐步符合下面这套主线 A：DOM-only + COCO 标注数据集测试场景下的智能体核心循环。

注意：不要重写整个项目，不要胡乱新增目录，不要重复造轮子。请先只读审查当前仓库已有模块、入口、脚本职责和调用关系，尤其重点检查：

1. input_layer/
2. ITD_agent/data_processing/
3. ITD_agent/evaluation_analysis/
4. ITD_agent/planning/scheduler/
5. ITD_agent/segmentation/
6. ITD_agent/orchestration/
7. ITD_agent/memory_store/
8. ITD_agent/finetune_pool/
9. output_layer/
10. scripts/

请先明确当前仓库中哪些能力已经存在，哪些可以复用，哪些需要轻量扩展，哪些确实缺失。凡是已有能力，优先复用或扩展原文件；只有当现有模块确实没有清晰承载位置时，才允许新增文件。新增文件必须说明为什么不能放入已有文件。

本次目标执行流程如下：

【前台主推理闭环】

COCO 数据集输入
→ input_layer 校验 COCO 数据集、构建 InputManifest、确认 mainline_profile=A_DOM_ONLY
→ data_processing 构建样本上下文、影像画像、public dataset summary
→ memory_store / skill / model_profiles 做 Experience Retrieval，检索历史经验、失败模式、成功策略、模型能力画像和专家路由历史
→ planning/scheduler 将 ExperienceContext 注入主模型计划、ROI 策略和专家路由上下文
→ segmentation 执行主模型推理
→ evaluation_analysis 对主模型结果做 COCO GT 评估、四类错误分解、几何质量评估
→ Main Loop Decision：
    - accept_main：直接冻结主模型结果
    - retry_main_plan：调用 main_plan_repair 修复主模型计划，再回到主模型推理
    - escalate_expert：进入专家模型循环
    - record_failure：保留主模型结果并记录失败
→ evolution/roi 或现有 ROI 相关模块构建 ROI candidates
→ ROI 分级：record_only / monitor / actionable，并标记 expert_eligible、training_eligible、distill_eligible
→ ROI 聚类为 1024×1024 ExpertTask
→ planning/scheduler + expert_router 进行专家路由
→ segmentation 执行专家模型推理，支持 real / mock / replay 三种模式
→ evaluation_analysis 对专家结果进行 ROI 内评估和 ROI 外安全检查
→ Expert Loop Decision：
    - accept：局部融合
    - partial_accept：部分融合
    - reject：回滚主模型结果
    - retry_expert_plan：修复 ExpertTask 后重试
    - try_next_expert：尝试备选专家
    - record_uncertain：不融合，只记录
→ fusion 后重新评估，若融合退化则回滚
→ final result freeze
→ output_layer 输出 final prediction / final instances / final metrics / final report

【旁路自进化闭环】

final result freeze 后，不阻塞前台结果输出，同时执行：
→ trajectory 写入完整 Inference Evolution Trajectory
→ state / artifact registry 记录 run、trajectory、ROI、ExpertTask、专家审查、融合事件、artifact 路径
→ background review 生成 pending candidates：
    - memory_candidate
    - skill_candidate
    - training_candidate
    - routing_update_candidate
    - distillation_candidate
→ memory_store / skill / finetune_pool / training_loop 只接收候选，不要自动训练、不要自动修改模型权重、不要自动更新主模型或专家模型

【模型权重更新闭环】

请只预留接口和数据流，不要在本次改造中强行实现完整训练。训练闭环应通过 training_loop 管理：
trajectory
→ Training Review
→ finetune_pool
→ 样本质量筛选
→ dataset packager
→ replay guard
→ pilot training
→ formal training
→ new checkpoint
→ COCO benchmark
→ replay regression
→ geometry review
→ model promotion
→ update model_registry / model_profiles
→ 下一轮推理使用新模型

本次代码改造原则：

1. 先做代码审查，输出当前实际执行流程和目标流程差距。
2. 优先修改现有文件，不要为了“好看”随便新增大目录。
3. 如果新增文件，请优先按当前模块边界放置：
   - 指标计算、COCO GT 匹配、几何评估、专家结果对比：放到 ITD_agent/evaluation_analysis/
   - 主模型/专家模型执行、adapter、mock/replay expert：放到 ITD_agent/segmentation/
   - 主模型计划、主模型修复、专家路由、ExperienceContext 注入：放到 ITD_agent/planning/scheduler/
   - 主—专家流程编排：放到 ITD_agent/orchestration/
   - ROI candidate、ExpertTask、trajectory、fusion 这些如果现有模块无法承载，再新增 ITD_agent/evolution/ 下的轻量文件
   - 状态库和 artifact registry 如果不存在，再新增 ITD_agent/state/
   - training_loop 可以新增，但本次只做 sample_intake、trigger_policy、contracts 的轻量骨架，不做真实训练
4. 不要破坏当前已有入口：
   - python -m ITD_agent.orchestration.orchestrator --config <config>
   - scripts/run_ITD_agent_experiment.py
   需要新增 evolve-infer 专用入口时，应保持旧入口兼容。
5. COCO GT 只能用于 evaluation_analysis，不允许泄漏进主模型推理计划和模型输入。
6. LLM 不能做硬裁决，只能生成解释、摘要、审查理由或建议。所有 accept/reject/fusion/training trigger 必须以客观指标和规则为主。
7. 专家模型必须支持 real / mock / replay 三种执行模式，以便先跑通主—专家闭环。
8. Trajectory 必须记录本轮使用过的历史经验、主模型计划、主模型结果、COCO 评估、四类错误、几何评估、ROI、ExpertTask、专家路由、专家结果、融合/回滚、pending candidates。
9. SQLite 或 state 层只保存状态、索引、路径和 metadata，不要把大影像、大 mask、大矢量直接写入数据库。GeoTIFF/GPKG/JSON/CSV/PNG 等 artifact 继续存文件系统。
10. 本次改造完成后，请提供：
    - 修改过的文件清单
    - 新增文件清单
    - 删除或合并建议
    - 当前执行流程图
    - 与目标流程的对应关系
    - 可运行 smoke test 命令
    - 每个阶段的预期输出文件
    - 尚未实现但已预留接口的内容

验收标准：

1. 能用 COCO 标注格式公开数据集跑通 evolve-infer 流程。
2. 主模型结果能被标准化为统一实例格式。
3. evaluation_analysis 能输出 COCO 指标、四类错误分解和几何评估。
4. 每个错误能生成 ROI candidate。
5. ROI 能分级为 record_only / monitor / actionable。
6. actionable ROI 能构建 ExpertTask。
7. 专家模型支持 mock / replay，真实专家模型未部署时也能跑通闭环。
8. 专家结果能与主模型在 ROI 内对比，并输出 accept / partial_accept / reject。
9. 融合结果退化时能回滚主模型结果。
10. 最终结果能正常输出到 output_layer。
11. trajectory 能完整记录全过程。
12. pending review candidates 能生成，但不会自动训练或自动更新模型权重。
13. 不出现重复实现已有功能的情况。
14. 不把 evaluation_analysis、data_processing、segmentation、orchestration 的职责混在一起。
```

