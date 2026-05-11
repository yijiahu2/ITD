**为避免全部开发，导致工程失控，改成如下三层落地：**

```
V1：可监督主—专家推理闭环
V2：轨迹审查 + memory/skill/finetune_pool
V3：训练触发 + 模型晋级 + 专家反哺主模型
```

第一版不要急着做真正训练，也不要急着让 Skill 自动改路由。第一版的目标应该是：

```
能跑通 COCO 数据集上的：
主模型推理 → 评估 → ROI 提取 → 专家任务构建 → 专家推理/占位推理 → 专家结果评估 → 融合/回滚 → 轨迹记录
```





# V1 正式方案：COCO 可监督主—专家推理闭环

## 0. V1 总体定位

V1 不是完整自进化系统，也不是训练系统。

V1 的正式定位是：

```text
在 COCO/公开标注数据集上，构建 DOM-only 主线 A 下的可监督主—专家推理闭环。
```

V1 的核心目标是证明 ITD_agent 是否具备以下基础能力：

```text
1. 主模型能够统一执行；
2. 主模型错误能够被 COCO GT 客观分解；
3. 欠分割、过分割、误检、漏检能够被转化为 ROI candidate；
4. ROI 能够被分级、聚类并组织成 ExpertTask；
5. ExpertTask 能够路由到对应专家模型；
6. 专家模型可以 real / mock / replay 三种模式执行，但是默认real模式；
7. 专家结果能够和主模型结果在 ROI 内客观比较；
8. 系统能够 accept / partial_accept / reject；
9. 专家融合后能够重新评估，不达标则回滚；
10. 全过程能够写入 trajectory；
11. SQLite 能够记录状态与索引；
12. Artifact Store 能够保存影像、矢量、JSON、CSV 等文件；
13. memory / skill / finetune / training / distillation 只生成 pending candidate，不自动生效。
```

V1 的一句话定义：

```text
V1 是 ITD_agent 从“普通模型推理系统”走向“主模型—专家模型自进化智能体”的第一块地基；
它不追求模型立即变强，而是先证明系统能否可监督地发现错误、组织专家纠错、验证专家是否有效、融合或回滚，并完整记录推理轨迹。
```

------

# 1. V1 做什么，不做什么

## 1.1 V1 必须做

```text
1. 读取 COCO / 公开标注数据集；
2. 调用主模型执行推理；
3. 根据 COCO GT 计算主模型错误；
4. 分解四类错误：
   - false_negative 漏检
   - false_positive 误检
   - under_segmentation 欠分割
   - over_segmentation 过分割
5. 计算几何指标与几何失败标签；
6. 生成 ROI candidate；
7. 给 ROI 打状态、严重度、专家候选、训练候选等标签；
8. 聚类 ROI，生成 1024×1024 ExpertTask；
9. 根据规则路由专家模型；
10. 支持 real / mock / replay 专家执行模式；
11. 对比专家结果与主模型结果；
12. 判断 accept / partial_accept / reject；
13. 局部融合或回滚；
14. 写入 trajectory；
15. 写入 SQLite 状态库；
16. 注册 artifact；
17. 生成 pending memory / skill / training / distillation candidates。
```

## 1.2 V1 明确不做

```text
1. 不自动训练；
2. 不自动微调；
3. 不自动更新 Skill；
4. 不自动写入长期 Memory；
5. 不自动更新专家路由权重；
6. 不做专家模型生命周期管理；
7. 不做专家模型晋级；
8. 不做专家结果反哺主模型；
9. 不做主线 B 中的 DEM / CHM / DSM / 小班清查 / 外部知识增强；
10. 不把 LLM 作为指标裁判、融合裁判或训练触发裁判。
```

V1 中所有进化相关内容只生成候选，不自动执行：

| 机制             | V1 行为                     |
| ---------------- | --------------------------- |
| Memory           | 只生成 `memory_candidate`   |
| Skill            | 只生成 `skill_candidate`    |
| Finetune Pool    | 只生成 `training_candidate` |
| Training Loop    | 只做 dry-run sample intake  |
| Routing Learning | 只记录 routing event        |
| Distillation     | 只生成 `distill_candidate`  |
| Model Promotion  | 不执行                      |

------

# 2. V1 核心原则

## 2.1 GT 优先原则

V1 是可监督版本，因此所有关键裁决必须以 COCO GT 为最高优先级。

正式判断顺序：

```text
COCO GT 匹配结果 > 几何质量指标 > LLM 解释
```

也就是说：

```text
四类错误必须首先由 GT 匹配确定；
几何指标只补充 severity_score、failure tags、reason tags 和训练价值；
LLM 只生成解释、摘要和审查理由，不参与硬裁决。
```

## 2.2 主模型优先原则

专家模型不是备用模型，也不是多模型投票工具。

V1 中主模型仍然是稳定推理入口：

```text
主模型先推理；
主模型先评估；
主模型暴露局部失败；
只有明确局部错误才进入 ROI 和 ExpertTask；
全局性错误优先修复主模型计划，而不是直接交给专家模型。
```

## 2.3 专家局部纠错原则

专家模型只负责局部问题：

```text
专家模型可以看完整 expert tile；
但系统只允许其修改 ROI mask 或 ROI mask + buffer；
ROI 外默认保留主模型结果。
```

## 2.4 融合可回滚原则

专家结果不能直接覆盖主模型。

必须经过：

```text
专家结果评估
→ ROI 内改善判断
→ ROI 外安全检查
→ 融合后重新评估
→ 不达标则回滚
```

## 2.5 Trajectory 优先原则

V1 最重要的产物不是最终 mask，而是：

```text
Inference Evolution Trajectory
```

每张 COCO image 都必须生成一条完整 trajectory，用于后续 V2/V3 的 memory、skill、finetune、training、routing learning 和 distillation。

------

# 3. V1 总体流程

V1 固定流程如下：

```text
COCO image + COCO GT
  ↓
InputManifest / CocoSampleContext
  ↓
主模型推理
  ↓
主模型 COCO 评估
  ↓
四类错误分解
  ↓
几何指标与失败标签
  ↓
ROI candidate 构建
  ↓
ROI 分级与专家触发判断
  ↓
ROI 聚类为 1024×1024 ExpertTask
  ↓
专家路由
  ↓
专家模型推理 / mock expert / replay expert
  ↓
专家结果 ROI 内评估
  ↓
accept / partial_accept / reject
  ↓
局部融合或回滚
  ↓
final candidate result
  ↓
trajectory.json
  ↓
SQLite 状态记录 + artifact registry
  ↓
pending review candidates
```

------

# 4. V1 状态机

V1 不能只写成普通顺序脚本，而要保留受限状态机骨架。

## 4.1 LoopStage

```python
from enum import Enum

class LoopStage(str, Enum):
    CONTEXT_LOAD = "context_load"
    MAIN_PLAN = "main_plan"
    MAIN_INFER = "main_infer"
    MAIN_EVAL = "main_eval"
    GEOMETRY_REVIEW = "geometry_review"
    MAIN_DECISION = "main_decision"
    MAIN_PLAN_REPAIR = "main_plan_repair"
    ROI_BUILD = "roi_build"
    ROI_STATUS_ASSIGN = "roi_status_assign"
    ROI_CLUSTER = "roi_cluster"
    EXPERT_TASK_BUILD = "expert_task_build"
    EXPERT_ROUTE = "expert_route"
    EXPERT_INFER = "expert_infer"
    EXPERT_EVAL = "expert_eval"
    FUSION = "fusion"
    FINALIZE = "finalize"
    TRAJECTORY_WRITE = "trajectory_write"
    BACKGROUND_REVIEW_PENDING = "background_review_pending"
```

## 4.2 MainDecision

```python
class MainDecision(str, Enum):
    ACCEPT_MAIN = "accept_main"
    RETRY_MAIN_PLAN = "retry_main_plan"
    ESCALATE_EXPERT = "escalate_expert"
    RECORD_FAILURE = "record_failure"
```

## 4.3 ExpertDecision

```python
class ExpertDecision(str, Enum):
    ACCEPT = "accept"
    PARTIAL_ACCEPT = "partial_accept"
    REJECT = "reject"
    RECORD_UNCERTAIN = "record_uncertain"
```

## 4.4 V1 护栏

```yaml
adaptive_inference:
  max_main_retries: 1
  max_expert_rounds: 1
  max_next_expert_trials: 0
  min_improvement_epsilon: 0.01
  allow_llm_direct_fusion_decision: false
  allow_llm_direct_training_trigger: false
  allow_llm_direct_model_promotion: false
```

LLM 可以做：

```text
解释
摘要
候选建议
审查理由生成
trajectory 压缩
错误模式自然语言总结
```

LLM 不允许做：

```text
指标计算
最终融合裁决
训练触发硬判定
模型晋级硬判定
专家路由权重自动更新
```

------

# 5. V1 模块边界

## 5.1 evaluation_analysis/

职责：

```text
COCO GT 匹配
四类错误分解
几何指标计算
几何失败标签
主模型评估
专家结果对比
融合后质量审查
```

新增或扩展：

```text
ITD_agent/evaluation_analysis/
  coco_error_decomposition.py
  geometry_metrics.py
  geometry_failure_tags.py
  expert_result_comparator.py
  fusion_quality_review.py
```

注意：

```text
evaluation_analysis 负责“算”；
不负责构建 ROI；
不负责写 trajectory；
不负责训练触发；
不负责 artifact registry。
```

------

## 5.2 evolution/

职责：

```text
组织主—专家闭环中的自进化证据对象；
构建 ROI；
构建 ExpertTask；
组织专家执行；
执行融合或回滚；
构建 trajectory；
生成 pending review candidates。
```

新增：

```text
ITD_agent/evolution/
  schemas/
  geometry/
  roi/
  expert/
  fusion/
  trajectory/
  review/
```

注意：

```text
evolution 不重复计算指标；
evolution 调用 evaluation_analysis 的结果。
```

------

## 5.3 segmentation/

职责：

```text
主模型执行
专家模型执行
模型 adapter
模型执行结果契约
mock expert
replay expert
```

扩展：

```text
ITD_agent/segmentation/
  adapters/
    base.py
    mock_expert.py
    replay_expert.py
```

V1 必须保留统一 adapter 思路：

```python
class SegmentationModelAdapter(Protocol):
    model_id: str
    model_role: str

    def predict(self, input_tile, params: dict):
        ...
```

------

## 5.4 planning/scheduler/

职责：

```text
主模型计划构建
主模型计划修复
专家路由策略
ROI policy 加载
```

扩展：

```text
ITD_agent/planning/scheduler/
  expert_routing_policy.py
  roi_policy_loader.py
  main_plan_repair.py
```

------

## 5.5 state/

职责：

```text
SQLite 状态库
run 状态
trajectory 索引
ROI 索引
ExpertTask 索引
artifact 索引
pending candidate 索引
```

新增：

```text
ITD_agent/state/
  db.py
  schema.sql
  repositories.py
  artifact_store.py
```

SQLite 不保存大影像、大 mask、大矢量本体。

------

## 5.6 training_loop/

V1 中可以新建，但只做 dry-run。

新增：

```text
ITD_agent/training_loop/
  contracts.py
  sample_intake.py
  trigger_policy.py
```

V1 功能：

```text
接收 training_eligible ROI；
生成 pending training candidate；
检查是否满足最低训练触发条件；
不启动训练；
不改模型权重。
```

------

## 5.7 orchestration/

新增：

```text
ITD_agent/orchestration/evolve_infer_runner.py
```

原则：

```text
不要直接大改旧 orchestrator.py；
旧 orchestrator.py 暂时保留；
V1 用 evolve_infer_runner.py 承载新闭环；
稳定后再考虑迁移和整合。
```

------

## 5.8 cli/

新增：

```text
ITD_agent/cli/
  main.py
  evolve_infer_cmd.py
```

V1 主要命令：

```bash
itd-agent evolve-infer --config configs/examples/itd_agent_evolve_coco_v1.yaml
```

------

# 6. V1 最小目录结构

最终 V1 建议新增和扩展如下：

```text
ITD_agent/
  orchestration/
    evolve_infer_runner.py

  evolution/
    schemas/
      roi.py
      expert_task.py
      trajectory.py
      decisions.py

    geometry/
      geometry_review_adapter.py

    roi/
      roi_candidate_builder.py
      roi_status_assigner.py
      roi_family_mapper.py
      roi_clusterer.py

    expert/
      expert_task_builder.py
      expert_router.py
      expert_task_runner.py
      expert_review_adapter.py

    fusion/
      local_roi_fusion.py

    trajectory/
      trajectory_builder.py
      trajectory_writer.py

    review/
      pending_review_writer.py

  state/
    db.py
    schema.sql
    repositories.py
    artifact_store.py

  training_loop/
    contracts.py
    sample_intake.py
    trigger_policy.py

  cli/
    main.py
    evolve_infer_cmd.py
```

扩展已有目录：

```text
ITD_agent/evaluation_analysis/
  coco_error_decomposition.py
  geometry_metrics.py
  geometry_failure_tags.py
  expert_result_comparator.py
  fusion_quality_review.py

ITD_agent/segmentation/
  adapters/
    base.py
    mock_expert.py
    replay_expert.py

ITD_agent/planning/scheduler/
  expert_routing_policy.py
  roi_policy_loader.py
  main_plan_repair.py
```

------

# 7. V1 配置文件

新增：

```text
configs/examples/itd_agent_evolve_coco_v1.yaml
```

建议配置如下：

```yaml
run:
  mode: supervised_coco_evolve_v1
  experiment_name: itd_agent_v1_coco_main_expert_loop
  run_name: coco_main_expert_loop_test
  output_dir: outputs/evolve_runs/coco_main_expert_loop_test

mainline:
  profile: A_DOM_ONLY

dataset:
  mode: supervised_coco
  image_dir: /path/to/coco/images
  annotation_json: /path/to/coco/annotations/instances_val.json
  category_name: tree
  sample_limit: 20
  image_size_policy: keep_original

main_model:
  model_id: legacy_cellpose_sam
  role: main_model
  execution:
    use_existing_prediction: false
    prediction_dir: null
  params:
    tile_size: 1024
    score_threshold: 0.35
    mask_threshold: 0.5
    merge_iou_threshold: 0.3

evaluation:
  matching:
    iou_threshold: 0.5
    weak_overlap_threshold: 0.1
  geometry:
    enabled: true
    threshold_mode: adaptive_percentile

adaptive_inference:
  max_main_retries: 1
  max_expert_rounds: 1
  max_next_expert_trials: 0
  min_improvement_epsilon: 0.01
  allow_llm_direct_fusion_decision: false
  allow_llm_direct_training_trigger: false
  allow_llm_direct_model_promotion: false

roi_policy:
  expert_tile_size_px: 1024
  fusion_buffer_px: 64
  threshold_mode: adaptive_percentile

  minor_error_policy:
    max_instances: 2
    max_failure_area_ratio_in_tile: 0.005
    max_severity_score: 0.35
    default_review_status: record_only

  min_trigger_per_tile:
    min_failure_instances: 3
    min_failure_area_ratio: 0.01
    min_affected_tree_ratio: 0.10

  tiny_fragment_policy:
    clustered_tiny_roi_min_count: 5
    clustered_tiny_roi_min_area_ratio: 0.01

  severe_failure_override:
    enabled: true
    min_severity_score: 0.75
    min_single_failure_area_ratio: 0.03
    min_affected_tree_ratio: 0.15

  global_failure_guard:
    max_tile_clusters_per_block: 8
    max_failure_area_ratio_in_block: 0.15
    tiny_roi_count_global_threshold: 30
    tiny_roi_spread_area_ratio_threshold: 0.20
    action: main_plan_repair

expert_models:
  enabled: true
  execution_mode: mock   # real | mock | replay
  max_expert_rounds: 1

  mock:
    strategy: use_gt_or_perturbed_gt
    allow_oracle_for_pipeline_test: true

  replay:
    prediction_root: null

expert_routing_policy:
  version: v1_rule_based
  route_map:
    under_segmentation: htc
    over_segmentation: mask2former
    false_positive: cascade_mask_rcnn
    false_negative: maskdino

fusion:
  scope: roi_masks_plus_buffer
  keep_main_result_outside_roi: true
  require_expert_review: true
  min_improvement_epsilon: 0.01

trajectory:
  enabled: true
  write_full_json: true

state:
  sqlite_path: outputs/runtime_state/itd_agent_state.db
  artifact_root: outputs/evolve_runs/coco_main_expert_loop_test

pending_review:
  enabled: true
  write_memory_candidate: true
  write_skill_candidate: true
  write_training_candidate: true
  write_distill_candidate: true
```

------

# 8. V1 输入与输出

## 8.1 输入

V1 输入：

```text
COCO annotation JSON
COCO image directory
主模型配置
专家模型配置
专家路由配置
ROI policy
融合策略
SQLite 路径
输出目录
```

V1 暂不输入：

```text
DEM
CHM
DSM
小班矢量
林业清查表
外部知识库
真实 DOM-only 无 GT 评估输入
```

## 8.2 输出目录

每个 run 输出结构：

```text
outputs/evolve_runs/{run_id}/
  config/
    normalized_config.yaml

  input/
    coco_manifest.json
    sample_index.json

  main_model/
    {image_id}/
      main_prediction.json
      main_prediction.tif
      main_instances.gpkg
      main_execution_result.json

  evaluation/
    {image_id}/
      coco_error_decomposition.json
      geometry_profile.json
      geometry_failure_tags.json
      main_eval_summary.json

  roi/
    {image_id}/
      roi_candidates.json
      roi_candidates.gpkg
      roi_clusters.json

  expert_tasks/
    {image_id}/
      expert_task_0001/
        expert_tile.tif
        valid_mask.tif
        expert_prediction.json
        expert_prediction.tif
        expert_instances.gpkg
        expert_review.json

  fusion/
    {image_id}/
      fused_prediction.json
      fused_prediction.tif
      fused_instances.gpkg
      fusion_summary.json

  trajectory/
    {image_id}/
      trajectory.json

  review/
    pending_reviews.json

  reports/
    run_summary.json
    aggregate_metrics.json
```

------

# 9. V1 核心数据结构

## 9.1 ROICandidate

```python
from dataclasses import dataclass

@dataclass
class ROICandidate:
    roi_id: str
    run_id: str
    trajectory_id: str
    image_id: str

    source: str  # coco_gt
    level1_error_type: str  # false_negative | false_positive | under_segmentation | over_segmentation
    level2_problem_tags: list[str]
    reason_tags: list[str]
    failure_family: str

    bbox_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    area_px: float

    affected_pred_ids: list[str]
    affected_gt_ids: list[str]

    severity_score: float
    confidence_level: str  # V1 中通常为 confirmed

    geometry_metrics: dict

    review_status: str  # record_only | monitor | actionable
    expert_eligible: bool
    training_eligible: bool
    distill_eligible: bool

    recommended_action: str
    recommended_expert: str | None
```

注意：

```text
expert_eligible、training_eligible、distill_eligible 不是互斥状态。
```

一个 ROI 可以同时是：

```text
专家候选
训练候选
蒸馏候选
失败记忆候选
```

------

## 9.2 ExpertTask

```python
@dataclass
class ExpertTask:
    expert_task_id: str
    run_id: str
    trajectory_id: str
    image_id: str

    expert_model: str
    execution_mode: str  # real | mock | replay

    failure_family: str
    level1_error_type: str
    roi_ids: list[str]

    tile_window_px: tuple[int, int, int, int]
    input_tile_path: str
    valid_mask_path: str | None

    fusion_scope: str
    trigger_reason: dict
    status: str
```

------

## 9.3 ExpertReview

```python
@dataclass
class ExpertReview:
    review_id: str
    expert_task_id: str

    decision: str  # accept | partial_accept | reject | record_uncertain

    roi_metric_before: dict
    roi_metric_after: dict
    improvement: dict
    safety_check: dict

    accepted_roi_ids: list[str]
    rejected_roi_ids: list[str]

    reason: str
```

------

## 9.4 InferenceEvolutionTrajectory

```python
@dataclass
class InferenceEvolutionTrajectory:
    trajectory_id: str
    run_id: str
    image_id: str
    mode: str  # supervised_coco_evolve_v1
    mainline_profile: str  # A_DOM_ONLY

    input_snapshot: dict
    main_model_stage: dict
    main_eval_stage: dict
    geometry_review_stage: dict
    main_decision_stage: dict
    roi_stage: dict
    expert_task_stage: dict
    expert_review_stage: dict
    fusion_stage: dict
    final_result: dict

    pending_review_candidates: dict
    review_status: str  # pending
```

------

# 10. V1 四类错误分解

新增文件：

```text
evaluation_analysis/coco_error_decomposition.py
```

## 10.1 匹配基础

使用 IoU matrix：

```text
pred_instances × gt_instances
```

基础阈值：

```yaml
iou_threshold: 0.5
weak_overlap_threshold: 0.1
```

输出结构：

```python
@dataclass
class CocoErrorDecomposition:
    matched_pairs: list[dict]
    false_positive_preds: list[str]
    false_negative_gts: list[str]
    under_segmentation_events: list[dict]
    over_segmentation_events: list[dict]
    iou_matrix_summary: dict
```

------

## 10.2 漏检 false_negative

定义：

```text
GT 没有任何 pred IoU ≥ 0.5
```

ROI core：

```text
GT mask / GT bbox + buffer
```

推荐标签：

```text
small_crown_miss
shadow_miss
low_contrast_miss
dense_area_miss
edge_miss
```

------

## 10.3 误检 false_positive

定义：

```text
Pred 没有任何 GT IoU ≥ 0.5
```

ROI core：

```text
Pred mask / pred bbox + buffer
```

推荐标签：

```text
tiny_false_positive
shadow_false_positive
background_texture_false_positive
elongated_false_positive
low_confidence_noise
```

------

## 10.4 欠分割 under_segmentation

定义：

```text
一个 pred 与多个 GT 有显著重叠。
```

建议规则：

```text
一个 pred 与 ≥2 个 GT 的 IoU 或 overlap_ratio 超过 weak_overlap_threshold；
且这些 GT 的 union 与该 pred 有较大重合。
```

ROI core：

```text
pred mask + matched GT union bbox
```

推荐标签：

```text
merged_crowns
oversized_crown
dense_crown_adhesion
crown_overlap
large_crown_abnormality
```

------

## 10.5 过分割 over_segmentation

定义：

```text
多个 pred 对应同一个 GT。
```

建议规则：

```text
一个 GT 与 ≥2 个 pred 有显著重叠；
多个 pred 的 union 覆盖该 GT。
```

ROI core：

```text
GT mask + related pred union bbox
```

推荐标签：

```text
over_split_crown
duplicate_detection
fragmented_boundary
same_crown_multi_instance
tile_edge_fragment
```

------

# 11. V1 几何指标与失败标签

新增：

```text
evaluation_analysis/geometry_metrics.py
evaluation_analysis/geometry_failure_tags.py
evolution/geometry/geometry_review_adapter.py
```

## 11.1 V1 保留几何指标

| 类型       | 指标                        | 作用                 |
| ---------- | --------------------------- | -------------------- |
| 单实例形状 | `area`                      | 判断异常小、异常大   |
| 单实例形状 | `equivalent_diameter`       | 对应冠幅尺度         |
| 单实例形状 | `axis_ratio`                | 判断细长误检         |
| 单实例形状 | `compactness`               | 判断形状紧凑性       |
| 单实例形状 | `circularity`               | 判断冠形规则程度     |
| 单实例形状 | `solidity`                  | 判断孔洞、凹陷、破碎 |
| 边界质量   | `boundary_complexity`       | 判断边界破碎         |
| 边界质量   | `hole_count`                | 判断异常孔洞         |
| 边界质量   | `hole_area_ratio`           | 判断 mask 破碎       |
| 空间关系   | `nearest_neighbor_distance` | 判断异常密集或空洞   |
| 空间关系   | `overlap_ratio`             | 判断重复检测         |
| 空间关系   | `touching_ratio`            | 判断粘连             |
| 空间关系   | `local_density`             | 判断局部密度异常     |

暂不进入 V1 的指标：

```text
voronoi_area_cv
cluster_index
chamfer_distance
vertex_density
convexity
```

这些留到 V2。

------

## 11.2 V1 几何失败标签

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

## 11.3 几何阈值策略

V1 支持两种模式：

```text
fixed_threshold
adaptive_percentile
```

默认建议：

```text
tiny = area < P5(area)
oversized = area > P95(area)
elongated = axis_ratio > P95(axis_ratio)
fragmented = boundary_complexity > P90(boundary_complexity)
```

注意：

```text
几何标签不替代 GT 错误类型；
几何标签只补充解释、严重度和训练价值判断。
```

------

# 12. V1 ROI 分级规则

新增：

```text
evolution/roi/roi_status_assigner.py
```

## 12.1 ROI 状态字段

| 字段                | 取值                                     |
| ------------------- | ---------------------------------------- |
| `review_status`     | `record_only` / `monitor` / `actionable` |
| `expert_eligible`   | true / false                             |
| `training_eligible` | true / false                             |
| `distill_eligible`  | true / false                             |

## 12.2 分级逻辑

```text
轻微错误：
  review_status = record_only
  expert_eligible = false

证据明确但未达专家触发：
  review_status = monitor
  expert_eligible = false

达到专家触发：
  review_status = actionable
  expert_eligible = true

有 GT 且错误明确：
  training_eligible = true

专家结果 accept 或 partial_accept 后：
  distill_eligible = true
```

## 12.3 专家触发条件

```yaml
roi_policy:
  expert_tile_size_px: 1024
  fusion_buffer_px: 64
  threshold_mode: adaptive_percentile

  minor_error_policy:
    max_instances: 2
    max_failure_area_ratio_in_tile: 0.005
    max_severity_score: 0.35
    default_review_status: record_only

  min_trigger_per_tile:
    min_failure_instances: 3
    min_failure_area_ratio: 0.01
    min_affected_tree_ratio: 0.10

  tiny_fragment_policy:
    clustered_tiny_roi_min_count: 5
    clustered_tiny_roi_min_area_ratio: 0.01

  severe_failure_override:
    enabled: true
    min_severity_score: 0.75
    min_single_failure_area_ratio: 0.03
    min_affected_tree_ratio: 0.15

  global_failure_guard:
    max_tile_clusters_per_block: 8
    max_failure_area_ratio_in_block: 0.15
    tiny_roi_count_global_threshold: 30
    tiny_roi_spread_area_ratio_threshold: 0.20
    action: main_plan_repair
```

------

# 13. V1 ROI 聚类与 ExpertTask 构建

新增：

```text
evolution/roi/roi_clusterer.py
evolution/expert/expert_task_builder.py
```

## 13.1 聚类原则

如果 COCO image 本身是 1024×1024：

```text
一张 image = 一个 expert tile；
同一 image 内同一 failure_family 的 ROI 聚合成一个 ExpertTask。
```

如果图像大于 1024×1024：

```text
以最高 severity ROI 为 anchor；
生成 1024×1024 expert tile；
聚合同 tile、同 failure_family 的 ROI。
```

## 13.2 ROI、Expert tile、Fusion mask 三者关系

```text
ROI core：问题发生的位置；
Expert tile：专家模型看的上下文；
Fusion mask：专家结果允许修改的区域。
```

## 13.3 failure family

```yaml
failure_families:
  small_crown_recall:
    - false_negative
    - small_crown_miss

  false_positive_cleanup:
    - false_positive
    - tiny_false_positive
    - shadow_false_positive
    - background_texture_false_positive

  crown_split:
    - under_segmentation
    - merged_crowns
    - oversized_crown

  crown_merge_cleanup:
    - over_segmentation
    - over_split_crown
    - duplicate_detection
    - fragmented_boundary

  boundary_refinement:
    - unstable_edge_mask
    - high_boundary_complexity
    - boundary_offset
```

------

# 14. V1 专家路由

新增：

```text
planning/scheduler/expert_routing_policy.py
```

V1 固定规则：

```python
ROUTE_MAP = {
    "under_segmentation": "htc",
    "over_segmentation": "mask2former",
    "false_positive": "cascade_mask_rcnn",
    "false_negative": "maskdino",
}
```

配置写法：

```yaml
expert_routing_policy:
  version: v1_rule_based
  route_map:
    under_segmentation: htc
    over_segmentation: mask2former
    false_positive: cascade_mask_rcnn
    false_negative: maskdino
```

注意：

```text
V1 只记录路由结果；
不学习路由权重；
不自动改变 route_map；
不根据单次结果判断哪个专家更强。
```

V2 才升级为：

```yaml
expert_routing_policy:
  version: v2_score_based
  score_components:
    historical_success_rate: 0.35
    geometry_improvement_gain: 0.25
    coco_metric_gain: 0.25
    replay_safety_score: 0.15
```

------

# 15. V1 专家执行模式

新增：

```text
evolution/expert/expert_task_runner.py
segmentation/adapters/mock_expert.py
segmentation/adapters/replay_expert.py
```

## 15.1 real 模式

```text
调用真实专家模型。
```

适用于专家模型已经部署完成的情况。

## 15.2 mock 模式

```text
用于跑通闭环。
```

可选策略：

```text
use_gt_or_perturbed_gt
copy_main_result
simulate_improvement
simulate_failure
```

V1 初始建议：

```text
use_gt_or_perturbed_gt
```

但必须记录：

```json
{
  "execution_mode": "mock",
  "oracle_mock": true
}
```

避免和真实专家结果混淆。

## 15.3 replay 模式

```text
从已有专家预测文件读取结果。
```

适用于：

```text
专家模型已经离线跑过；
当前只测试 ROI / expert review / fusion / trajectory 逻辑。
```

------

# 16. V1 专家结果评估

新增：

```text
evaluation_analysis/expert_result_comparator.py
evolution/expert/expert_review_adapter.py
```

## 16.1 ROI 内比较指标

专家结果与主模型结果在 ROI 内比较：

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

## 16.2 ROI 外安全检查

检查专家是否破坏 ROI 外结果：

```text
ROI 外新增 FP 是否过多；
ROI 外原本 TP 是否被破坏；
实例数量是否异常变化；
是否引入新严重几何异常；
是否出现大面积无关 mask 改动。
```

## 16.3 决策规则

### accept

```text
ROI 内目标错误明显减少；
总体 IoU / boundary_iou 改善；
ROI 外无明显副作用。
```

### partial_accept

```text
部分 ROI 改善；
部分 ROI 退化或无改善；
只融合 accepted_roi_ids。
```

### reject

```text
ROI 内无改善；
或者 ROI 外副作用明显；
或者专家结果比主模型更差。
```

### record_uncertain

```text
指标改善不稳定；
证据不足；
不融合，只记录。
```

------

# 17. V1 融合与回滚

新增：

```text
evolution/fusion/local_roi_fusion.py
```

融合原则：

```text
专家看完整 tile；
系统只改 ROI mask 或 ROI mask + buffer；
ROI 外永远保留主模型；
融合后必须重新评估；
退化则回滚。
```

## 17.1 accept

```text
ROI mask + buffer 内采用专家结果；
ROI 外保留主模型。
```

## 17.2 partial_accept

```text
只融合 accepted_roi_ids；
rejected_roi_ids 保留主模型。
```

## 17.3 reject

```text
完全回滚，保留主模型。
```

## 17.4 record_uncertain

```text
不融合，只记录。
```

## 17.5 融合后校验

```text
fused_result_eval >= main_result_eval + min_improvement_epsilon
```

如果不满足：

```text
rollback_to_main
```

------

# 18. V1 Trajectory

新增：

```text
evolution/trajectory/trajectory_builder.py
evolution/trajectory/trajectory_writer.py
```

每张 COCO image 生成一条 trajectory。

```json
{
  "trajectory_id": "traj_xxx",
  "run_id": "run_xxx",
  "image_id": "xxx",
  "mode": "supervised_coco_evolve_v1",
  "mainline_profile": "A_DOM_ONLY",

  "input_snapshot": {
    "image_path": "...",
    "annotation_json": "...",
    "gt_instance_count": 35
  },

  "main_model_stage": {
    "model_id": "legacy_cellpose_sam",
    "execution_result": {},
    "prediction_artifacts": {}
  },

  "main_eval_stage": {
    "coco_metrics": {},
    "error_decomposition": {}
  },

  "geometry_review_stage": {
    "geometry_profile": {},
    "failure_tags": []
  },

  "main_decision_stage": {
    "decision": "escalate_expert",
    "reason": "actionable ROI clusters found"
  },

  "roi_stage": {
    "roi_candidates": [],
    "roi_clusters": []
  },

  "expert_task_stage": {
    "expert_tasks": [],
    "routing_events": []
  },

  "expert_review_stage": {
    "expert_reviews": []
  },

  "fusion_stage": {
    "fusion_events": [],
    "final_result_source": "main_only | expert_fused | partial_expert_fused | rollback_to_main"
  },

  "pending_review_candidates": {
    "memory_candidates": [],
    "skill_candidates": [],
    "training_candidates": [],
    "distillation_candidates": []
  },

  "review_status": "pending"
}
```

Trajectory 是后续 V2/V3 的统一数据资产。

------

# 19. V1 SQLite 状态库

新增：

```text
state/
  db.py
  schema.sql
  repositories.py
  artifact_store.py
```

SQLite 只保存：

```text
run 状态
trajectory 索引
ROI 元数据
ExpertTask 元数据
专家审查结果
融合事件
训练候选
artifact 路径
```

不保存：

```text
GeoTIFF
mask
GPKG
大 JSON
模型权重
影像本体
```

## 19.1 V1 最小表

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    mainline_profile TEXT NOT NULL,
    config_path TEXT,
    output_dir TEXT,
    status TEXT NOT NULL,
    summary_json TEXT
);

CREATE TABLE trajectories (
    trajectory_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    main_model TEXT,
    final_status TEXT,
    final_result_source TEXT,
    trajectory_path TEXT NOT NULL,
    review_status TEXT NOT NULL
);

CREATE TABLE roi_candidates (
    roi_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    image_id TEXT,
    level1_error_type TEXT,
    failure_family TEXT,
    severity_score REAL,
    confidence_level TEXT,
    review_status TEXT,
    expert_eligible INTEGER,
    training_eligible INTEGER,
    distill_eligible INTEGER,
    bbox_json TEXT,
    tags_json TEXT,
    geometry_json TEXT
);

CREATE TABLE expert_tasks (
    expert_task_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    expert_model TEXT,
    failure_family TEXT,
    level1_error_type TEXT,
    roi_ids_json TEXT,
    tile_window_json TEXT,
    status TEXT,
    trigger_reason_json TEXT
);

CREATE TABLE expert_reviews (
    review_id TEXT PRIMARY KEY,
    expert_task_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    improvement_json TEXT,
    safety_json TEXT,
    accepted_roi_ids_json TEXT,
    rejected_roi_ids_json TEXT
);

CREATE TABLE fusion_events (
    fusion_event_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    decision TEXT,
    fused_result_path TEXT,
    summary_json TEXT
);

CREATE TABLE training_candidates (
    candidate_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    roi_id TEXT,
    sample_type TEXT,
    target_model_role TEXT,
    failure_category TEXT,
    quality_status TEXT,
    approved INTEGER DEFAULT 0,
    artifact_refs_json TEXT
);

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    trajectory_id TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    format TEXT,
    metadata_json TEXT,
    sha256 TEXT
);
```

------

# 20. V1 training_loop 的位置

V1 可以新建 `training_loop/`，但不能启动真实训练。

```text
training_loop/
  contracts.py
  sample_intake.py
  trigger_policy.py
```

## 20.1 V1 功能

```text
1. 接收 training_eligible ROI；
2. 生成 pending training candidate；
3. 检查是否满足最低训练触发条件；
4. 输出 dry-run trigger result；
5. 不启动训练；
6. 不改模型权重。
```

## 20.2 输出示例

```json
{
  "candidate_id": "traincand_xxx",
  "trajectory_id": "traj_xxx",
  "roi_id": "roi_xxx",
  "failure_category": "false_negative",
  "target_model_role": "main_model",
  "quality_status": "pending_review",
  "trigger_training": false,
  "reason": "V1 only supports dry-run training candidate intake."
}
```

------

# 21. V1 CLI

V1 开放 3 个命令。

## 21.1 主命令

```bash
itd-agent evolve-infer --config configs/examples/itd_agent_evolve_coco_v1.yaml
```

用途：

```text
运行 COCO/公开标注数据集上的主—专家可监督推理闭环。
```

## 21.2 状态查询

```bash
itd-agent state summary
```

用途：

```text
查看 SQLite 中 runs、trajectories、ROI、expert tasks、artifacts 的统计。
```

## 21.3 待审查查询

```bash
itd-agent review pending
```

用途：

```text
查看 pending memory / skill / training / distillation candidates。
```

V1 不开放：

```bash
itd-agent train trigger
itd-agent model promote
itd-agent skill approve
itd-agent memory approve
```

这些留到 V2/V3。

------

# 22. V1 主流程伪代码

```python
def run_evolve_infer_v1(config_path: str) -> dict:
    cfg = load_evolve_config(config_path)
    run_ctx = create_run_context(cfg)

    samples = load_coco_samples(cfg)

    run_summary = []

    for sample in samples:
        trajectory = start_trajectory(run_ctx, sample)

        main_plan = build_main_plan(cfg, sample)
        trajectory.add_stage("main_plan", main_plan)

        main_result = run_main_model(
            cfg=cfg,
            sample=sample,
            model_id=cfg["main_model"]["model_id"],
        )
        trajectory.add_stage("main_model_stage", main_result)

        error_decomp = decompose_coco_errors(
            gt_instances=sample.gt_instances,
            pred_instances=main_result.instances,
            iou_threshold=cfg["evaluation"]["matching"]["iou_threshold"],
            weak_overlap_threshold=cfg["evaluation"]["matching"]["weak_overlap_threshold"],
        )

        geometry_review = build_geometry_review(
            pred_instances=main_result.instances,
            gt_instances=sample.gt_instances,
            threshold_mode=cfg["evaluation"]["geometry"]["threshold_mode"],
        )

        trajectory.add_stage("main_eval_stage", error_decomp)
        trajectory.add_stage("geometry_review_stage", geometry_review)

        main_decision = decide_main_action(
            error_decomposition=error_decomp,
            geometry_review=geometry_review,
            adaptive_policy=cfg["adaptive_inference"],
        )
        trajectory.add_stage("main_decision_stage", main_decision)

        if main_decision.action == "accept_main":
            final_result = freeze_main_result(main_result)

        elif main_decision.action == "retry_main_plan":
            repaired_plan = repair_main_plan(
                main_plan=main_plan,
                error_decomposition=error_decomp,
                geometry_review=geometry_review,
            )
            repaired_result = run_main_model_with_plan(repaired_plan)

            repaired_eval = decompose_coco_errors(
                gt_instances=sample.gt_instances,
                pred_instances=repaired_result.instances,
                iou_threshold=cfg["evaluation"]["matching"]["iou_threshold"],
                weak_overlap_threshold=cfg["evaluation"]["matching"]["weak_overlap_threshold"],
            )

            if should_accept_repaired_result(error_decomp, repaired_eval):
                final_result = freeze_main_result(repaired_result)
            else:
                final_result = freeze_main_result(main_result)

        elif main_decision.action == "escalate_expert":
            final_result = run_expert_stage_v1(
                cfg=cfg,
                run_ctx=run_ctx,
                sample=sample,
                main_result=main_result,
                error_decomp=error_decomp,
                geometry_review=geometry_review,
                trajectory=trajectory,
            )

        else:
            final_result = freeze_failure_result(main_result, error_decomp)

        pending_candidates = build_pending_review_candidates(
            trajectory=trajectory,
            final_result=final_result,
        )

        dry_run_training_candidates = intake_training_candidates_dry_run(
            pending_candidates=pending_candidates,
        )

        trajectory.add_stage("pending_review_candidates", {
            "pending_candidates": pending_candidates,
            "dry_run_training_candidates": dry_run_training_candidates,
        })

        trajectory.finalize(final_result)

        write_trajectory(trajectory)
        write_state_records(trajectory)
        register_artifacts(trajectory)

        run_summary.append(summarize_trajectory(trajectory))

    return build_run_summary(run_summary)
```

------

# 23. V1 专家阶段伪代码

```python
def run_expert_stage_v1(
    cfg,
    run_ctx,
    sample,
    main_result,
    error_decomp,
    geometry_review,
    trajectory,
):
    roi_candidates = build_roi_candidates(
        sample=sample,
        main_result=main_result,
        error_decomposition=error_decomp,
        geometry_review=geometry_review,
    )

    roi_candidates = assign_roi_status(
        roi_candidates=roi_candidates,
        roi_policy=cfg["roi_policy"],
    )

    trajectory.add_stage("roi_candidates", roi_candidates)

    if is_global_failure(roi_candidates, cfg["roi_policy"]):
        trajectory.add_stage("global_failure_guard", {
            "action": "main_plan_repair"
        })
        return freeze_main_result(main_result)

    roi_clusters = cluster_rois_for_expert_tiles(
        roi_candidates=roi_candidates,
        image_size=(sample.width, sample.height),
        tile_size=cfg["roi_policy"]["expert_tile_size_px"],
    )

    trajectory.add_stage("roi_clusters", roi_clusters)

    expert_tasks = build_expert_tasks(
        roi_clusters=roi_clusters,
        routing_policy=cfg["expert_routing_policy"],
        execution_mode=cfg["expert_models"]["execution_mode"],
    )

    trajectory.add_stage("expert_tasks", expert_tasks)

    expert_results = run_expert_tasks(
        expert_tasks=expert_tasks,
        execution_mode=cfg["expert_models"]["execution_mode"],
    )

    trajectory.add_stage("expert_results", expert_results)

    expert_reviews = compare_expert_with_main(
        sample=sample,
        main_result=main_result,
        expert_results=expert_results,
        expert_tasks=expert_tasks,
    )

    trajectory.add_stage("expert_reviews", expert_reviews)

    fused_result = fuse_or_rollback(
        main_result=main_result,
        expert_results=expert_results,
        expert_reviews=expert_reviews,
        fusion_policy=cfg["fusion"],
    )

    trajectory.add_stage("fusion", fused_result)

    return freeze_result(fused_result)
```

------

# 24. V1 开发顺序

必须按这个顺序开发：

```text
1. schemas
2. state/artifact_store
3. coco_error_decomposition
4. geometry_metrics + geometry_failure_tags
5. roi_candidate_builder
6. roi_status_assigner
7. roi_clusterer
8. expert_routing_policy
9. expert_task_builder
10. mock/replay expert adapter
11. expert_result_comparator
12. local_roi_fusion
13. trajectory_builder/writer
14. sample_intake dry-run
15. evolve_infer_runner
16. cli evolve-infer
```

不要先写：

```text
training execution
skill auto update
memory auto write
model promotion
expert-to-main distillation
PostGIS
复杂真实 DOM-only 无 GT 判断
```

------

# 25. V1 验收标准

V1 完成后，必须能回答以下问题。

## 25.1 主模型是否跑通？

```text
每张 COCO image 是否生成 main prediction？
是否记录 execution_result？
是否保存 artifact？
```

## 25.2 四类错误是否能识别？

```text
false_negative 数量是多少？
false_positive 数量是多少？
under_segmentation 数量是多少？
over_segmentation 数量是多少？
```

## 25.3 ROI 是否正确生成？

```text
每个错误是否生成 ROI candidate？
ROI 是否包含 affected_gt_ids？
ROI 是否包含 affected_pred_ids？
ROI 是否有 bbox_px？
ROI 是否有 severity_score？
ROI 是否有 failure_family？
```

## 25.4 ExpertTask 是否能构建？

```text
actionable ROI 是否能聚类？
是否生成 1024×1024 expert tile？
是否有 expert_model？
是否有 trigger_reason？
是否记录 execution_mode？
```

## 25.5 专家结果是否能评估？

```text
专家结果是否和主模型在 ROI 内比较？
是否输出 accept / partial_accept / reject？
是否进行 ROI 外安全检查？
```

## 25.6 融合/回滚是否可靠？

```text
accept 是否局部融合？
partial_accept 是否只融合 accepted ROI？
reject 是否回滚？
ROI 外是否保留主模型？
融合后是否重新评估？
退化是否回滚？
```

## 25.7 trajectory 是否完整？

```text
是否记录主模型阶段？
是否记录评估阶段？
是否记录几何审查阶段？
是否记录 ROI 阶段？
是否记录专家任务阶段？
是否记录专家评估阶段？
是否记录融合阶段？
是否记录 pending candidates？
```

## 25.8 SQLite 是否可查？

```text
runs 有记录；
trajectories 有记录；
roi_candidates 有记录；
expert_tasks 有记录；
expert_reviews 有记录；
fusion_events 有记录；
training_candidates 有记录；
artifacts 有记录。
```

------

# 26. V1 最终判断

V1 正式方案应定义为：

```text
COCO/公开标注数据集上的可监督主—专家推理闭环验证版。
```

它不负责让模型真正进化，而是让 ITD_agent 第一次具备：

```text
发现主模型错误
→ 结构化生成 ROI
→ 构建专家任务
→ 验证专家是否有效
→ 局部融合或回滚
→ 记录完整 trajectory
→ 生成后续进化候选
```



