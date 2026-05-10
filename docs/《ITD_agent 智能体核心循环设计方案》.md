

# 《ITD_agent 智能体核心循环总体设计方案》

## 0. 方案定位

本方案面向 **ITD_agent 主线 A：DOM-only 条件下的单木树冠检测与实例分割智能体闭环**。

暂时不考虑主线 B 中的 DEM、CHM、DSM、外部知识、小班清查等增强输入。本方案只围绕：

```text
DOM / COCO 公开数据集
主模型
专家模型
ROI
几何评估
Trajectory
Memory / Skill / Finetune Pool
Training Loop
```

构建最终总体架构。

本项目的目标不是单纯训练一个分割模型，也不是简单堆叠多个 SOTA 模型，而是构建一个：

```text
面向 DOM-only 单木树冠检测与提取任务的
主模型—专家模型自进化智能体系统
```

它的核心思想是：

```text
主模型负责稳定、通用、大范围推理；
专家模型负责针对主模型暴露出的稳定短板进行局部纠错；
ROI 负责结构化记录主模型失败区域；
Trajectory 负责记录完整推理轨迹；
后台审查负责决定哪些经验进入 memory、skill、finetune_pool 和 training_loop；
训练闭环负责把高质量失败样本和专家成功样本反哺模型权重。
```

------



# 1. 总体架构

最终系统由 10 个核心层组成：

```text
1. Input Layer
2. Data Processing Layer
3. Planning / Scheduler Layer
4. Segmentation Execution Layer
5. Evaluation Analysis Layer
6. Evolution Loop Layer
7. State / Artifact Layer
8. Memory / Skill / Finetune Pool Layer
9. Training Loop Layer
10. Output Layer
```

整体流程为：

```text
DOM / COCO 输入
  ↓
InputManifest / 主线 A Profile 校验
  ↓
影像画像、tile/block 准备、数据处理摘要
  ↓
主模型推理计划
  ↓
主模型推理
  ↓
主模型评估 + 几何质量审查
  ↓
主模型决策
    ├── 直接接受
    ├── 修复主模型计划后重推
    ├── 提取 ROI 并调用专家模型
    └── 记录失败，进入后台审查
  ↓
ROI candidate 构建
  ↓
ROI 分级、聚类、ExpertTask 构建
  ↓
专家模型路由
  ↓
专家模型推理
  ↓
专家结果评估
  ↓
局部融合 / 部分融合 / 回滚
  ↓
冻结最终候选结果
  ↓
写入 Inference Evolution Trajectory
  ↓
写入 SQLite 状态库 + Artifact Registry
  ↓
后台审查
    ├── Memory Review
    ├── Skill Review
    ├── Finetune Sample Review
    ├── Routing Review
    └── Distillation Review
  ↓
训练触发 / 模型晋级 / 专家能力反哺主模型
```

------

# 2. 核心循环定义

最终智能体核心循环命名为：

```text
Main–Expert Adaptive Evolution Loop
主模型—专家模型自适应进化循环
```

它不是线性流程，而是一个受限状态机：

```text
START
  ↓
CONTEXT_LOAD
  ↓
MAIN_PLAN
  ↓
MAIN_INFER
  ↓
MAIN_EVAL
  ↓
GEOMETRY_REVIEW
  ↓
MAIN_DECISION
    ├── ACCEPT_MAIN → FINALIZE
    ├── RETRY_MAIN_PLAN → MAIN_PLAN_REPAIR → MAIN_INFER
    ├── ESCALATE_EXPERT → ROI_BUILD
    └── RECORD_FAILURE → FINALIZE_WITH_FAILURE

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
    ├── ACCEPT → LOCAL_FUSION
    ├── PARTIAL_ACCEPT → PARTIAL_LOCAL_FUSION
    ├── REJECT → ROLLBACK_TO_MAIN
    ├── RETRY_EXPERT_PLAN → EXPERT_ROUTE
    └── RECORD_UNCERTAIN → KEEP_MAIN_RESULT

FINALIZE
  ↓
TRAJECTORY_WRITE
  ↓
BACKGROUND_REVIEW_PENDING
  ↓
MEMORY / SKILL / FINETUNE / TRAINING LOOP
```

核心护栏：

```yaml
adaptive_inference:
  max_main_retries: 1
  max_expert_rounds: 1
  max_next_expert_trials: 0
  min_improvement_epsilon: 0.01
  allow_llm_direct_fusion_decision: false
  allow_llm_direct_training_trigger: false
```

LLM 只做：

```text
解释、摘要、候选建议、审查理由生成、经验压缩
```

LLM 不做：

```text
指标计算
最终融合裁决
训练触发硬判定
模型晋级硬判定
```

------

# 3. 模块边界

根据当前仓库 `docs/codemap.md`，项目已有核心包包括 `input_layer/`、`ITD_agent/orchestration/`、`ITD_agent/data_processing/`、`ITD_agent/evaluation_analysis/`、`ITD_agent/llm_gateway/`、`ITD_agent/planning/`、`ITD_agent/segmentation/`、`ITD_agent/memory_store/`、`ITD_agent/finetune_pool/` 和 `output_layer/`。其中 `evaluation_analysis/` 已被定义为负责主模型、ROI、子模型、微调效果、参考质量、benchmark 和最终评估逻辑，并拥有指标公式、规则决策和派生质量/错误摘要。

因此最终边界如下。

------

## 3.1 input_layer/

职责：

```text
输入路径校验
schema 校验
InputManifest 构建
主线 A Profile 校验
COCO / public dataset 输入注册
输入工作区准备
```

不负责：

```text
ROI 生成
模型推理
指标计算
专家路由
训练触发
```

------

## 3.2 data_processing/

职责：

```text
DOM 画像
影像质量评估
tile/block 计划
实例后处理准备
ROI 数据裁剪准备
public dataset 摘要
```

不负责：

```text
四类错误判定
专家任务构建
主/专家结果融合裁决
训练闭环
```

------

## 3.3 evaluation_analysis/

职责：

```text
主模型评估
专家模型评估
COCO GT 匹配
四类错误分解
几何指标计算
几何失败标签
ROI 内专家改进评估
融合后质量审查
规则型质量判断
```

建议扩展：

```text
ITD_agent/evaluation_analysis/
  geometry_metrics.py
  geometry_failure_tags.py
  coco_error_decomposition.py
  expert_result_comparator.py
  fusion_quality_review.py
```

不负责：

```text
生成 ROI candidate
构建 ExpertTask
写 trajectory
写 memory
启动训练
```

------

## 3.4 segmentation/

职责：

```text
主模型执行
专家模型执行
模型注册
模型 adapter
模型推理结果契约
底层训练器入口
```

当前 `segmentation/executor.py` 已经通过 `SegmentationExecutionRequest` 和 `SegmentationExecutionResult` 统一记录模型执行请求和输出结果，应继续保留并扩展，而不是重写。

建议扩展：

```text
ITD_agent/segmentation/
  adapters/
    base.py
    legacy_cellpose_sam.py
    mmdet_instance.py
    mock_expert.py

  model_profiles/
    main_model_profiles.yaml
    expert_model_profiles.yaml
```

------

## 3.5 planning/scheduler/

职责：

```text
主模型推理计划
主模型计划修复
专家模型路由策略
ROI policy 加载
模型参数建议
```

建议扩展：

```text
ITD_agent/planning/scheduler/
  main_plan_builder.py
  main_plan_repair.py
  expert_routing_policy.py
  roi_policy_loader.py
```

------

## 3.6 evolution/

这是新增核心层。

职责：

```text
组织主—专家核心循环中的自进化对象
ROI candidate 构建
ROI 分级
ROI 聚类
ExpertTask 构建
专家任务运行组织
局部融合
Trajectory 构建
pending review 写入
```

注意：

```text
evolution 不重复计算指标；
evolution 调用 evaluation_analysis 的计算结果。
```

------

## 3.7 state/

职责：

```text
SQLite 状态库
artifact registry
run 状态
trajectory 索引
ROI 索引
ExpertTask 索引
训练候选索引
```

不直接存大影像、大矢量。

------

## 3.8 memory_store/

职责：

```text
长期经验
失败模式
成功策略
执行轨迹压缩摘要
run retrospective
```

但 memory_store 不能每次无条件写入。必须由后台审查批准。

------

## 3.9 finetune_pool/

职责：

```text
失败 ROI 样本池
replay 样本池
public dataset candidate
训练样本 bundle 输出
样本质量状态管理
```

它是数据池，不是训练编排器。

------

## 3.10 training_loop/

这是新增目录，应当建立。

职责：

```text
训练触发审查
训练计划构建
样本包组织
pilot 训练
正式训练
replay guard
训练后评估
模型晋级/拒绝/归档
专家能力向主模型蒸馏
```

当前公开分割模型微调脚本已经包含训练、读取 best checkpoint、finetuned inference、可选 rerun、对比微调前后结果等训练闭环雏形，后续应逐步迁入 `training_loop/`，scripts 只保留薄 wrapper。

------

# 4. 最终目录规范

```text
ITD_agent/
  input_layer/
    # 输入 manifest、schema、mainline A profile、COCO/public dataset 输入注册

  data_processing/
    imagery/
    terrain/
    inventory/
    knowledge/
    public_data/
    roi/
    fusion/

  evaluation_analysis/
    evaluator.py
    benchmark_engine.py
    online_quality_engine.py
    reference_quality_engine.py
    decision_flags.py

    geometry_metrics.py
    geometry_failure_tags.py
    coco_error_decomposition.py
    expert_result_comparator.py
    fusion_quality_review.py

  segmentation/
    contracts.py
    executor.py
    model_registry/
    model_training/
    finetuning/

    adapters/
      base.py
      legacy_cellpose_sam.py
      mmdet_instance.py
      mock_expert.py

    model_profiles/
      main_model_profiles.yaml
      expert_model_profiles.yaml

  planning/
    scheduler/
      main_plan_builder.py
      main_plan_repair.py
      expert_routing_policy.py
      roi_policy_loader.py
      context_builder.py
      expert_taxonomy.py

  evolution/
    schemas/
      roi.py
      expert_task.py
      trajectory.py
      decisions.py

    geometry/
      geometry_review_adapter.py
      geometry_evidence_builder.py

    roi/
      roi_candidate_builder.py
      roi_status_assigner.py
      roi_family_mapper.py
      roi_clusterer.py
      expert_tile_builder.py

    expert/
      expert_task_builder.py
      expert_router.py
      expert_task_runner.py
      expert_review_adapter.py

    fusion/
      local_roi_fusion.py
      fusion_conflict_resolver.py

    trajectory/
      trajectory_builder.py
      trajectory_writer.py
      trajectory_reader.py

    review/
      pending_review_writer.py
      memory_review.py
      skill_review.py
      training_review.py
      routing_review.py
      distillation_review.py

  memory_store/
    traces/
    success_strategies/
    failure_patterns/
    retrospectives/
    pending_reviews.py

  finetune_pool/
    samples/
    replay/
    public_candidates/
    bundle_export/
    recommendation.py

  training_loop/
    contracts.py
    sample_intake.py
    dataset_packager.py
    trigger_policy.py
    training_plan_builder.py
    pilot_trainer.py
    formal_trainer.py
    replay_guard.py
    post_train_evaluator.py
    model_promotion.py
    expert_to_main_distill.py

  state/
    db.py
    schema.sql
    repositories.py
    artifact_store.py
    migrations/

  orchestration/
    orchestrator.py
    evolve_infer_runner.py
    runtime_paths.py
    runtime_steps.py
    summary_builder.py

  cli/
    main.py
    evolve_infer_cmd.py
    infer_cmd.py
    review_cmd.py
    state_cmd.py
    train_cmd.py
    model_cmd.py

output_layer/
  # 最终成果发布、可视化、报告、交付物
```

------

# 5. 数据存储方案

## 5.1 总体原则

SQLite 不存影像本体、不存大型矢量本体、不存大 mask。

最终采用：

```text
SQLite 状态库 + 文件系统 Artifact Store
```

后续可扩展：

```text
DuckDB：用于大规模 CSV / Parquet 指标分析
PostgreSQL + PostGIS：用于生产级空间检索、多 run 空间查询、多用户服务
```

但当前总体方案中，默认使用：

```text
SQLite + GeoTIFF/COG + GPKG/GeoJSON + JSON/CSV/Parquet
```

------

## 5.2 文件系统存储

推荐输出结构：

```text
outputs/evolve_runs/{run_id}/
  config/
    runtime_config.yaml
    normalized_config.yaml

  input/
    input_manifest.json

  main_model/
    main_prediction.tif
    main_instances.gpkg
    main_execution_result.json
    main_eval.json

  geometry/
    geometry_profile.json
    geometry_failure_tags.json

  roi/
    roi_candidates.gpkg
    roi_candidates.json
    roi_clusters.json

  expert_tasks/
    expert_task_0001/
      expert_tile.tif
      valid_mask.tif
      roi_masks.gpkg
      expert_prediction.tif
      expert_instances.gpkg
      expert_review.json

  fusion/
    fused_prediction.tif
    fused_instances.gpkg
    fusion_summary.json

  trajectory/
    trajectory.json

  review/
    pending_reviews.json

  training_candidates/
    candidates.json

  reports/
    summary.json
    metrics.json
    details.csv
```

------

## 5.3 推荐格式

| 数据类型          | 推荐格式           |
| ----------------- | ------------------ |
| DOM / tile / mask | GeoTIFF / COG      |
| 实例矢量          | GeoPackage `.gpkg` |
| ROI 矢量          | GeoPackage `.gpkg` |
| 调试矢量          | GeoJSON            |
| 大规模指标表      | Parquet            |
| 普通指标表        | CSV                |
| trajectory        | JSON               |
| run summary       | JSON               |
| 可视化            | PNG                |

------

## 5.4 SQLite 表

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

CREATE TABLE review_events (
    review_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    review_type TEXT NOT NULL,
    decision TEXT NOT NULL,
    candidate_json TEXT,
    approved INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
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

# 6. 主模型机制

## 6.1 主模型定位

主模型是：

```text
DOM-only 条件下的大范围、通用、稳定、可复现推理入口。
```

第一版默认主模型：

```text
legacy_cellpose_sam
```

它的职责：

```text
1. 对 DOM / COCO tile 执行基础实例分割；
2. 输出完整初始结果；
3. 暴露欠分割、过分割、误检、漏检四类错误；
4. 为 ROI 提取和专家模型介入提供空间证据；
5. 为后续训练闭环提供失败样本来源。
```

主模型不是永远最强模型，而是当前闭环中的稳定入口。未来主模型是否替换，应由：

```text
COCO benchmark
真实 DOM-only 几何审查
replay 回归
专家修正收益
模型晋级机制
```

共同决定。

------

## 6.2 主模型计划

主模型计划包括：

```json
{
  "stage": "main_model_plan",
  "main_model": "legacy_cellpose_sam",
  "model_role": "main_model",
  "input_mode": "A_DOM_ONLY",
  "tile_size": 1024,
  "runtime_params": {
    "score_threshold": 0.35,
    "mask_threshold": 0.5,
    "merge_iou_threshold": 0.3
  },
  "fallback_policy": {
    "enable_main_retry": true,
    "enable_expert_escalation": true
  }
}
```

------

## 6.3 主模型决策

主模型评估后产生四种动作：

| 动作              | 含义                                       |
| ----------------- | ------------------------------------------ |
| `accept_main`     | 主模型结果达标，直接冻结                   |
| `retry_main_plan` | 问题更像参数、tile、后处理问题，先修复计划 |
| `escalate_expert` | 局部问题明确，进入 ROI 和专家模型流程      |
| `record_failure`  | 证据不足或失败不可自动修正，仅记录         |

优先修复主模型计划的情况：

```text
tile 边缘效应
postprocess threshold 问题
scale parameter 问题
全局小碎片异常
全局漏检
全局误检
```

这类问题不应一开始就交给专家模型。

------

# 7. 专家模型机制

## 7.1 专家模型定位

专家模型是：

```text
针对主模型稳定短板的局部纠错和能力补强模块。
```

专家模型不是备用模型，不是简单多模型投票，而是差异化能力模块。

------

## 7.2 初始专家路由

初始规则：

| 一级错误类型  | 首选专家模型       | 定位                                       |
| ------------- | ------------------ | ------------------------------------------ |
| 欠分割        | HTC                | 粘连树冠拆分、merged crown correction      |
| 过分割        | Mask2Former        | 碎片合并、区域一致性修正                   |
| 误检          | Cascade Mask R-CNN | false positive cleanup                     |
| 漏检          | MaskDINO           | missed crown recall                        |
| mask 质量校准 | Mask Scoring R-CNN | 暂不进入第一版专家池，后续作为质量校准专家 |

配置：

```yaml
expert_routing_policy:
  version: v1_rule_based
  default_main_model: legacy_cellpose_sam

  expert_map:
    under_segmentation:
      primary_expert: htc
    over_segmentation:
      primary_expert: mask2former
    false_positive:
      primary_expert: cascade_mask_rcnn
    false_negative:
      primary_expert: maskdino
```

注意：

```text
这只是初始人工先验，不是永久科学定论。
```

长期应升级为：

```text
人工先验 + 历史成功率 + 几何改善收益 + replay 安全性 + COCO 指标收益
```

------

## 7.3 专家模型推理原则

```text
专家模型看完整 1024×1024 expert tile；
系统只允许专家修改 ROI mask 或 ROI mask + buffer；
ROI 外默认保留主模型结果。
```

如果 ROI 靠近边界：

```text
允许 padding；
必须记录 valid mask；
valid mask 外不参与评估、不允许融合。
```

------

# 8. ROI 机制

## 8.1 ROI 定位

ROI 不是简单裁剪图像，而是：

```text
主模型推理后对欠分割、过分割、误检和漏检四类问题区域的结构化证据。
```

ROI 服务五件事：

```text
1. 定位主模型哪里失败；
2. 判断失败属于哪一类；
3. 决定是否专家介入；
4. 为专家模型构建 1024×1024 输入 tile；
5. 为 memory、skill、finetune_pool、training_loop、distillation 提供样本证据。
```

------

## 8.2 ROI 一级类型

只保留四类：

| 一级类型             | 含义                       |
| -------------------- | -------------------------- |
| `false_negative`     | 漏检，GT 有树冠但预测没有  |
| `false_positive`     | 误检，预测有实例但 GT 没有 |
| `under_segmentation` | 欠分割，多个树冠被合并     |
| `over_segmentation`  | 过分割，一个树冠被拆碎     |

------

## 8.3 ROI 二级问题

| 一级错误 | 二级问题                                                 |
| -------- | -------------------------------------------------------- |
| 欠分割   | 相邻树冠重叠、冠内多峰、大冠异常、边界粘连、高郁闭度合并 |
| 过分割   | 冠内碎片、重复检测、边界破碎、同冠多实例、tile 边缘断裂  |
| 误检     | 灌木误检、阴影误检、背景纹理误检、异常小斑块、低置信噪声 |
| 漏检     | 小冠漏检、阴影漏检、低对比度漏检、密集区漏检、边缘漏检   |

------

## 8.4 原因标签

```text
stand_density
crown_size
crown_overlap
shadow
low_contrast
background_noise
scale_parameter
training_sample_gap
postprocess_threshold
tiling_context
edge_boundary_artifact
annotation_uncertainty
model_domain_shift
```

因为当前只考虑主线 A，所以暂不使用：

```text
missing_height_information
terrain
slope
aspect
CHM
DEM
```

这些留给主线 B。

------

## 8.5 ROI Schema

```python
@dataclass
class ROICandidate:
    roi_id: str
    image_id: str
    source: str  # coco_gt | geometry_anomaly | uncertainty | history_failure

    level1_error_type: str
    level2_problem_tags: list[str]
    reason_tags: list[str]
    failure_family: str

    bbox_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    area_px: float

    affected_pred_ids: list[str]
    affected_gt_ids: list[str]

    severity_score: float
    confidence_level: str  # confirmed | suspected | weak

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

## 8.6 ROI 分级

| 状态                     | 含义             | 动作                             |
| ------------------------ | ---------------- | -------------------------------- |
| `record_only`            | 问题轻微         | 只记录                           |
| `monitor`                | 有风险但证据不足 | 累积观察                         |
| `actionable`             | 明确且值得处理   | 可进入 ExpertTask                |
| `training_eligible=True` | 有训练价值       | 进入 training_loop sample intake |
| `distill_eligible=True`  | 专家修正成功     | 进入专家到主模型蒸馏候选         |

------

## 8.7 ROI 聚类

专家模型输入固定为：

```text
1024×1024 expert tile
```

所以 ROI 聚类规则是：

```text
ROI core 表示问题位置；
Expert tile 表示专家模型看什么；
Fusion mask 表示专家模型允许改哪里。
```

流程：

```text
1. 获取所有 ROI candidate；
2. 按 failure_family 分组；
3. 按 severity_score 排序；
4. 以高优先级 ROI 为 anchor；
5. 生成 1024×1024 expert tile；
6. 合并同 tile、同 family ROI；
7. 判断是否达到专家触发条件；
8. 达到则生成 ExpertTask；
9. 未达到则保持 record_only 或 monitor。
```

------

## 8.8 专家触发条件

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

阈值不能永久写死，应支持：

```text
fixed_threshold
adaptive_percentile_threshold
```

------

# 9. 几何评估机制

主线 A DOM-only 没有 DEM/CHM，所以几何评估是核心证据。

## 9.1 单实例形状指标

```text
area
equivalent_diameter
major_axis_length
minor_axis_length
axis_ratio
compactness
circularity
solidity
```

## 9.2 边界质量指标

```text
boundary_complexity
hole_count
hole_area_ratio
boundary_iou  # COCO 阶段
boundary_f_score  # 可选
```

## 9.3 空间关系指标

```text
nearest_neighbor_distance
overlap_ratio
touching_ratio
local_density
```

## 9.4 COCO 阶段误差指标

```text
mask_iou
boundary_iou
centroid_distance_error
area_error_ratio
diameter_error_ratio
false_positive_count
false_negative_count
under_split_count
over_split_count
```

## 9.5 几何失败标签

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

建议阈值：

```text
tiny_instance = area < P5(area_distribution)
oversized_crown = area > P95(area_distribution)
elongated_instance = axis_ratio > P95(axis_ratio_distribution)
abnormal_boundary = boundary_complexity > P90(boundary_complexity_distribution)
```

------

# 10. ExpertTask 机制

ExpertTask 是专家模型执行的基本单位。

```python
@dataclass
class ExpertTask:
    expert_task_id: str
    trajectory_id: str
    image_id: str
    expert_model: str

    failure_family: str
    level1_error_type: str
    roi_ids: list[str]

    tile_window_px: tuple[int, int, int, int]
    input_tile_path: str
    valid_mask_path: str | None

    fusion_scope: str  # roi_masks | roi_masks_plus_buffer
    trigger_reason: dict
```

每个 ExpertTask 包含：

```text
1024×1024 expert tile
ROI masks
failure family
目标专家模型
融合范围
触发原因
```

------

# 11. 专家结果评估与融合机制

## 11.1 专家结果评估

COCO 阶段评估：

```text
ROI 内 IoU 是否提升
Boundary IoU 是否提升
false positive 是否减少
false negative 是否减少
under_segmentation 是否改善
over_segmentation 是否改善
area_error 是否下降
diameter_error 是否下降
```

DOM-only 阶段评估：

```text
异常小实例是否减少
异常大实例是否减少
重复检测是否减少
边界复杂度是否下降
冠幅分布是否更合理
局部密度是否更稳定
```

## 11.2 决策

| 决策                | 动作                       |
| ------------------- | -------------------------- |
| `accept`            | 融合专家结果               |
| `partial_accept`    | 只融合改善 ROI             |
| `reject`            | 回滚，保留主模型           |
| `record_uncertain`  | 不融合，仅记录             |
| `retry_expert_plan` | 修正专家任务，默认最多一次 |

## 11.3 融合原则

```text
专家看整 tile；
系统只改 ROI；
ROI 外保留主模型；
融合后重新评估；
退化则回滚。
```

------

# 12. Trajectory 机制

Trajectory 是智能体自进化的核心数据资产。

命名：

```text
Inference Evolution Trajectory
```

每一次完整运行都必须生成一条 trajectory。

```json
{
  "trajectory_id": "traj_xxx",
  "run_id": "run_xxx",
  "mode": "A_DOM_ONLY_supervised_or_runtime",
  "mainline_profile": "A_DOM_ONLY",

  "input_snapshot": {},
  "context_snapshot": {},

  "main_model_stage": {
    "plan": {},
    "inference": {},
    "evaluation": {},
    "geometry_review": {},
    "decision": {}
  },

  "roi_stage": {
    "roi_candidates": [],
    "roi_clusters": [],
    "roi_policy": {}
  },

  "expert_stage": {
    "expert_tasks": [],
    "routing": {},
    "expert_results": [],
    "expert_reviews": []
  },

  "fusion_stage": {
    "fusion_events": [],
    "final_candidate_result": {}
  },

  "self_evolution_candidates": {
    "memory_candidates": [],
    "skill_candidates": [],
    "training_candidates": [],
    "distillation_candidates": [],
    "routing_update_candidates": []
  },

  "review_status": "pending"
}
```

Trajectory 后续进入：

```text
Memory Review
Skill Review
Training Review
Routing Review
Distillation Review
```

------

# 13. 后台审查机制

后台审查遵循：

```text
前台完成推理；
后台审查轨迹；
审查通过后才写长期记忆、更新 skill、进入训练池或触发训练。
```

不允许：

```text
每次运行都写长期 memory；
每次失败都进入训练；
每次专家成功都直接蒸馏；
每次 LLM 建议都直接改策略。
```

------

## 13.1 Memory Review

判断：

```text
这次推理是否产生值得长期保留的经验？
```

写入条件：

```text
同类失败重复出现；
主模型稳定失败；
专家修正稳定有效；
该 ROI 代表典型场景；
该策略可复用。
```

输出：

```json
{
  "write_memory": true,
  "memory_type": "failure_pattern",
  "scene_tags": [],
  "failure_tags": [],
  "summary": "...",
  "usefulness_score": 0.82
}
```

------

## 13.2 Skill Review

判断：

```text
这次经验是否值得形成或更新可复用 skill？
```

触发条件：

```text
同类 ROI 提取规则反复有效；
某专家在某 failure family 下连续修正成功；
某类问题应优先主模型计划修复而不是专家介入；
某旧 skill 连续导致退化。
```

------

## 13.3 Training Review

判断：

```text
ROI 是否进入 finetune_pool；
是否达到训练触发条件；
是否需要人工批准。
```

训练候选条件：

```text
错误类型明确；
有 GT、人工确认或专家高质量修正；
不是孤立轻微问题；
有 replay 样本；
样本质量通过审查；
失败模式具有训练价值。
```

------

## 13.4 Routing Review

判断：

```text
专家路由是否正确；
某专家在某类 ROI 上是否持续有效；
是否需要调整专家权重。
```

------

## 13.5 Distillation Review

判断：

```text
专家成功修正的 ROI 是否可反哺主模型。
```

进入蒸馏池条件：

```text
专家结果被 accept 或 partial_accept；
COCO 阶段专家结果更接近 GT；
真实 DOM-only 阶段通过几何审查；
无明显 ROI 外副作用；
样本质量达到 gold/silver 等级。
```

------

# 14. Training Loop 机制

训练闭环不是推理闭环起点，而是后台审查之后的深层动作。

训练闭环为：

```text
training candidates
  ↓
样本质量筛选
  ↓
训练包构建
  ↓
pilot 小规模训练
  ↓
pilot 审查
  ↓
正式训练
  ↓
COCO benchmark
  ↓
replay regression guard
  ↓
真实 DOM-only 几何审查
  ↓
模型晋级 / 拒绝 / 归档
  ↓
更新模型能力画像
  ↓
更新 memory / skill / routing policy
```

------

## 14.1 training_loop/ 目录职责

```text
training_loop/
  contracts.py
  sample_intake.py
  dataset_packager.py
  trigger_policy.py
  training_plan_builder.py
  pilot_trainer.py
  formal_trainer.py
  replay_guard.py
  post_train_evaluator.py
  model_promotion.py
  expert_to_main_distill.py
```

------

## 14.2 训练触发条件

```text
同类失败 ROI 达到最小数量；
样本质量通过率达标；
失败类型稳定重复；
当前模型存在明确短板；
有 replay good samples；
当前没有同类训练任务正在运行；
训练资源允许；
人工或规则审批通过。
```

建议初始阈值：

| 条件                   | 建议  |
| ---------------------- | ----- |
| 同类失败 ROI           | ≥ 30  |
| 高质量 COCO / 人工样本 | ≥ 100 |
| replay good samples    | ≥ 50  |
| 同类失败最近出现次数   | ≥ 3   |
| 样本质量通过率         | ≥ 80% |

------

## 14.3 模型晋级

新模型训练后必须经历：

```text
candidate
  ↓
shadow
  ↓
active
  ↓
specialized
```

如果退化：

```text
active → shadow → deprecated → retired
```

主模型晋级必须满足：

```text
COCO 指标提升；
四类错误不恶化；
几何异常不增加；
replay 不退化；
真实 DOM-only 稳定；
artifact 可复现。
```

------

# 15. 主模型与专家模型长期进化

最终系统的长期演化目标是：

```text
主模型越来越通用；
专家模型越来越差异化；
专家路由越来越可靠；
训练触发越来越克制；
ROI 经验越来越结构化；
智能体不是凭 LLM 变聪明，而是靠 trajectory 和客观评估积累变稳定。
```

专家能力反哺主模型：

```text
专家成功 ROI
  ↓
质量审查
  ↓
distillation candidate
  ↓
主模型训练样本
  ↓
main_model_vNext
  ↓
replay 验证
  ↓
主模型能力增强
```

------

# 16. CLI 设计

最终命令：

```bash
itd-agent infer --config configs/examples/itd_agent_dom_only_mainline_a.yaml
```

正式 DOM-only 推理。

```bash
itd-agent evolve-infer --config configs/examples/itd_agent_evolve_coco.yaml
```

COCO/公开数据集可监督推理进化。

```bash
itd-agent review pending
```

查看待审查 trajectory。

```bash
itd-agent review trajectory --id traj_xxx
```

查看单次 trajectory。

```bash
itd-agent train candidates
```

查看训练候选样本。

```bash
itd-agent train trigger --review-id xxx
```

审批训练触发。

```bash
itd-agent model promote --training-run xxx
```

审批模型晋级。

```bash
itd-agent state summary
```

查看 SQLite 状态库摘要。

------

# 17. 最终执行伪代码

```python
def run_main_expert_adaptive_loop(context):
    trajectory = InferenceTrajectory.start(context)

    main_plan = build_main_plan(context)
    trajectory.add("main_plan", main_plan)

    main_result = run_main_model(main_plan)
    trajectory.add("main_inference", main_result)

    main_eval = evaluate_main_result(
        main_result=main_result,
        gt=context.gt if context.has_gt else None,
    )
    trajectory.add("main_eval", main_eval)

    geometry_review = build_geometry_review(
        main_result=main_result,
        gt=context.gt if context.has_gt else None,
    )
    trajectory.add("geometry_review", geometry_review)

    main_decision = review_main_result(
        main_eval=main_eval,
        geometry_review=geometry_review,
        policy=context.adaptive_policy,
    )
    trajectory.add("main_decision", main_decision)

    if main_decision.action == "accept_main":
        final_result = freeze_main_result(main_result)

    elif main_decision.action == "retry_main_plan":
        repaired_plan = repair_main_plan(main_plan, main_eval, geometry_review)
        repaired_result = run_main_model(repaired_plan)
        repaired_eval = evaluate_main_result(repaired_result, context.gt)

        if repaired_eval.is_better_than(main_eval):
            final_result = freeze_main_result(repaired_result)
        else:
            final_result = freeze_main_result(main_result)

    elif main_decision.action == "escalate_expert":
        final_result = run_expert_stage(
            context=context,
            main_result=main_result,
            main_eval=main_eval,
            geometry_review=geometry_review,
            trajectory=trajectory,
        )

    else:
        final_result = freeze_failure_result(main_result, main_eval)

    trajectory.finalize(final_result)

    write_trajectory(trajectory)
    write_state_db(trajectory)
    register_artifacts(trajectory)
    enqueue_background_review(trajectory)

    return final_result
```

专家阶段：

```python
def run_expert_stage(context, main_result, main_eval, geometry_review, trajectory):
    roi_candidates = build_roi_candidates(
        main_result=main_result,
        main_eval=main_eval,
        geometry_review=geometry_review,
        gt=context.gt if context.has_gt else None,
    )

    roi_candidates = assign_roi_status(
        roi_candidates=roi_candidates,
        policy=context.roi_policy,
    )

    if is_global_failure(roi_candidates, context.roi_policy):
        repaired_plan = repair_main_plan(context.main_plan, main_eval, geometry_review)
        repaired_result = run_main_model(repaired_plan)
        return freeze_main_result(repaired_result)

    roi_clusters = cluster_rois_by_expert_tile(
        roi_candidates=roi_candidates,
        tile_size=1024,
    )

    expert_tasks = build_expert_tasks(
        roi_clusters=roi_clusters,
        routing_policy=context.expert_routing_policy,
    )

    expert_results = run_expert_tasks(expert_tasks)

    expert_reviews = review_expert_results(
        main_result=main_result,
        expert_results=expert_results,
        gt=context.gt if context.has_gt else None,
    )

    fused_result = fuse_accepted_expert_results(
        main_result=main_result,
        expert_reviews=expert_reviews,
        fusion_policy=context.fusion_policy,
    )

    return freeze_result(fused_result)
```

------

# 18. 最终一句话总结

**ITD_agent 的最终核心循环应被定义为：在 DOM-only 主线 A 下，以主模型作为稳定推理入口，以四类 ROI 错误为诊断核心，以专家模型作为局部纠错模块，以几何评估和 COCO GT 评估作为客观裁决依据，以 inference trajectory 作为自进化数据资产，并通过后台审查把高价值经验分别流向 memory、skill、finetune_pool、training_loop 和专家到主模型蒸馏，从而逐步实现主模型更通用、专家模型更差异化、路由策略更可靠、训练触发更克制的单木树冠检测与提取智能体闭环。**





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



------

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



