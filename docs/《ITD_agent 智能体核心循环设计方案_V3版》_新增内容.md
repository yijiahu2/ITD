下面这些内容可以**直接补充进第一版 V3 方案**。我按“应插入的位置 → 新增标题 → 可直接粘贴的正文”来写，尽量让你不用再重新组织。

------

# 一、建议在第一版第 7 节后新增：`7.14 model_capability_profile.py`

## 7.14 新增 `training_loop/model_capability_profile.py`

职责：

```text
为每一个 V3 训练产生的新模型版本生成能力画像，明确该模型擅长什么、不擅长什么、适合在什么场景下被调用，以及是否具备进入 shadow / routing_candidate 的证据基础。
```

V3 不能只记录一个 checkpoint，也不能只记录 COCO 指标。因为 ITD_agent 的目标不是单模型训练，而是主模型—专家模型协同进化，所以每个训练后的模型版本都必须有结构化能力画像。

能力画像应至少包含：

```text
1. 基础模型信息；
2. 训练目标；
3. 目标错误类型改善情况；
4. 非目标错误类型退化情况；
5. COCO / mask 指标变化；
6. ROI 内改善情况；
7. replay 样本退化情况；
8. DOM-only 几何异常变化；
9. 适用 expert_family；
10. 推荐使用方式；
11. 是否允许进入 shadow；
12. 是否可作为 routing_candidate 的证据。
```

输出文件建议为：

```text
outputs/evolve_runs/{run_id}/v3_training/model_registry/model_cards/{model_version_id}.json
outputs/evolve_runs/{run_id}/v3_training/model_registry/capability_profiles/{model_version_id}_capability_profile.json
```

能力画像结构建议：

```json
{
  "model_version_id": "mmdet_htc_dense_adhesion_v3_candidate_001",
  "model_id": "mmdet_htc",
  "model_role": "expert_model",
  "target_expert_family": "dense_adhesion",
  "target_failure_category": "under_segmentation",
  "training_job_id": "formal_xxx",
  "base_checkpoint": "xxx.pth",
  "new_checkpoint": "best_checkpoint.pth",

  "training_objective": {
    "primary_goal": "reduce_under_segmentation",
    "secondary_goals": [
      "improve_dense_crown_separation",
      "reduce_under_count_in_closed_canopy_roi"
    ]
  },

  "metric_delta_summary": {
    "target_error_improved": true,
    "target_error_delta": -0.12,
    "coco_ap_delta": 0.018,
    "coco_ap50_delta": 0.022,
    "precision_delta": -0.003,
    "recall_delta": 0.015,
    "geometry_anomaly_delta": 0.004,
    "replay_guard_passed": true
  },

  "strengths": [
    "dense_crown_adhesion",
    "under_segmentation",
    "closed_canopy_roi"
  ],

  "weaknesses": [
    "possible_precision_drop_on_sparse_background",
    "needs_more_replay_on_low_density_blocks"
  ],

  "recommended_usage": {
    "allowed_status": "shadow",
    "recommended_model_role": "expert_model",
    "recommended_failure_categories": [
      "under_segmentation"
    ],
    "recommended_expert_families": [
      "dense_adhesion"
    ],
    "not_recommended_for": [
      "false_positive_cleanup",
      "general_main_model_replacement"
    ]
  },

  "routing_evidence": {
    "can_generate_routing_candidate": true,
    "routing_scope": "under_segmentation_only",
    "requires_human_review": true
  }
}
```

注意：

```text
model_capability_profile 只负责记录模型能力，不直接修改 active route_map；
不直接替换主模型；
不直接替换专家模型；
不直接激活新的路由策略。
```

------

# 二、建议在第一版第 7 节后新增：`7.15 dom_only_geometry_guard.py`

## 7.15 新增 `training_loop/dom_only_geometry_guard.py`

职责：

```text
在没有人工 GT 或 COCO 标注的真实 DOM-only 场景下，对新模型输出结果进行几何稳定性回归检查，防止模型虽然在 COCO / ROI 测试集上提升，但在真实大范围 DOM 推理中产生明显几何异常。
```

V3 的训练后评估不能只依赖 COCO 指标。因为本项目的实际目标是高分辨率 DOM 影像上的单木树冠检测与提取，很多真实 DOM block 并没有人工标注。对于这类样本，需要通过客观几何指标做无标注回归保护。

该模块应复用现有：

```text
evaluation_analysis/geometry_metrics.py
evaluation_analysis/geometry_failure_tags.py
evaluation_analysis/roi_assessment.py
evaluation_analysis/finetune_effect_assessment.py
```

不要新建一套独立 evaluator。

DOM-only geometry guard 应检查：

```text
1. 异常小实例是否增加；
2. 异常大实例是否增加；
3. 重复检测是否增加；
4. mask 破碎程度是否增加；
5. 边界复杂度是否异常升高；
6. 局部实例密度是否异常；
7. 空洞 mask 是否增加；
8. 细长异常 mask 是否增加；
9. ROI 外误检是否增加；
10. 融合后实例数量是否出现异常波动。
```

建议输出：

```text
outputs/evolve_runs/{run_id}/v3_training/evaluation/dom_only_geometry_guard_report.json
```

报告结构建议：

```json
{
  "model_version_id": "mmdet_htc_dense_adhesion_v3_candidate_001",
  "evaluated_dom_samples": 50,
  "geometry_guard_passed": true,

  "baseline_geometry_summary": {
    "small_instance_ratio": 0.08,
    "large_instance_ratio": 0.04,
    "duplicate_instance_ratio": 0.03,
    "fragmented_mask_ratio": 0.06,
    "abnormal_density_block_ratio": 0.05
  },

  "candidate_geometry_summary": {
    "small_instance_ratio": 0.075,
    "large_instance_ratio": 0.041,
    "duplicate_instance_ratio": 0.028,
    "fragmented_mask_ratio": 0.055,
    "abnormal_density_block_ratio": 0.052
  },

  "geometry_delta": {
    "small_instance_ratio_delta": -0.005,
    "large_instance_ratio_delta": 0.001,
    "duplicate_instance_ratio_delta": -0.002,
    "fragmented_mask_ratio_delta": -0.005,
    "abnormal_density_block_ratio_delta": 0.002
  },

  "failed_guard_items": [],
  "warning_items": [
    "abnormal_density_block_ratio slightly increased, keep model in shadow"
  ],

  "decision_hint": "allow_shadow"
}
```

初始阈值建议：

```yaml
dom_only_geometry_guard:
  max_small_instance_ratio_increase: 0.03
  max_large_instance_ratio_increase: 0.02
  max_duplicate_instance_ratio_increase: 0.02
  max_fragmented_mask_ratio_increase: 0.03
  max_abnormal_density_block_ratio_increase: 0.03
  max_roi_outside_false_positive_increase: 0.02
```

判断规则：

```text
1. 如果目标错误改善，但几何异常明显增加，不允许 promote_to_shadow；
2. 如果几何异常轻微增加，但 COCO / replay 均通过，只能 keep_candidate；
3. 如果几何异常下降或稳定，且 replay guard 通过，才允许进入 shadow；
4. DOM-only geometry guard 不能替代 COCO benchmark，只能作为无标注真实场景补充保护。
```

------

# 三、建议在第一版第 7 节后新增：`7.16 routing_candidate_builder.py`

## 7.16 新增 `training_loop/routing_candidate_builder.py`

职责：

```text
根据训练后模型能力画像、错误类型改善结果、replay guard 结果和 DOM-only geometry guard 结果，生成新的专家路由候选建议。
```

V3 训练完成后，如果某个专家模型在特定错误类型上表现更好，系统应该记录这个证据，用于后续更新专家路由策略。但 V3.1 不应直接修改 active route_map，只能生成待审查的 routing_update_candidate。

输入：

```text
1. model_capability_profile.json；
2. promotion_decision.json；
3. replay_guard_report.json；
4. dom_only_geometry_guard_report.json；
5. error_type_delta.json；
6. expert_family 配置；
7. V2 routing_candidates。
```

输出：

```text
outputs/evolve_runs/{run_id}/v3_training/routing/routing_update_candidate.json
```

输出结构建议：

```json
{
  "candidate_id": "routing_candidate_underseg_htc_v3_001",
  "source_model_version_id": "mmdet_htc_dense_adhesion_v3_candidate_001",
  "source_training_job_id": "formal_xxx",

  "target_failure_category": "under_segmentation",
  "target_expert_family": "dense_adhesion",

  "recommended_route_update": {
    "failure_category": "under_segmentation",
    "primary_expert_candidate": "mmdet_htc_dense_adhesion_v3_shadow_001",
    "fallback_expert_candidate": "maskdino_official_current",
    "activation_scope": "shadow_only"
  },

  "evidence": {
    "target_error_improved": true,
    "target_error_delta": -0.12,
    "replay_guard_passed": true,
    "dom_only_geometry_guard_passed": true,
    "promotion_decision": "promote_to_shadow"
  },

  "risk_notes": [
    "not allowed to replace active route_map automatically",
    "requires human review before active routing update"
  ],

  "status": "pending_review"
}
```

约束：

```text
1. routing_candidate_builder 不修改 route_map；
2. 不覆盖 configs/expert_taxonomy/expert_families.yaml；
3. 不更新 active routing policy；
4. 不让 LLM 直接决定路由更新；
5. 只生成可审查、可回滚、可追溯的候选建议。
```

V3.1 中允许的路由结果只有：

```text
routing_update_candidate
```

不允许：

```text
active_route_map_update
```

------

# 四、建议在第一版第 7 节后新增：`7.17 training_feedback_writer.py`

## 7.17 新增 `training_loop/training_feedback_writer.py`

职责：

```text
将 V3 训练、评估、晋级和失败原因写成结构化反馈候选，供后续 memory / skill / review 系统使用。
```

V3 的训练结果不应只停留在 checkpoint 和 metrics 上。无论训练成功还是失败，都应该把“为什么成功 / 为什么失败 / 哪些样本有效 / 哪些样本有风险 / 哪些规则需要调整”沉淀为下一轮 V1/V2 可使用的经验资产。

但 V3.1 只生成候选，不自动激活 skill，不自动修改执行策略。

输入：

```text
1. training_plan.json；
2. sample_quality_report.json；
3. dataset_card.json；
4. training_metrics.json；
5. post_train_eval.json；
6. replay_guard_report.json；
7. dom_only_geometry_guard_report.json；
8. promotion_decision.json；
9. model_capability_profile.json。
```

输出：

```text
outputs/evolve_runs/{run_id}/v3_training/feedback/
  memory_feedback_candidate.json
  skill_feedback_candidate.json
  training_lesson_report.json
```

`memory_feedback_candidate.json` 示例：

```json
{
  "feedback_id": "memory_feedback_v3_001",
  "source_stage": "v3_training",
  "source_model_version_id": "mmdet_htc_dense_adhesion_v3_candidate_001",

  "memory_type": "training_lesson",
  "target_scope": "dense_adhesion_under_segmentation",

  "positive_lessons": [
    "dense closed-canopy under-segmentation samples improved after HTC fine-tuning",
    "hard replay samples prevented precision collapse"
  ],

  "negative_lessons": [
    "samples with invalid masks were frequently rejected by quality gate",
    "low-density sparse ROI should not be mixed into dense_adhesion expert training"
  ],

  "recommended_future_actions": [
    "collect more under-segmentation samples from dense canopy ROI",
    "increase replay coverage for sparse background blocks"
  ],

  "status": "pending_review"
}
```

`skill_feedback_candidate.json` 示例：

```json
{
  "feedback_id": "skill_feedback_v3_001",
  "source_stage": "v3_training",
  "source_model_version_id": "mmdet_htc_dense_adhesion_v3_candidate_001",

  "skill_candidate_type": "readonly_analysis_skill",
  "suggested_skill_name": "dense_adhesion_training_case_review",

  "suggested_usage": {
    "can_affect_report": true,
    "can_affect_training_sample_selection": false,
    "can_affect_routing_policy": false,
    "can_affect_fusion_policy": false
  },

  "content_summary": [
    "When under-segmentation is concentrated in dense canopy ROI, HTC expert fine-tuning is preferred over main model training.",
    "Replay samples must include sparse background and normal good cases to prevent false positive increase."
  ],

  "activation_requirement": "manual_review",
  "status": "pending_review"
}
```

约束：

```text
1. 不自动写入 active memory；
2. 不自动激活 hard skill；
3. 不自动改变 ROI 触发阈值；
4. 不自动改变专家路由；
5. 不自动改变融合策略；
6. 只生成 feedback candidate，交给后续审查。
```

------

# 五、建议修改第一版第 6 节 V3 输出结构

在原来的输出结构中增加以下目录：

```text
outputs/evolve_runs/{run_id}/v3_training/
  geometry_guard/
    dom_only_geometry_guard_report.json
    dom_only_geometry_failed_cases.jsonl

  model_registry/
    model_versions.jsonl
    model_cards/
      {model_version_id}.json
    capability_profiles/
      {model_version_id}_capability_profile.json

  routing/
    routing_update_candidate.json
    routing_candidate_report.json

  feedback/
    memory_feedback_candidate.json
    skill_feedback_candidate.json
    training_lesson_report.json
```

修改后的关键输出应包括：

```text
1. checkpoint；
2. training_job_summary；
3. post_train_eval；
4. replay_guard_report；
5. DOM-only geometry guard report；
6. model version record；
7. model card；
8. capability profile；
9. promotion decision；
10. expert-to-main distillation manifest；
11. routing update candidate；
12. memory / skill feedback candidate。
```

------

# 六、建议修改第一版第 10 节 V3 配置文件

在 `configs/examples/itd_agent_training_coco_v3.yaml` 中新增以下配置段。

## 1. 新增 geometry guard 配置

```yaml
dom_only_geometry_guard:
  enabled: true
  use_existing_geometry_metrics: true
  sample_source: replay_dom_only_samples
  max_eval_samples: 50

  checks:
    small_instance_ratio: true
    large_instance_ratio: true
    duplicate_instance_ratio: true
    fragmented_mask_ratio: true
    abnormal_density_block_ratio: true
    roi_outside_false_positive_ratio: true
    boundary_complexity_ratio: true

  thresholds:
    max_small_instance_ratio_increase: 0.03
    max_large_instance_ratio_increase: 0.02
    max_duplicate_instance_ratio_increase: 0.02
    max_fragmented_mask_ratio_increase: 0.03
    max_abnormal_density_block_ratio_increase: 0.03
    max_roi_outside_false_positive_increase: 0.02
    max_boundary_complexity_increase: 0.03
```

## 2. 新增 capability profile 配置

```yaml
capability_profile:
  enabled: true
  write_model_capability_profile: true
  include_coco_metrics: true
  include_error_type_delta: true
  include_geometry_delta: true
  include_replay_guard_summary: true
  include_recommended_usage: true
  allow_as_routing_evidence: true
```

## 3. 新增 routing candidate 配置

```yaml
routing_candidate:
  enabled: true
  build_routing_update_candidate: true
  allow_active_route_map_update: false
  require_promotion_to_shadow: true
  require_replay_guard_pass: true
  require_geometry_guard_pass: true
  status: pending_review
```

## 4. 新增 feedback 配置

```yaml
training_feedback:
  enabled: true
  write_memory_feedback_candidate: true
  write_skill_feedback_candidate: true
  allow_active_memory_update: false
  allow_active_skill_activation: false
  allow_policy_skill_activation: false
  status: pending_review
```

## 5. 修改 promotion 配置

原来第一版里已有：

```yaml
promotion:
  register_candidate: true
  allow_promote_to_shadow: true
  allow_promote_to_active: false
  require_replay_guard_pass: true
```

建议改为：

```yaml
promotion:
  register_candidate: true
  allow_promote_to_shadow: true
  allow_promote_to_active: false

  require_replay_guard_pass: true
  require_dom_only_geometry_guard_pass: true
  require_capability_profile: true

  allowed_decisions:
    - reject
    - keep_candidate
    - promote_to_shadow

  forbidden_decisions:
    - promote_to_active
    - replace_active_model
    - update_active_route_map
```

------

# 七、建议修改第一版第 7.12 `model_promotion.py`

原文已有 candidate → shadow / rejected / keep_candidate。建议补充以下内容。

## 7.12 补充：模型晋级必须同时依赖能力画像与几何回归保护

`promote_to_shadow` 不能只看 COCO 指标，也不能只看目标错误类型下降。必须同时满足：

```text
1. 目标错误类型有明确改善；
2. 总体 COCO / mask 指标不明显下降；
3. precision / recall 不出现严重单侧退化；
4. 非目标错误类型不明显恶化；
5. replay guard 通过；
6. DOM-only geometry guard 通过；
7. model_capability_profile 已生成；
8. checkpoint / config / dataset_bundle / evaluation report 可追溯。
```

晋级决策输入应包括：

```text
baseline_eval.json
candidate_eval.json
delta_eval.json
error_type_delta.json
geometry_delta.json
replay_guard_report.json
dom_only_geometry_guard_report.json
model_capability_profile.json
```

晋级决策输出应包括：

```json
{
  "model_version_id": "mmdet_htc_dense_adhesion_v3_candidate_001",
  "decision": "promote_to_shadow",
  "allow_active": false,
  "reasons": [
    "target under_segmentation error decreased",
    "replay guard passed",
    "DOM-only geometry guard passed",
    "no significant precision regression"
  ],
  "blocked_actions": [
    "promote_to_active",
    "replace_active_model",
    "update_active_route_map"
  ],
  "next_recommended_actions": [
    "run shadow comparison in next V1 evolve-infer",
    "collect more replay samples from sparse background blocks",
    "review routing_update_candidate manually"
  ]
}
```

如果 DOM-only geometry guard 不通过：

```text
即使 COCO 指标提升，也不能 promote_to_shadow。
```

如果 replay guard 通过但 geometry guard 轻微 warning：

```text
只能 keep_candidate，不能 shadow。
```

------

# 八、建议修改第一版第 8 节“专家模型进化路线”

原文中第一条最小链建议是：

```text
V2 approved false_negative / dense_adhesion 样本
→ maskdino_official
```

建议改得更严谨一些。

## 8.3 V3.1 第一条专家训练链应保持 failure_category 与 expert_family 一致

V3.1 不建议把 `false_negative` 和 `dense_adhesion` 强行绑定。因为：

```text
false_negative 主要对应漏检；
dense_adhesion 更偏闭冠粘连、树冠贴连、under-count、欠分割。
```

因此 V3.1 的第一条最小训练链建议二选一。

### 推荐优先方案 A：欠分割 / dense_adhesion / HTC

```text
V2 approved under_segmentation / dense_adhesion 样本
  ↓
finetune_pool.dataset_exporter
  ↓
sample_quality_gate
  ↓
training_bundle_materializer
  ↓
family_config_resolver 选择 mmdet_htc
  ↓
training_plan_builder 生成 train_mmdet_instance.py 配置
  ↓
trainer_runner 执行 pilot
  ↓
post_train_evaluator
  ↓
replay_guard
  ↓
dom_only_geometry_guard
  ↓
model_capability_profile
  ↓
model_registry 注册 mmdet_htc_dense_adhesion_v3_candidate_001
  ↓
model_promotion 判断 promote_to_shadow / keep_candidate / rejected
  ↓
routing_candidate_builder 生成 under_segmentation 路由候选
```

该方案优点：

```text
1. 与 dense_adhesion 专家族定义更一致；
2. 与 HTC 的结构优势更匹配；
3. ROI 错误目标更聚焦；
4. 训练后可以直接观察欠分割是否减少；
5. 不容易把漏检召回和粘连拆分混成一个训练目标。
```

### 备选方案 B：漏检 / recall_enhancement / MaskDINO

如果当前 expert taxonomy 中已有漏检召回类专家族，则可以使用：

```text
V2 approved false_negative / small_crown_recall 样本
  ↓
maskdino_official
```

适用目标：

```text
small_crown_miss
low_contrast_miss
edge_miss
dense_area_miss
```

不建议写成：

```text
false_negative / dense_adhesion
```

除非样本中明确标注的是：

```text
由于闭冠粘连导致主模型 under-count，并最终表现为漏检。
```

------

# 九、建议修改第一版第 12 节“V3 第一版最小落地路径”

将原来的最小路径替换为下面这一版。

## 12. V3.1 第一版最小落地路径

V3.1 不要一开始同时训练主模型和多个专家模型。第一条链建议选择一个专家模型、一个专家族、一类错误，跑通完整闭环。

推荐第一条链：

```text
V2 approved under_segmentation / dense_adhion 样本
  ↓
finetune_pool.dataset_exporter.export_finetune_dataset_bundle()
  ↓
sample_quality_gate
  ↓
training_bundle_materializer
  ↓
family_config_resolver 读取 expert_families.yaml
  ↓
training_plan_builder 生成 mmdet_htc 训练计划
  ↓
trainer_runner 执行 pilot training
  ↓
post_train_evaluator
  ↓
replay_guard
  ↓
dom_only_geometry_guard
  ↓
model_registry 注册 candidate model
  ↓
model_capability_profile 生成能力画像
  ↓
model_promotion 判断 reject / keep_candidate / promote_to_shadow
  ↓
expert_to_main_distill 生成专家反哺主模型 manifest
  ↓
routing_candidate_builder 生成 routing_update_candidate
  ↓
training_feedback_writer 生成 memory / skill feedback candidate
```

注意这里的 `dense_adhion` 应统一拼写为：

```text
dense_adhesion
```

V3.1 验收目标不是“训练出最强模型”，而是验证：

```text
1. V2 样本能否被正确读取；
2. 数据包能否正确构建；
3. 训练脚本能否被复用调用；
4. pilot 能否跑通；
5. formal 能否在 pilot 通过后执行；
6. 训练后评估能否复用 evaluation_analysis；
7. replay guard 是否生效；
8. DOM-only geometry guard 是否生效；
9. 新模型是否能被注册为 candidate；
10. 是否能生成能力画像；
11. 是否能只进入 shadow，不直接 active；
12. 是否能生成专家反哺主模型 manifest；
13. 是否能生成路由候选；
14. 是否能生成经验反馈候选。
```

------

# 十、建议修改第一版第 13 节文件级任务清单

## 13.2 新增文件部分补充

在第一版原有新增文件列表中加入：

```text
ITD_agent/training_loop/dom_only_geometry_guard.py
ITD_agent/training_loop/model_capability_profile.py
ITD_agent/training_loop/routing_candidate_builder.py
ITD_agent/training_loop/training_feedback_writer.py
```

更新后的新增文件列表建议为：

```text
ITD_agent/training_loop/v2_asset_loader.py
ITD_agent/training_loop/family_config_resolver.py
ITD_agent/training_loop/sample_quality_gate.py
ITD_agent/training_loop/training_bundle_materializer.py
ITD_agent/training_loop/training_plan_builder.py
ITD_agent/training_loop/trainer_runner.py
ITD_agent/training_loop/post_train_evaluator.py
ITD_agent/training_loop/replay_guard.py
ITD_agent/training_loop/dom_only_geometry_guard.py
ITD_agent/training_loop/model_registry.py
ITD_agent/training_loop/model_card_builder.py
ITD_agent/training_loop/model_capability_profile.py
ITD_agent/training_loop/model_promotion.py
ITD_agent/training_loop/expert_to_main_distill.py
ITD_agent/training_loop/routing_candidate_builder.py
ITD_agent/training_loop/training_feedback_writer.py
ITD_agent/training_loop/training_runner.py
```

## 13.5 暂不新增部分补充

第一版原本写了暂不新增：

```text
ITD_agent/cli/
新的 evaluator/
新的 trainer/
新的 expert registry/
新的 COCO exporter/
新的 active route_map updater/
active skill policy loader/
完整数据库化 model registry/
```

建议补充为：

```text
ITD_agent/cli/
state/migrations/003_v3_training_loop.sql
完整 SQLite 状态管理系统
新的 evaluator/
新的 trainer/
新的 trainer_adapters/
新的 expert registry/
新的 COCO exporter/
新的 active route_map updater/
active skill policy loader/
active memory writer/
active skill activator/
完整数据库化 model registry/
自动 promote_to_active
自动 rollback_manager
```

解释：

```text
这些内容不是方向错误，而是不适合 V3.1 首版落地。等 V3.1 的训练—评估—shadow 闭环稳定后，再进入 V3.2 或 V4 扩展。
```

------

# 十一、建议修改第一版第 14 节 V3 验收标准

在原有 15 条验收标准后增加：

```text
16. 是否能在训练后生成 model_capability_profile？
17. 是否能判断新模型具体擅长哪类 failure_category / expert_family？
18. 是否能执行 DOM-only geometry guard？
19. 是否能在无 GT 的真实 DOM 样本上检查几何异常是否增加？
20. 是否能在 geometry guard 不通过时阻止 promote_to_shadow？
21. 是否能生成 routing_update_candidate，但不直接修改 active route_map？
22. 是否能生成 memory_feedback_candidate，但不直接写入 active memory？
23. 是否能生成 skill_feedback_candidate，但不直接激活 policy skill？
24. 是否能保证 V3.1 训练结果最多进入 shadow，而不能 active？
25. 是否能完整追溯：V2 sample → dataset_bundle → training_job → checkpoint → evaluation → replay_guard → geometry_guard → model_version → capability_profile？
```

最终 V3.1 验收标准应能回答：

```text
这个模型为什么被训练？
用哪些样本训练？
样本是否通过质量门控？
是否存在 train / val / test 泄漏？
训练是否成功？
目标错误是否改善？
旧能力是否退化？
真实 DOM-only 几何是否稳定？
模型适合解决什么问题？
是否允许进入 shadow？
是否禁止 active？
是否生成专家反哺主模型 manifest？
是否生成路由候选？
是否生成经验反馈候选？
```

------

# 十二、建议修改第一版第 15 节最终判断

可以把第一版最后一句扩展为下面这版：

```text
V3 应该在总体设计方案的 Training Loop Layer 下，基于 V1 已跑通的主—专家推理轨迹和 V2 已审查通过的 finetune_pool / distillation_candidate / routing_candidate，复用当前仓库的 finetune_pool.dataset_exporter、configs/expert_taxonomy/expert_families.yaml、segmentation/model_training 训练入口和 evaluation_analysis 评估模块，完成“质量筛选 → 数据包实体化 → pilot 小规模训练 → formal 正式训练 → 自动评估 → replay guard → DOM-only geometry guard → candidate/shadow 模型注册 → 模型能力画像 → 专家反哺主模型 manifest → 路由候选 → 经验反馈候选”的受控权重更新闭环。
```

再补充一句：

```text
V3.1 的训练结果最多进入 shadow，不允许直接 active；V3.1 可以生成 routing_update_candidate、memory_feedback_candidate 和 skill_feedback_candidate，但不允许自动更新 active route_map、active memory 或 active policy skill。
```

------

# 十三、最终建议你加入第一版的新增模块清单

最终第一版应新增的核心模块是：

| 新增模块                        | 必要性   | 作用                                           |
| ------------------------------- | -------- | ---------------------------------------------- |
| `dom_only_geometry_guard.py`    | 必须     | 检查真实 DOM-only 无标注场景下几何异常是否增加 |
| `model_capability_profile.py`   | 必须     | 记录新模型擅长什么、不擅长什么、适合什么路由   |
| `routing_candidate_builder.py`  | 建议必须 | 生成路由候选，但不直接改 active route_map      |
| `training_feedback_writer.py`   | 建议必须 | 生成 memory / skill 反馈候选，但不直接激活     |
| `capability_profiles/` 输出目录 | 必须     | 存放能力画像                                   |
| `geometry_guard/` 输出目录      | 必须     | 存放 DOM-only 几何回归报告                     |
| `routing/` 输出目录             | 建议必须 | 存放路由候选                                   |
| `feedback/` 输出目录            | 建议必须 | 存放经验反馈候选                               |

最终 V3.1 的边界可以概括为：

```text
训练可以发生；
权重可以更新；
模型可以注册；
模型可以进入 shadow；
能力画像必须生成；
几何回归必须检查；
专家反哺主模型 manifest 必须生成；
路由和 skill 可以生成候选；
但不能自动 active，不能自动改路由，不能自动激活 policy skill。
```