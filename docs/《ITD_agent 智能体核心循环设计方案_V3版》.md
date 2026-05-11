# 第一版V3方案

# 1. V3 版本方案

V3 正式名称建议为：

```text
V3：受控权重更新、模型晋级与专家反哺主模型闭环
Controlled Weight Update, Model Promotion and Expert-to-Main Feedback Loop
```

一句话定义：

```text
V3 基于 V1 真实主—专家推理闭环产生的 trajectory / ROI / expert_review / fusion_event，
以及 V2 审查通过的 finetune_pool / routing_candidate / distillation_candidate / skill draft / memory，
调用项目已有专家模型和主模型官方微调配置，
完成训练触发、训练数据包实体化、pilot 小规模训练、formal 正式训练、训练后自动评估、replay 回归保护、模型版本注册和 shadow 晋级 / 拒绝。
```

总体设计中已经明确，最终系统不是单纯训练一个分割模型，也不是堆叠多个 SOTA，而是主模型负责稳定、通用、大范围推理，专家模型负责针对主模型短板进行局部纠错，训练闭环负责把高质量失败样本和专家成功样本反哺模型权重。 V2 文档也明确：V2 只沉淀 memory、skill draft、finetune_pool、routing_candidate、distillation_candidate，不训练、不改权重、不晋级，这些正是 V3 的输入燃料。

Hermes 文档中真正值得借鉴的是“数据合成、质量筛选、小规模实验、正式训练、自动评估”的工程闭环，而不是 RL 本身；文档也明确把 Skill 沉淀和训练闭环区分为外部经验沉淀与权重内化两条路径。 对 ITD_agent 来说，V3 的权重更新不做 GRPO/RL，而是调用 MaskDINO、MMDetection 系列、Mask2Former、HTC、Cascade Mask R-CNN、Mask Scoring R-CNN 等已有官方/项目训练入口并适配本地计算机和项目真实情况完成监督微调。

------

# 2. V3 与 V1/V2 的关系

最终三阶段关系必须固定：

| 阶段 | 定位                   | 产物                                                         | V3 是否重复 |
| ---- | ---------------------- | ------------------------------------------------------------ | ----------- |
| V1   | 可监督主—专家推理闭环  | trajectory、ROI、expert_task、expert_review、fusion_event、pending candidates | 不重复      |
| V2   | 轨迹审查与经验沉淀     | memory、skill draft、finetune_pool、routing_candidate、distillation_candidate | 不重复      |
| V3   | 受控权重更新与模型晋级 | training_job、checkpoint、model_version、promotion_report、distillation_manifest | 新增        |

所以 V3 的输入必须是：

```text
V2 审查后的资产，而不是 V1 原始 pending candidates。
```

V3 的输出必须是：

```text
可追溯的新模型版本，而不是直接覆盖主模型或专家模型。
```

------

# 3. V3 必须复用的项目现状

## 3.1 复用 `finetune_pool`

当前仓库里 `ITD_agent/finetune_pool` 已经有 `dataset_exporter.py`、`policy.py`、`query.py`、`recommendation.py`、`store.py`、`contracts.py` 等文件。其中 `dataset_exporter.py` 已经提供 `export_finetune_dataset_bundle()`，会读取 finetune pool snapshot、recent samples、public candidates，并按 `target_model_role`、`target_expert_family`、`failure_category` 过滤样本；它已经区分 `training_ready_samples`、`weak_supervision_candidates`、`label_preparation_queue`、`replay_samples`、`public_dataset_candidates`。

因此 V3 不应该重写一个从零开始的 dataset exporter，而应该：

```text
调用 finetune_pool.dataset_exporter.export_finetune_dataset_bundle()
→ 再做 V3 sample quality gate
→ 再实体化为官方训练脚本需要的 train/val/test/replay 数据包。
```

## 3.2 复用 `configs/expert_taxonomy/expert_families.yaml`

当前 `expert_families.yaml` 已经定义了专家族、优先算法、训练模板、训练默认参数。例如 `dense_adhesion` 对应闭冠、树冠粘连、under-count、dense mixed stands，优先算法包括 `mmdet_htc` 和 `maskdino_official`，并已有 `template_candidates`、`epochs`、`batch_size`、`lr`、`replay_ratio`、`hard_case_ratio` 等训练默认配置。

因此 V3 不应该另造一个新的 `expert_training_registry.yaml`。最合理做法是新增一个轻量解析器：

```text
training_loop/family_config_resolver.py
```

它只读取现有 `configs/expert_taxonomy/expert_families.yaml`，解析：

```text
target_expert_family
algorithms_priority
template_candidates
training_defaults
replay_ratio
hard_case_ratio
prior_axes
```

## 3.3 复用 `segmentation/model_training`

当前 `ITD_agent/segmentation/model_training` 已经包含：

```text
expert_injection.py
infer_segmentation_finetuned.py
maskdino_train_entry.py
prepare_public_coco_segmentation_dataset.py
test_maskdino_instance.py
test_mmdet_instance.py
train_maskdino_instance.py
train_mmdet_instance.py
train_segmentation_template.py
mmdet_custom/
```

这些就是 V3 权重更新的底层入口。所以 V3 不应该新写 MaskDINO trainer 或 MMDetection trainer，而应该用 training plan 调用这些已有入口。

## 3.4 复用 `evaluation_analysis`

当前 `ITD_agent/evaluation_analysis` 已经包含 `benchmark_engine.py`、`coco_error_decomposition.py`、`expert_result_comparator.py`、`finetune_effect_assessment.py`、`geometry_failure_tags.py`、`geometry_metrics.py`、`main_model_assessment.py`、`roi_assessment.py` 等文件。因此 V3 的训练后评估不能重写指标计算，而应该调用现有 `evaluation_analysis` 完成：

```text
COCO 指标
四类错误变化
ROI 内专家改善
几何异常变化
微调前后效果对比
benchmark 评估
```

------

# 4. V3 不做什么

为了避免重复造轮子和堆屎山，V3 明确不做：

```text
1. 不新建一套 COCO evaluator；
2. 不重写 MaskDINO / MMDetection / Mask2Former / HTC / Cascade Mask R-CNN 训练逻辑；
3. 不新建平行专家模型注册表，优先复用 expert_families.yaml；
4. 不把运行期 patch 后的配置写回 configs/；
5. 不从 V1 raw pending candidates 直接训练；
6. 不重新做 V2 review；
7. 不做 RL / GRPO；
8. 不自动 promote_to_active；
9. 不自动更新 active route_map；
10. 不自动激活 hard skill policy；
11. 不让 LLM 直接决定训练、晋级或路由更新。
```

------

# 5. V3 总体运行流程

最终流程如下：

```text
V2 review output
  ↓
v2_asset_loader.py
  ↓
finetune_pool.dataset_exporter.export_finetune_dataset_bundle()
  ↓
family_config_resolver.py 读取 expert_families.yaml
  ↓
sample_quality_gate.py
  ↓
training_bundle_materializer.py 生成 train / val / test / replay
  ↓
training_plan_builder.py 生成已有训练入口可执行配置
  ↓
trainer_runner.py 调用 train_maskdino_instance.py 或 train_mmdet_instance.py
  ↓
post_train_evaluator.py 调用 evaluation_analysis
  ↓
replay_guard.py 判断是否退化
  ↓
model_registry.py 注册 candidate / shadow / rejected
  ↓
model_promotion.py 给出 promote_to_shadow / reject / keep_candidate
  ↓
expert_to_main_distill.py 生成专家反哺主模型 manifest
```

这条线严格遵循：

```text
已有样本池 → 已有专家 taxonomy → 已有训练入口 → 已有评估模块 → 新增受控训练编排。
```

------

# 6. V3 输出结构

建议输出到：

```text
outputs/evolve_runs/{run_id}/v3_training/
```

目录如下：

```text
outputs/evolve_runs/{run_id}/v3_training/
  config/
    training_config.yaml
    normalized_training_config.yaml

  trigger/
    trigger_context.json
    trigger_decision.json
    trigger_report.json

  finetune_bundle/
    finetune_dataset_bundle.json
    sample_quality_report.json
    rejected_samples.csv

  dataset_bundle/
    dataset_card.json
    train/
    val/
    test/
    replay/
    annotations/
      instances_train.json
      instances_val.json
      instances_test.json
    manifest_train.csv
    manifest_val.csv
    manifest_test.csv
    manifest_replay.csv

  training_jobs/
    pilot_{job_id}/
      training_plan.json
      generated_config.yaml 或 generated_config.py
      command.sh
      stdout.log
      stderr.log
      checkpoints/
      best_checkpoint.*
      training_metrics.json

    formal_{job_id}/
      training_plan.json
      generated_config.yaml 或 generated_config.py
      command.sh
      stdout.log
      stderr.log
      checkpoints/
      best_checkpoint.*
      training_metrics.json

  evaluation/
    baseline_eval.json
    candidate_eval.json
    delta_eval.json
    error_type_delta.json
    geometry_delta.json
    finetune_effect_assessment.json

  replay_guard/
    replay_guard_report.json
    replay_failed_cases.jsonl

  model_registry/
    model_versions.jsonl
    model_cards/
      {model_version_id}.json

  promotion/
    promotion_decision.json
    promotion_report.json

  distillation/
    main_model_distillation_manifest.csv
    main_model_distillation_manifest.json
    distillation_report.json

  reports/
    v3_training_summary.json
    v3_training_summary.csv
```

------

# 7. V3 模块设计

## 7.1 修改现有 `training_loop/contracts.py`

保留已有 V1 dry-run `TrainingCandidate`，不要破坏 V1/V2 兼容。新增 V3 正式数据结构：

```python
@dataclass(frozen=True)
class TrainingTriggerContext:
    source_run_id: str
    source_v2_review_dir: str

    target_model_role: str          # main_model | expert_model
    target_model_id: str            # maskdino_official | mmdet_htc | ...
    target_expert_family: str | None
    failure_category: str | None

    training_ready_sample_count: int
    weak_supervision_candidate_count: int
    replay_sample_count: int
    public_dataset_candidate_count: int

    dataset_bundle_path: str | None
    evidence: dict[str, Any]
@dataclass(frozen=True)
class TrainingPlan:
    training_job_id: str
    training_mode: str              # pilot | formal
    target_model_role: str
    target_model_id: str
    algorithm_name: str
    target_expert_family: str | None
    failure_category: str | None

    source_config_path: str
    generated_config_path: str
    output_dir: str
    command: list[str]
    expected_checkpoint_glob: str
    metadata: dict[str, Any]
@dataclass(frozen=True)
class ModelVersionRecord:
    model_version_id: str
    model_id: str
    model_role: str
    algorithm_name: str
    checkpoint_path: str
    source_training_job_id: str
    status: str                     # candidate | shadow | rejected
    metrics_summary: dict[str, Any]
    replay_guard_summary: dict[str, Any]
```

------

## 7.2 修改现有 `training_loop/trigger_policy.py`

保留已有 dry-run 触发函数，新增：

```python
def evaluate_v3_training_trigger(
    context: TrainingTriggerContext,
    *,
    min_training_ready: int = 100,
    min_replay: int = 30,
    min_public_candidates: int = 0,
    allow_weak_supervision: bool = True,
) -> dict[str, object]:
    ...
```

初始决策规则：

| 条件                                               | 决策              |
| -------------------------------------------------- | ----------------- |
| `training_ready_sample_count < min_training_ready` | reject 或 defer   |
| `replay_sample_count < min_replay`                 | defer             |
| 找不到 `target_model_id` 对应训练入口              | reject            |
| 找不到 `target_expert_family` 对应配置             | reject            |
| 样本来源过度集中                                   | need_human_review |
| 条件通过                                           | approve_pilot     |

V3 初版只允许：

```text
approve_pilot
```

不允许直接：

```text
approve_formal
```

------

## 7.3 新增 `training_loop/v2_asset_loader.py`

职责：

```text
读取 V2 审查后的资产，不重新扫描 V1 原始 trajectory。
```

读取：

```text
v2_review/reports/review_summary.json
v2_review/reports/asset_summary.json
v2_review/finetune_pool/manifest.json 或 manifest.csv
v2_review/distillation/distillation_candidates.jsonl
v2_review/routing/routing_candidates.jsonl
```

输出：

```text
TrainingTriggerContext 的基础输入
```

如果你当前 V2 落地后的实际路径与方案略有差异，这个模块负责做兼容，而不是让后续训练模块到处写路径判断。

------

## 7.4 新增 `training_loop/family_config_resolver.py`

职责：

```text
读取 configs/expert_taxonomy/expert_families.yaml，
根据 target_expert_family / failure_category / target_model_id 解析训练模板和默认参数。
```

输出：

```text
algorithm_name
source_config_path
training_defaults
replay_ratio
hard_case_ratio
```

注意：

```text
不新建专家训练配置体系；
不覆盖 expert_families.yaml；
只解析、只引用。
```

------

## 7.5 新增 `training_loop/sample_quality_gate.py`

职责：

```text
对 finetune_pool.dataset_exporter 的输出做训练前最后质量门控。
```

检查项：

```text
1. sample 是否来自 V2 approved；
2. ready_for_training 是否为 true；
3. label_status 是否为 manual / pseudo；
4. image / mask / annotation artifact 是否存在；
5. bbox / segmentation 是否有效；
6. target_model_role 是否匹配；
7. target_expert_family 是否匹配；
8. failure_category 是否匹配；
9. replay_good_sample 是否进入 replay，不进入 train；
10. weak supervision 是否满足配置允许。
```

输出：

```text
accepted_samples
rejected_samples
deferred_samples
sample_quality_report.json
```

------

## 7.6 新增 `training_loop/training_bundle_materializer.py`

职责：

```text
把 finetune_dataset_bundle.json 实体化为官方训练脚本可读取的数据集。
```

输入：

```text
finetune_dataset_bundle.json
accepted_samples
expert_families.yaml 中的 replay_ratio / hard_case_ratio
```

输出：

```text
dataset_bundle/
  train/
  val/
  test/
  replay/
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
  dataset_card.json
```

关键规则：

```text
1. 同一 original_image_id 不跨 train / val / test；
2. 同一 source_trajectory_id 不跨 train / val / test；
3. replay 不参与训练；
4. hard case 比例受 expert_families.yaml 的 hard_case_ratio 控制；
5. replay 比例受 expert_families.yaml 的 replay_ratio 控制；
6. 生成 dataset_card，记录来源、切分、错误类型、专家族、标签质量。
```

------

## 7.7 新增 `training_loop/training_plan_builder.py`

职责：

```text
把 TrainingTriggerContext + family_config_resolver 输出 + dataset_bundle
翻译成 train_maskdino_instance.py 或 train_mmdet_instance.py 能执行的计划。
```

输出：

```text
training_plan.json
generated_config.yaml / generated_config.py
command.sh
```

规则：

```text
1. 不重写模型结构；
2. 不重写官方 optimizer；
3. 不重写官方 scheduler；
4. 只 patch dataset path、annotation path、work_dir、load_from、num_classes、max_epochs / iterations、batch_size、lr 等必要项；
5. patch 后配置写入 outputs/.../v3_training/training_jobs/{job_id}/，不写回 configs/。
```

------

## 7.8 新增 `training_loop/trainer_runner.py`

职责：

```text
执行 training_plan_builder 生成的 command.sh。
```

它只做：

```text
1. 执行命令；
2. 保存 stdout.log / stderr.log；
3. 记录 return code；
4. 定位 best checkpoint；
5. 写 training_job_summary.json。
```

不做：

```text
不理解 MaskDINO 内部；
不理解 MMDetection 内部；
不重写训练逻辑。
```

------

## 7.9 新增 `training_loop/post_train_evaluator.py`

职责：

```text
调用现有 evaluation_analysis 做训练后评估。
```

输出：

```text
baseline_eval.json
candidate_eval.json
delta_eval.json
error_type_delta.json
geometry_delta.json
finetune_effect_assessment.json
```

应优先复用：

```text
evaluation_analysis/benchmark_engine.py
evaluation_analysis/coco_error_decomposition.py
evaluation_analysis/geometry_metrics.py
evaluation_analysis/geometry_failure_tags.py
evaluation_analysis/expert_result_comparator.py
evaluation_analysis/finetune_effect_assessment.py
```

------

## 7.10 新增 `training_loop/replay_guard.py`

职责：

```text
防止新模型为了修复某一类错误而破坏已有能力。
```

初始规则：

```yaml
replay_guard:
  max_ap_drop: 0.01
  max_ap50_drop: 0.015
  max_recall_drop: 0.02
  max_precision_drop: 0.02
  max_error_type_regression_ratio: 0.05
  max_catastrophic_cases: 0
```

不通过 replay guard：

```text
不能 promote_to_shadow；
只能 keep_candidate 或 rejected。
```

------

## 7.11 新增 `training_loop/model_registry.py`

这里要注意：当前仓库已有 `segmentation/model_registry/` 语义更偏模型/adapter 注册，不等于训练后版本管理。因此 V3 可以在 `training_loop/model_registry.py` 中只做轻量模型版本记录。

初版写 JSONL 即可：

```text
outputs/.../v3_training/model_registry/model_versions.jsonl
```

记录：

```text
model_version_id
model_id
model_role
algorithm_name
checkpoint_path
training_job_id
status
metrics_summary
replay_guard_summary
model_card_path
```

------

## 7.12 新增 `training_loop/model_promotion.py`

职责：

```text
决定 candidate model 是否可以进入 shadow。
```

初版只允许：

```text
candidate → shadow
candidate → rejected
candidate → keep_candidate
```

不允许自动：

```text
shadow → active
```

`promote_to_shadow` 条件：

```text
1. 目标错误类型改善；
2. 总体 COCO 指标不明显下降；
3. Precision / Recall 不明显单侧退化；
4. 其他三类错误不明显恶化；
5. 几何异常不增加；
6. replay guard 通过；
7. checkpoint / config / dataset_bundle 可追溯。
```

------

## 7.13 新增 `training_loop/expert_to_main_distill.py`

V3.1 只做 manifest，不直接训练主模型。

输入：

```text
V2 distillation_candidates
专家 accept / partial_accept 样本
专家预测结果
GT 或 pseudo-label
```

规则：

```text
1. 有 GT 时优先使用 GT；
2. 无 GT 时只允许 gold / silver pseudo-label；
3. expert_reject 不能作为正样本；
4. 必须记录 source_expert_model、expert_review_decision、fusion_result、quality_tier；
5. 输出 main_model_distillation_manifest.csv/json。
```

主模型真正训练放到 V3.2，不建议 V3.1 立刻做。

------

# 8. 专家模型进化路线

V3 必须服务你已经确定的主模型/专家模型进化逻辑：**主模型越来越通用，专家模型越来越差异化**。

## 8.1 专家模型优先

V3 第一阶段建议先训练专家模型，而不是直接训练主模型。

原因：

```text
1. V1/V2 的 ROI 样本天然是局部错误样本；
2. 专家模型本来就是针对主模型短板；
3. 专项训练目标更清晰；
4. 训练后可通过 ROI 内改善直接评估；
5. 不容易破坏主模型全局稳定性。
```

## 8.2 四类错误与专家模型对应

结合总体方案、前面讨论和当前 `expert_families.yaml`，建议这样落地：

| 四类错误                  | 首选训练目标                                     | 当前项目可复用入口                                      |
| ------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| 漏检 false_negative       | `maskdino_official`                              | `train_maskdino_instance.py`、`maskdino_train_entry.py` |
| 欠分割 under_segmentation | `mmdet_htc`                                      | `train_mmdet_instance.py`，对应 dense_adhesion          |
| 误检 false_positive       | `mmdet_cascade_mask_rcnn`                        | `train_mmdet_instance.py`                               |
| 过分割 over_segmentation  | `mmdet_mask2former` 或 `mmdet_cascade_mask_rcnn` | `train_mmdet_instance.py`                               |
| 边界 / mask 质量校准      | `mmdet_mask_scoring_rcnn`                        | 后续作为 boundary_calibration 专家                      |

`dense_adhesion` 当前已经明确对应闭冠、树冠粘连、under-count、dense mixed stands，优先算法为 `mmdet_htc` 和 `maskdino_official`，因此它非常适合作为 V3 第一批专家微调对象之一。([GitHub](https://github.com/yijiahu2/ITD/blob/main/configs/expert_taxonomy/expert_families.yaml))

------

# 9. 主模型进化路线

主模型不建议 V3.1 直接训练。主模型训练应晚于专家模型专项训练，原因是主模型目标是：

```text
稳定、通用、大范围推理。
```

主模型训练样本必须来自多类型平衡数据：

```text
false_negative
false_positive
under_segmentation
over_segmentation
boundary_refine
hard_negative
normal_good_case
expert_success
replay_good_sample
```

主模型训练触发条件应比专家更严格：

```text
1. 至少覆盖四类错误中的 3 类；
2. gold / silver 样本比例达标；
3. replay 样本充足；
4. 单一错误类型占比不能过高；
5. 单一图像 / 单一 trajectory 占比不能过高；
6. 专家成功样本必须通过 V2 distillation review；
7. 新主模型只能 candidate → shadow，不允许直接 active。
```

专家反哺主模型路线：

```text
专家成功 ROI
  ↓
V2 distillation review
  ↓
V3 distillation manifest
  ↓
主模型训练包
  ↓
main_model_vNext candidate
  ↓
replay guard
  ↓
shadow
```

------

# 10. V3 配置文件

新增示例配置：

```text
configs/examples/itd_agent_training_coco_v3.yaml
```

内容建议：

```yaml
version: v3
mode: controlled_training_v3
mainline_profile: A_DOM_ONLY

source:
  run_id: coco_main_expert_loop_test
  v2_review_dir: outputs/evolve_runs/coco_main_expert_loop_test/v2_review
  finetune_pool_root: outputs/finetune_pool

target:
  target_model_role: expert_model
  target_model_id: maskdino_official
  target_expert_family: dense_adhesion
  failure_category: false_negative

expert_taxonomy:
  path: configs/expert_taxonomy/expert_families.yaml
  use_training_defaults: true

dataset:
  use_finetune_pool_exporter: true
  split_by:
    - original_image_id
    - source_trajectory_id
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  build_replay: true

quality_gate:
  min_training_ready_samples: 100
  min_replay_samples: 30
  allow_manual_labels: true
  allow_pseudo_labels: true
  reject_missing_artifacts: true
  reject_invalid_masks: true
  reject_empty_annotations: true

training:
  pilot_first: true

  pilot:
    enabled: true
    max_samples: 200
    override_epochs: 3

  formal:
    enabled: true
    require_pilot_pass: true

official_config_patch:
  patch_dataset_paths: true
  patch_num_classes: true
  patch_work_dir: true
  patch_load_from: true
  patch_seed: true
  keep_model_architecture: true
  keep_official_optimizer: true
  keep_official_scheduler: true

runner:
  output_dir: outputs/evolve_runs/coco_main_expert_loop_test/v3_training
  write_command_sh: true
  execute_training: true

evaluation:
  use_existing_evaluation_analysis: true
  run_coco_eval: true
  run_error_decomposition: true
  run_geometry_eval: true
  run_finetune_effect_assessment: true
  run_replay_guard: true

replay_guard:
  max_ap_drop: 0.01
  max_ap50_drop: 0.015
  max_recall_drop: 0.02
  max_precision_drop: 0.02
  max_error_type_regression_ratio: 0.05
  max_catastrophic_cases: 0

promotion:
  register_candidate: true
  allow_promote_to_shadow: true
  allow_promote_to_active: false
  require_replay_guard_pass: true

distillation:
  enabled: true
  build_distillation_manifest: true
  run_distillation_training: false
  use_gt_when_available: true
  pseudo_label_quality_min: silver

guardrails:
  allow_weight_update: true
  allow_active_model_replace: false
  allow_active_routing_policy_update: false
  allow_active_skill_policy: false
  allow_llm_direct_training_decision: false
  allow_llm_direct_model_promotion: false
```

运行期生成的配置必须写入：

```text
outputs/evolve_runs/{run_id}/v3_training/training_jobs/{job_id}/generated_config.*
```

不要写回 `configs/`。

------

# 11. V3 运行入口

当前项目不一定需要马上新增完整 `ITD_agent/cli/` 体系。更稳的是新增模块入口和薄脚本。

新增：

```text
ITD_agent/training_loop/training_runner.py
scripts/run_training_loop_v3.py
```

推荐命令：

```bash
python -m ITD_agent.training_loop.training_runner \
  --config configs/examples/itd_agent_training_coco_v3.yaml
```

或薄脚本：

```bash
python scripts/run_training_loop_v3.py \
  --config configs/examples/itd_agent_training_coco_v3.yaml
```

后续稳定后再接入统一 `itd-agent train run` 命令。

------

# 12. V3 第一版最小落地路径

不要一开始四个专家和主模型一起训练。V3.1 只跑通一条最小闭环。

建议第一条链：

```text
V2 approved false_negative / dense_adhesion 样本
  ↓
finetune_pool.dataset_exporter
  ↓
sample_quality_gate
  ↓
training_bundle_materializer
  ↓
family_config_resolver 选择 maskdino_official
  ↓
training_plan_builder 生成 train_maskdino_instance.py 配置
  ↓
trainer_runner 执行 pilot
  ↓
post_train_evaluator
  ↓
replay_guard
  ↓
model_registry 注册 maskdino_official_v3_candidate_001
  ↓
model_promotion 判断 promote_to_shadow / rejected / keep_candidate
```

跑通后扩展：

```text
mmdet_htc → dense_adhesion / under_segmentation
mmdet_cascade_mask_rcnn → false_positive / large_crown_over_split
mmdet_mask2former → over_segmentation / cross_domain_generalist
mmdet_mask_scoring_rcnn → boundary_calibration
```

最后再做：

```text
expert_to_main_distillation
main_model_vNext
routing_policy_candidate
```

------

# 13. 文件级任务清单

## 13.1 修改现有文件

```text
ITD_agent/training_loop/contracts.py
```

保留 V1 `TrainingCandidate`，新增 V3 `TrainingTriggerContext`、`TrainingPlan`、`TrainingRunResult`、`ModelVersionRecord`。

```text
ITD_agent/training_loop/trigger_policy.py
```

保留 V1 dry-run trigger，新增 `evaluate_v3_training_trigger()`。

```text
ITD_agent/training_loop/sample_intake.py
```

保留 V1 dry-run intake，不扩成巨型模块。V3 资产读取放到新文件 `v2_asset_loader.py`。

------

## 13.2 新增文件

```text
ITD_agent/training_loop/v2_asset_loader.py
ITD_agent/training_loop/family_config_resolver.py
ITD_agent/training_loop/sample_quality_gate.py
ITD_agent/training_loop/training_bundle_materializer.py
ITD_agent/training_loop/training_plan_builder.py
ITD_agent/training_loop/trainer_runner.py
ITD_agent/training_loop/post_train_evaluator.py
ITD_agent/training_loop/replay_guard.py
ITD_agent/training_loop/model_registry.py
ITD_agent/training_loop/model_card_builder.py
ITD_agent/training_loop/model_promotion.py
ITD_agent/training_loop/expert_to_main_distill.py
ITD_agent/training_loop/training_runner.py
```

------

## 13.3 新增薄脚本

```text
scripts/run_training_loop_v3.py
```

只做参数转发，不写核心逻辑。

------

## 13.4 新增配置

```text
configs/examples/itd_agent_training_coco_v3.yaml
```

------

## 13.5 暂不新增

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

这些都容易导致架构变重、重复造轮子。

------

# 14. V3 验收标准

V3.1 完成后，必须能回答：

```text
1. 是否能读取 V2 审查后的 finetune_pool，而不是 V1 raw pending candidates？
2. 是否调用了 finetune_pool.dataset_exporter.export_finetune_dataset_bundle()？
3. 是否复用了 configs/expert_taxonomy/expert_families.yaml？
4. 是否能根据 target_expert_family 选择已有 template_candidate？
5. 是否能生成 train / val / test / replay 数据包？
6. 是否能按 original_image_id / source_trajectory_id 防止数据泄漏？
7. 是否能调用 train_maskdino_instance.py 或 train_mmdet_instance.py？
8. 是否能完成 pilot training？
9. 是否能找到 best checkpoint？
10. 是否能调用 evaluation_analysis 做训练后评估？
11. 是否能生成 replay_guard_report？
12. 是否能注册 candidate model？
13. 是否不会自动替换 active 模型？
14. 是否不会自动改 active route_map？
15. 是否能生成 expert_to_main_distill manifest？
```

------

# 15. 最终判断

最终版 V3 的核心不是“训练脚本”，而是：

```text
把当前 ITD 仓库已经存在的训练资源纳入主—专家自进化闭环。
```

最终一句话：

```text
V3 应该在总体设计方案的 Training Loop Layer 下，基于 V1 已跑通的主—专家推理轨迹和 V2 已审查通过的 finetune_pool / distillation_candidate / routing_candidate，复用当前仓库的 finetune_pool.dataset_exporter、configs/expert_taxonomy/expert_families.yaml、segmentation/model_training 训练入口和 evaluation_analysis 评估模块，完成“质量筛选 → pilot 小规模训练 → formal 正式训练 → 自动评估 → replay guard → candidate/shadow 模型注册 → 专家反哺主模型 manifest”的受控权重更新闭环。
```

也就是说：

```text
官方/已有训练入口负责真正更新权重；
V3 负责决定该不该训练、用什么样本训练、训练哪个模型、训练后是否可信、是否允许进入 shadow。
```





# 第二版V3方案

大哥，结论先说清楚：

**V3 可以正式进入“训练触发、模型权重更新、模型晋级、专家反哺主模型、策略回流”阶段。**
但 V3 不是推翻 V1/V2，也不是重写系统，而是把 V1/V2 形成的资产真正接入训练闭环，让总体方案最后一段“训练触发 / 模型晋级 / 专家能力反哺主模型”落地。总体方案中已经明确：主模型负责稳定通用推理，专家模型负责局部纠错，ROI/Trajectory 记录失败证据，后台审查决定哪些经验进入 memory、skill、finetune_pool 和 training_loop，训练闭环再把高质量失败样本和专家成功样本反哺模型权重。

所以 V3 的一句话定义是：

```text
V3 = 基于 V2 审查后的高质量 memory / skill / finetune_pool / distillation_candidate，
构建受控训练闭环，实现主模型与专家模型的可验证权重更新、模型晋级、专家能力反哺主模型，以及策略资产回流。
```

------

# ITD_agent V3 正式完整方案

```text
V3 名称：
Controlled Training, Promotion and Expert-to-Main Evolution Loop

中文名称：
受控训练、模型晋级与专家反哺主模型进化闭环
```

------

# 0. V3 总体定位

V1 解决的是：

```text
主模型是否能发现错误，并调用专家模型局部纠错。
```

V2 解决的是：

```text
V1 产生的错误、纠错、回滚和路由经验，是否能被审查并沉淀为 memory / skill / finetune_pool。
```

V3 解决的是：

```text
V2 沉淀出来的高质量样本、规则、记忆和专家成功案例，是否能真正驱动模型训练、模型晋级、专家差异化强化、主模型能力增强，并把结果回流到系统策略中。
```

V3 是总体方案中的最后一环，对应总体流程末尾：

```text
后台审查
  ↓
Memory Review / Skill Review / Finetune Sample Review / Routing Review / Distillation Review
  ↓
训练触发 / 模型晋级 / 专家能力反哺主模型
```

这正是总体方案定义的闭环终点。

------

# 1. V3 做什么，不做什么

## 1.1 V3 必须做

```text
1. 读取 V2 产生的 finetune_samples；
2. 读取 V2 产生的 memory_records；
3. 读取 V2 产生的 skill_records；
4. 读取 V2 产生的 routing_candidates；
5. 读取 V2 产生的 distillation_candidates；
6. 判断是否达到训练触发条件；
7. 构建训练计划 training_plan；
8. 构建训练数据包 dataset_bundle；
9. 执行 pilot 小规模训练；
10. 对 pilot 模型进行自动评估；
11. 判断是否进入 formal 正式训练；
12. 执行正式训练；
13. 训练后进行 COCO benchmark；
14. 训练后进行 replay regression guard；
15. 训练后进行真实 DOM-only 几何审查；
16. 判断模型是否晋级；
17. 更新 model registry；
18. 更新 model capability profile；
19. 更新专家路由策略候选；
20. 将训练结果回写 memory / skill；
21. 将专家成功样本用于主模型蒸馏或监督微调；
22. 生成完整 training report；
23. 形成 V1 → V2 → V3 的闭环审计链。
```

## 1.2 V3 不应该做

```text
1. 不重新定义 V1 推理闭环；
2. 不重新定义 V2 审查系统；
3. 不绕过 V2 直接用原始 trajectory 训练；
4. 不把所有失败样本无脑放进训练；
5. 不让 LLM 直接决定训练触发；
6. 不让 LLM 直接决定模型晋级；
7. 不做没有 replay guard 的模型替换；
8. 不做无评估的专家模型替换；
9. 不把单次训练结果直接设为 active；
10. 不进入 DEM / CHM / DSM / 小班清查主线 B。
```

------

# 2. V3 和 V1/V2 的衔接关系

| 阶段 | 输入                                | 处理                                 | 输出                                                  | 是否改权重 |
| ---- | ----------------------------------- | ------------------------------------ | ----------------------------------------------------- | ---------- |
| V1   | COCO / DOM、主模型、专家模型        | 推理、评估、ROI、专家纠错、融合/回滚 | trajectory、pending candidates                        | 否         |
| V2   | V1 trajectory 和 pending candidates | 审查、压缩、沉淀、筛选               | memory、skill draft、finetune_pool、distill_candidate | 否         |
| V3   | V2 高质量资产                       | 训练、评估、晋级、蒸馏、策略回流     | 新模型、模型画像、晋级记录、回流策略                  | 是         |

V3 的核心原则是：

```text
只消费 V2 审查通过的资产；
不直接消费 V1 的原始失败样本。
```

也就是：

```text
V1 原始错误 → V2 审查过滤 → V3 训练使用
```

不能变成：

```text
V1 原始错误 → V3 直接训练
```

否则会把低质量、重复、错误专家结果、回滚样本全部混入训练，容易把模型训坏。

------

# 3. V3 的总流程

```text
V2 approved finetune_samples / distillation_candidates
  ↓
训练触发审查 trigger_policy
  ↓
训练目标定义 training_objective
  ↓
样本分组 sample_grouping
  ↓
训练数据包构建 dataset_bundle
  ↓
replay good samples 注入
  ↓
pilot 小规模训练
  ↓
pilot 自动评估
  ↓
是否进入 formal training
  ↓
formal 正式训练
  ↓
训练后 COCO benchmark
  ↓
replay regression guard
  ↓
真实 DOM-only 几何审查
  ↓
模型晋级判断
    ├── reject
    ├── candidate
    ├── shadow
    ├── active
    └── specialized
  ↓
更新 model registry
  ↓
更新 model capability profile
  ↓
更新 routing policy candidate
  ↓
更新 memory / skill
  ↓
生成 training evolution trajectory
```

这与 Hermes 文档中提到的“小步快跑”训练思想一致：先小规模实验性训练，验证可行性，再启动正式训练；训练后自动评估，如果效果未达预期，就回到数据或参数优化阶段，如果效果显著，才固化模型版本。

------

# 4. V3 的核心原则

## 4.1 受控训练原则

V3 可以训练，但必须是“受控训练”。

```text
训练触发 = 规则判断 + 样本质量审查 + replay guard 准备 + 人工批准或强规则批准
```

不是：

```text
LLM 觉得该训练 → 直接训练
```

总体方案中已经明确 LLM 不做指标计算、最终融合裁决、训练触发硬判定和模型晋级硬判定。
所以 V3 中 LLM 只能做：

```text
训练目标解释
训练报告摘要
失败原因总结
候选策略建议
```

不能做：

```text
训练触发硬判定
模型晋级硬判定
模型替换硬判定
```

------

## 4.2 小步快跑原则

V3 不允许一上来就全量正式训练。

必须经过：

```text
pilot training
  ↓
pilot evaluation
  ↓
formal training
```

pilot 失败则不进入 formal training。

------

## 4.3 Replay Guard 原则

任何新模型都不能只看新样本提升。

必须同时检查：

```text
新问题是否改善；
旧能力是否退化；
专家纠错是否仍有效；
主模型大范围稳定性是否下降；
几何异常是否增加。
```

所以每次训练包必须包含：

```text
target failure samples
expert success samples
hard negative samples
replay good samples
replay difficult samples
```

------

## 4.4 主模型通用化、专家模型差异化原则

V3 不能把所有模型都训成一样。

主模型训练目标：

```text
提高通用召回、降低四类错误总体发生率、增强大范围稳定性。
```

专家模型训练目标：

```text
针对某一类错误强化，例如欠分割、过分割、误检、漏检。
```

总体方案中已经明确长期目标是：主模型越来越通用，专家模型越来越差异化，专家路由越来越可靠，训练触发越来越克制，ROI 经验越来越结构化。

------

# 5. V3 新增目录结构

建议新增和扩展：

```text
ITD_agent/
  training_loop/
    contracts.py
    trigger_policy.py
    sample_intake.py
    sample_selector.py
    dataset_packager.py
    dataset_validator.py

    training_plan_builder.py
    training_job_runner.py
    pilot_trainer.py
    formal_trainer.py

    trainer_adapters/
      base.py
      mmdet_trainer.py
      maskdino_trainer.py
      mask2former_trainer.py
      htc_trainer.py
      cascade_mask_rcnn_trainer.py
      legacy_cellpose_sam_trainer.py

    evaluation/
      post_train_evaluator.py
      coco_benchmark_runner.py
      replay_guard.py
      geometry_regression_guard.py
      expert_capability_evaluator.py

    promotion/
      model_registry.py
      model_versioning.py
      model_promotion.py
      model_capability_profile.py
      rollback_manager.py

    distillation/
      expert_to_main_distill.py
      distillation_sample_builder.py
      distillation_plan_builder.py
      teacher_student_dataset.py

    strategy_update/
      routing_policy_updater.py
      skill_activation_reviewer.py
      memory_feedback_writer.py

    reports/
      training_report_builder.py
      promotion_report_builder.py
      v3_closure_report_builder.py

  state/
    migrations/
      003_v3_training_loop.sql

  configs/
    examples/
      itd_agent_train_coco_v3.yaml
      itd_agent_promote_model_v3.yaml
      itd_agent_distill_expert_to_main_v3.yaml

  cli/
    train_cmd.py
    model_cmd.py
```

------

# 6. V3 数据对象设计

## 6.1 TrainingObjective

```python
@dataclass
class TrainingObjective:
    objective_id: str
    source_review_run_id: str
    target_model_role: str  # main_model | expert_model
    target_model_id: str

    target_error_type: str
    target_failure_family: str | None

    objective_type: str
    # reduce_false_negative
    # reduce_false_positive
    # reduce_under_segmentation
    # reduce_over_segmentation
    # improve_boundary_quality
    # improve_general_robustness
    # expert_to_main_distillation

    success_criteria: dict
    safety_criteria: dict
    created_at: str
```

------

## 6.2 TrainingPlan

```python
@dataclass
class TrainingPlan:
    training_plan_id: str
    objective_id: str

    target_model_role: str
    target_model_id: str
    base_model_version: str

    dataset_bundle_id: str
    trainer_backend: str

    pilot_config: dict
    formal_config: dict

    replay_guard_config: dict
    evaluation_config: dict

    approval_status: str  # pending | approved | rejected
    created_at: str
```

------

## 6.3 DatasetBundle

```python
@dataclass
class DatasetBundle:
    bundle_id: str
    source_review_run_ids: list[str]

    target_model_role: str
    target_error_type: str

    train_samples: list[str]
    val_samples: list[str]
    test_samples: list[str]
    replay_samples: list[str]

    sample_stats: dict
    quality_stats: dict
    format: str  # coco | mask | custom
    path: str

    status: str  # built | validated | rejected
    created_at: str
```

------

## 6.4 TrainingJob

```python
@dataclass
class TrainingJob:
    training_job_id: str
    training_plan_id: str

    stage: str  # pilot | formal
    status: str  # pending | running | completed | failed | cancelled

    command: str
    config_path: str
    output_dir: str

    started_at: str | None
    finished_at: str | None

    checkpoint_path: str | None
    log_path: str | None
    metrics_path: str | None
```

------

## 6.5 ModelVersion

```python
@dataclass
class ModelVersion:
    model_version_id: str
    model_id: str
    model_role: str

    base_model_version: str | None
    training_job_id: str | None

    checkpoint_path: str
    config_path: str

    status: str
    # candidate | shadow | active | specialized | deprecated | retired

    capability_profile_id: str | None
    created_at: str
```

------

## 6.6 CapabilityProfile

```python
@dataclass
class CapabilityProfile:
    capability_profile_id: str
    model_version_id: str

    coco_metrics: dict
    error_breakdown: dict
    geometry_metrics: dict
    replay_guard_metrics: dict
    dom_only_metrics: dict

    strengths: list[str]
    weaknesses: list[str]
    recommended_usage: dict

    created_at: str
```

------

# 7. 训练触发机制

V3 训练不能随便触发，必须满足条件。

## 7.1 主模型训练触发条件

主模型训练适合处理：

```text
全局泛化问题
漏检整体偏高
误检整体偏高
欠分割/过分割在多图像上稳定出现
专家多次修正成功且样本质量高
```

建议初始阈值：

```yaml
main_model_trigger:
  min_total_samples: 200
  min_target_failure_samples: 80
  min_expert_success_samples: 50
  min_replay_good_samples: 100
  min_source_trajectories: 20
  min_quality_pass_rate: 0.80
  max_uncertain_sample_ratio: 0.10
  require_human_approval: true
```

## 7.2 专家模型训练触发条件

专家模型训练应该更聚焦。

例如欠分割专家 HTC：

```yaml
expert_model_trigger:
  target_expert: htc
  target_error_type: under_segmentation
  min_target_failure_samples: 50
  min_expert_related_samples: 40
  min_hard_cases: 20
  min_replay_good_samples: 50
  min_quality_pass_rate: 0.85
  require_human_approval: true
```

专家模型不要用泛化训练目标，而要用：

```text
HTC：欠分割 / 粘连树冠拆分
Mask2Former：过分割 / 碎片合并和区域一致性
Cascade Mask R-CNN：误检清理
MaskDINO：漏检召回
```

------

# 8. 样本选择与数据包构建

V3 样本来源只允许来自 V2：

```text
finetune_samples
distillation_candidates
replay_samples
approved memory-linked samples
```

## 8.1 样本分层

每个训练包至少包含：

```text
target_failure_samples
expert_success_samples
hard_negative_samples
boundary_refine_samples
replay_good_samples
replay_difficult_samples
validation_holdout_samples
```

建议比例：

| 样本类型                 | 主模型训练比例 | 专家模型训练比例 |
| ------------------------ | -------------- | ---------------- |
| target_failure_samples   | 35%            | 50%              |
| expert_success_samples   | 25%            | 20%              |
| hard_negative_samples    | 15%            | 10%              |
| boundary_refine_samples  | 10%            | 10%              |
| replay_good_samples      | 10%            | 8%               |
| replay_difficult_samples | 5%             | 2%               |

## 8.2 数据划分原则

必须避免数据泄漏：

```text
同一原始图像 / 同一 tile / 同一 trajectory 来源的高度相似样本，不应同时出现在 train 和 test。
```

建议按：

```text
source_image_id
source_trajectory_id
original_dom_id
```

做 group split。

## 8.3 数据包输出

```text
outputs/training_loop/{training_plan_id}/dataset_bundle/
  train/
    images/
    annotations.json
  val/
    images/
    annotations.json
  test/
    images/
    annotations.json
  replay/
    images/
    annotations.json
  manifest.csv
  bundle_summary.json
  quality_report.json
```

------

# 9. Pilot 训练机制

V3 正式训练前必须先 pilot。

## 9.1 Pilot 目的

```text
验证数据包是否正确；
验证训练脚本是否能跑；
验证 loss 是否正常下降；
验证模型是否出现严重退化；
验证目标错误是否有初步改善；
降低直接正式训练的试错成本。
```

## 9.2 Pilot 配置

```yaml
pilot_training:
  enabled: true
  max_epochs: 3
  sample_limit: 200
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate_scale: conservative
  save_best_checkpoint: true
```

## 9.3 Pilot 通过条件

```yaml
pilot_acceptance:
  loss_must_be_finite: true
  no_nan: true
  target_metric_improvement_min: 0.005
  replay_drop_max: 0.02
  geometry_anomaly_increase_max: 0.03
```

pilot 不通过：

```text
不进入 formal training；
生成 failure_report；
回到 V2 样本筛选或训练计划调整。
```

------

# 10. Formal 正式训练机制

formal training 只在 pilot 通过后启动。

## 10.1 Formal 训练输入

```text
approved TrainingPlan
validated DatasetBundle
pilot accepted checkpoint or base checkpoint
trainer adapter
evaluation config
replay guard config
```

## 10.2 Formal 训练输出

```text
candidate checkpoint
training metrics
loss curve
validation metrics
test metrics
training artifacts
candidate model version
```

输出目录：

```text
outputs/training_loop/{training_plan_id}/formal_training/
  config.yaml
  command.sh
  checkpoints/
    best.pth
    last.pth
  logs/
  metrics/
    train_metrics.json
    val_metrics.json
    test_metrics.json
  artifacts/
```

------

# 11. 训练后评估体系

V3 训练后评估必须包含三层。

## 11.1 COCO Benchmark

评估：

```text
AP
AP50
AP75
AR
mask IoU
boundary IoU
四类错误数量
false_positive_count
false_negative_count
under_segmentation_count
over_segmentation_count
```

## 11.2 Replay Regression Guard

检查：

```text
原本做对的样本是否变差；
原本稳定区域是否新增错误；
主模型是否在非目标错误上退化；
专家模型是否丢失原有特长。
```

## 11.3 DOM-only Geometry Guard

对于无 GT 的真实 DOM-only 样本，检查：

```text
异常小实例是否增加；
异常大实例是否增加；
重复检测是否增加；
边界复杂度是否异常；
局部密度是否异常；
mask 破碎是否变严重。
```

这和总体方案中“主模型晋级必须满足 COCO 指标提升、四类错误不恶化、几何异常不增加、replay 不退化、真实 DOM-only 稳定、artifact 可复现”的要求一致。

------

# 12. 模型晋级机制

V3 中新模型不能直接 active。

必须经历：

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

这是总体方案中已经定义的模型生命周期。

------

## 12.1 candidate

训练完成但未验证充分。

条件：

```text
formal training 完成；
checkpoint 存在；
基础 COCO 指标不崩；
artifact 可复现。
```

------

## 12.2 shadow

可以旁路运行，但不替换线上/默认模型。

条件：

```text
COCO benchmark 有提升；
目标错误下降；
replay guard 未明显退化；
DOM-only geometry guard 未明显恶化。
```

shadow 阶段用途：

```text
在后续 V1 evolve-infer 中旁路比较；
不参与最终融合决策；
只记录 shadow_eval。
```

------

## 12.3 active

可以替换当前默认模型。

主模型 active 条件：

```text
目标错误明显下降；
总体 COCO 指标提升；
四类错误无明显恶化；
replay 不退化；
真实 DOM-only 几何稳定；
连续 N 个 run 表现稳定；
人工批准。
```

专家模型 active 条件：

```text
目标错误类型改善明显；
非目标错误不明显恶化；
ROI 外副作用不增加；
专家被 accept / partial_accept 的比例提升；
reject / rollback 比例下降；
人工批准。
```

------

## 12.4 specialized

对于专家模型尤其重要。

例如：

```text
htc_underseg_v2_specialized
maskdino_fn_recall_v3_specialized
cascade_fp_cleanup_v2_specialized
```

specialized 不一定替代通用 expert，而是在 routing policy 中作为某类 failure_family 的优先专家。

------

# 13. 专家反哺主模型机制

这是 V3 的关键。

V1 中专家模型局部纠错；
V2 中专家成功样本被审查为 distillation_candidate；
V3 中这些样本可以进入主模型训练。

总体方案中已经明确了专家能力反哺主模型的链路：

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

## 13.1 反哺方式

不建议一开始做复杂蒸馏损失，V3 第一版用更稳的方式：

```text
专家成功 mask + COCO GT 校验
→ 生成高质量 pseudo/teacher sample
→ 加入主模型 supervised finetune 数据包
```

也就是：

```text
expert-to-main distillation in V3.1 = supervised finetune with expert-validated samples
```

后续 V3.2 再考虑：

```text
soft mask distillation
teacher logits distillation
feature distillation
```

------

## 13.2 蒸馏样本等级

```text
gold：
专家结果 accept，且优于 GT 匹配下主模型，ROI 外无副作用。

silver：
专家 partial_accept，部分 ROI 明显改善。

bronze：
专家改善有限，但有参考价值。

reject：
专家结果不稳定或回滚。
```

V3 只使用：

```text
gold + silver
```

bronze 只保留，不训练。

------

# 14. 专家模型差异化进化

V3 不能只训练主模型，也要训练专家模型。

## 14.1 HTC 专家

目标：

```text
under_segmentation
merged_crowns
dense_crown_adhesion
large_crown_abnormality
```

训练数据：

```text
欠分割 ROI
专家成功拆分案例
GT 多实例粘连区域
高密度树冠样本
```

------

## 14.2 Mask2Former 专家

目标：

```text
over_segmentation
fragmented_boundary
same_crown_multi_instance
duplicate_detection
```

训练数据：

```text
过分割 ROI
同一 GT 被多个 pred 切碎的样本
边界破碎样本
碎片合并成功案例
```

------

## 14.3 Cascade Mask R-CNN 专家

目标：

```text
false_positive
background_texture_false_positive
shadow_false_positive
tiny_false_positive
```

训练数据：

```text
hard negative
FP ROI
背景纹理误检
阴影误检
低置信噪声样本
```

------

## 14.4 MaskDINO 专家

目标：

```text
false_negative
small_crown_miss
low_contrast_miss
dense_area_miss
edge_miss
```

训练数据：

```text
漏检 ROI
小冠样本
低对比度样本
边缘漏检样本
高密度漏检样本
```

------

# 15. 路由策略回流

V3 训练完成后，不应只更新模型，还要更新“模型能力画像”。

## 15.1 Capability Profile

每个模型版本都要记录：

```text
擅长什么；
不擅长什么；
在哪类 ROI 上提升；
在哪类 ROI 上退化；
适合主模型还是专家；
适合哪个 failure_family；
是否可以 active；
是否只适合 specialized routing。
```

## 15.2 路由策略更新

V3 可以生成 routing policy update，但仍建议分两步：

```text
routing_update_candidate
  ↓
人工或规则审批
  ↓
routing_policy.yaml 更新
```

更新后的路由不是简单替换，而是带版本：

```yaml
expert_routing_policy:
  version: v3_score_based_202605
  route_map:
    under_segmentation:
      primary_expert: htc_v2_specialized
      fallback_expert: maskdino_v1
    over_segmentation:
      primary_expert: mask2former_v2_specialized
    false_positive:
      primary_expert: cascade_mask_rcnn_v2_specialized
    false_negative:
      primary_expert: maskdino_v3_specialized
```

------

# 16. Skill 激活机制

V2 只能生成 skill draft；
V3 可以审批激活 skill。

但是 skill 分两类：

## 16.1 active_readonly_skill

只影响提示、解释、报告，不改变执行策略。

可以自动激活。

## 16.2 active_policy_skill

会影响：

```text
ROI 触发阈值
专家路由
融合保护
训练样本选择
```

必须人工审批。

V3 可以开放：

```bash
itd-agent skill approve --skill-id skill_xxx
```

但要记录：

```text
批准人
批准时间
来源 evidence
影响范围
可回滚版本
```

------

# 17. V3 SQLite 设计

新增：

```text
state/migrations/003_v3_training_loop.sql
```

建议表：

```sql
CREATE TABLE IF NOT EXISTS training_objectives (
    objective_id TEXT PRIMARY KEY,
    source_review_run_id TEXT,
    target_model_role TEXT NOT NULL,
    target_model_id TEXT NOT NULL,
    target_error_type TEXT,
    target_failure_family TEXT,
    objective_type TEXT NOT NULL,
    success_criteria_json TEXT,
    safety_criteria_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_plans (
    training_plan_id TEXT PRIMARY KEY,
    objective_id TEXT NOT NULL,
    target_model_role TEXT NOT NULL,
    target_model_id TEXT NOT NULL,
    base_model_version TEXT,
    dataset_bundle_id TEXT,
    trainer_backend TEXT,
    pilot_config_json TEXT,
    formal_config_json TEXT,
    replay_guard_config_json TEXT,
    evaluation_config_json TEXT,
    approval_status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_bundles (
    bundle_id TEXT PRIMARY KEY,
    source_review_run_ids_json TEXT,
    target_model_role TEXT,
    target_error_type TEXT,
    path TEXT NOT NULL,
    sample_stats_json TEXT,
    quality_stats_json TEXT,
    format TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_jobs (
    training_job_id TEXT PRIMARY KEY,
    training_plan_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    command TEXT,
    config_path TEXT,
    output_dir TEXT,
    started_at TEXT,
    finished_at TEXT,
    checkpoint_path TEXT,
    log_path TEXT,
    metrics_path TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_role TEXT NOT NULL,
    base_model_version TEXT,
    training_job_id TEXT,
    checkpoint_path TEXT NOT NULL,
    config_path TEXT,
    status TEXT NOT NULL,
    capability_profile_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capability_profiles (
    capability_profile_id TEXT PRIMARY KEY,
    model_version_id TEXT NOT NULL,
    coco_metrics_json TEXT,
    error_breakdown_json TEXT,
    geometry_metrics_json TEXT,
    replay_guard_metrics_json TEXT,
    dom_only_metrics_json TEXT,
    strengths_json TEXT,
    weaknesses_json TEXT,
    recommended_usage_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_events (
    promotion_event_id TEXT PRIMARY KEY,
    model_version_id TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    evidence_refs_json TEXT,
    approved_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS distillation_jobs (
    distillation_job_id TEXT PRIMARY KEY,
    training_plan_id TEXT,
    teacher_model TEXT,
    student_model TEXT,
    sample_ids_json TEXT,
    status TEXT NOT NULL,
    output_dir TEXT,
    created_at TEXT NOT NULL
);
```

------

# 18. V3 配置文件

新增：

```text
configs/examples/itd_agent_train_coco_v3.yaml
```

示例：

```yaml
version: v3
mode: controlled_training_loop
mainline_profile: A_DOM_ONLY

source:
  state_db_path: outputs/runtime_state/itd_agent_state.db
  v2_review_run_id: review_xxx
  finetune_pool_root: outputs/evolve_runs/coco_main_expert_loop_test/v2_review/finetune_pool

training_objective:
  target_model_role: main_model
  target_model_id: legacy_cellpose_sam
  objective_type: reduce_false_negative
  target_error_type: false_negative
  target_failure_family: small_crown_recall

trigger_policy:
  require_human_approval: true
  min_total_samples: 200
  min_target_failure_samples: 80
  min_expert_success_samples: 50
  min_replay_good_samples: 100
  min_quality_pass_rate: 0.80
  max_uncertain_sample_ratio: 0.10

dataset_bundle:
  format: coco
  group_split_by:
    - source_image_id
    - source_trajectory_id
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  include_replay_samples: true

pilot_training:
  enabled: true
  max_epochs: 3
  sample_limit: 200
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate_scale: conservative

formal_training:
  enabled: true
  max_epochs: 24
  batch_size: 1
  gradient_accumulation_steps: 8
  save_best_checkpoint: true

trainer:
  backend: external
  adapter: mmdet
  conda_env: mmdetection
  work_dir: outputs/training_loop/{training_plan_id}
  config_template: configs/training_templates/maskdino_coco_tree.yaml

evaluation:
  coco_benchmark: true
  replay_guard: true
  geometry_guard: true
  min_target_metric_improvement: 0.01
  max_replay_drop: 0.02
  max_geometry_anomaly_increase: 0.03

promotion:
  allow_candidate: true
  allow_shadow: true
  allow_active: false
  require_human_approval_for_active: true

guardrails:
  allow_llm_direct_training_trigger: false
  allow_llm_direct_model_promotion: false
  require_pilot_before_formal: true
  require_replay_guard: true
  require_artifact_reproducibility: true
```

------

# 19. V3 CLI 设计

## 19.1 查看训练候选

```bash
itd-agent train candidates --review-run-id review_xxx
```

## 19.2 构建训练计划

```bash
itd-agent train plan --config configs/examples/itd_agent_train_coco_v3.yaml
```

## 19.3 审批训练计划

```bash
itd-agent train approve --training-plan-id plan_xxx
```

## 19.4 执行 pilot 训练

```bash
itd-agent train pilot --training-plan-id plan_xxx
```

## 19.5 执行正式训练

```bash
itd-agent train formal --training-plan-id plan_xxx
```

## 19.6 训练后评估

```bash
itd-agent train evaluate --training-job-id job_xxx
```

## 19.7 查看模型版本

```bash
itd-agent model list
```

## 19.8 模型晋级

```bash
itd-agent model promote --model-version-id modelver_xxx --to shadow
itd-agent model promote --model-version-id modelver_xxx --to active
```

## 19.9 回滚模型

```bash
itd-agent model rollback --model-id legacy_cellpose_sam
```

------

# 20. V3 主流程伪代码

```python
def run_training_loop_v3(config_path: str) -> dict:
    cfg = load_v3_config(config_path)
    assert_v3_guardrails(cfg)

    v2_assets = load_v2_assets(
        review_run_id=cfg["source"]["v2_review_run_id"],
        db_path=cfg["source"]["state_db_path"],
    )

    trigger_result = evaluate_training_trigger(
        v2_assets=v2_assets,
        trigger_policy=cfg["trigger_policy"],
        objective=cfg["training_objective"],
    )

    if not trigger_result.should_train:
        return write_no_train_report(trigger_result)

    objective = build_training_objective(cfg, trigger_result)

    dataset_bundle = build_dataset_bundle(
        objective=objective,
        v2_assets=v2_assets,
        bundle_cfg=cfg["dataset_bundle"],
    )

    validate_dataset_bundle(dataset_bundle)

    training_plan = build_training_plan(
        objective=objective,
        dataset_bundle=dataset_bundle,
        cfg=cfg,
    )

    if cfg["trigger_policy"]["require_human_approval"]:
        mark_plan_pending_approval(training_plan)
        return training_plan.summary()

    pilot_job = run_pilot_training(training_plan)

    pilot_eval = evaluate_trained_model(
        training_job=pilot_job,
        evaluation_cfg=cfg["evaluation"],
    )

    if not pilot_eval.pass_gate:
        reject_training_plan(
            training_plan=training_plan,
            reason=pilot_eval.reason,
        )
        return build_pilot_reject_report(pilot_eval)

    formal_job = run_formal_training(training_plan)

    formal_eval = evaluate_trained_model(
        training_job=formal_job,
        evaluation_cfg=cfg["evaluation"],
    )

    model_version = register_candidate_model(
        formal_job=formal_job,
        formal_eval=formal_eval,
    )

    promotion_decision = evaluate_model_promotion(
        model_version=model_version,
        evaluation=formal_eval,
        promotion_cfg=cfg["promotion"],
    )

    apply_promotion_decision(
        model_version=model_version,
        decision=promotion_decision,
    )

    update_capability_profile(
        model_version=model_version,
        evaluation=formal_eval,
    )

    write_v3_feedback_to_memory_and_skill(
        model_version=model_version,
        evaluation=formal_eval,
        promotion_decision=promotion_decision,
    )

    return build_v3_training_report(
        training_plan=training_plan,
        pilot_job=pilot_job,
        formal_job=formal_job,
        evaluation=formal_eval,
        model_version=model_version,
        promotion_decision=promotion_decision,
    )
```

------

# 21. V3 开发顺序

建议严格按这个顺序开发：

```text
1. 003_v3_training_loop.sql
2. training_loop/contracts.py
3. training_loop/sample_selector.py
4. training_loop/dataset_packager.py
5. training_loop/dataset_validator.py
6. training_loop/trigger_policy.py
7. training_loop/training_plan_builder.py
8. trainer_adapters/base.py
9. trainer_adapters/mmdet_trainer.py
10. training_job_runner.py
11. pilot_trainer.py
12. post_train_evaluator.py
13. coco_benchmark_runner.py
14. replay_guard.py
15. geometry_regression_guard.py
16. formal_trainer.py
17. model_registry.py
18. model_versioning.py
19. model_capability_profile.py
20. model_promotion.py
21. rollback_manager.py
22. expert_to_main_distill.py
23. routing_policy_updater.py
24. skill_activation_reviewer.py
25. training_report_builder.py
26. promotion_report_builder.py
27. cli/train_cmd.py
28. cli/model_cmd.py
29. configs/examples/itd_agent_train_coco_v3.yaml
```

不要先做：

```text
复杂 RL；
GRPO；
多智能体自动训练；
无人工审批 active promotion；
主线 B 的 DEM/CHM/DSM 训练；
外部大规模数据自动爬取；
自动替换线上模型。
```

Hermes 的 RL 闭环可以借鉴“任务定义、轨迹数据组织、质量筛选、小规模实验、自动评估”的工程思想，但你的项目当前更适合受控监督微调和专家样本反哺，不应直接上 RL。Hermes 文档中的 RL 训练强调数据合成、质量筛选、小规模训练、正式训练与自动评估的完整闭环，这一点可以作为 V3 工程节奏参考，而不是照搬算法形态。

------

# 22. V3 验收标准

V3 完成后，必须能回答这些问题：

```text
1. 是否只读取 V2 approved finetune_samples？
2. 是否能判断训练触发条件？
3. 是否能在样本不足时拒绝训练？
4. 是否能构建 dataset_bundle？
5. 是否能避免 train/test 数据泄漏？
6. 是否能注入 replay good samples？
7. 是否能生成 training_plan？
8. 是否能执行 pilot training？
9. 是否能在 pilot 不通过时停止？
10. 是否能执行 formal training？
11. 是否能注册 candidate model？
12. 是否能执行 COCO benchmark？
13. 是否能执行 replay guard？
14. 是否能执行 DOM-only geometry guard？
15. 是否能生成 capability profile？
16. 是否能把模型从 candidate 晋级到 shadow？
17. 是否能在人工审批后晋级 active？
18. 是否能回滚退化模型？
19. 是否能把专家成功样本反哺主模型？
20. 是否能更新 routing_candidate？
21. 是否能生成 V3 training report？
22. 是否能完整追溯 V1 trajectory → V2 sample → V3 model？
```

最终验收命令链：

```bash
itd-agent train candidates --review-run-id review_xxx
itd-agent train plan --config configs/examples/itd_agent_train_coco_v3.yaml
itd-agent train approve --training-plan-id plan_xxx
itd-agent train pilot --training-plan-id plan_xxx
itd-agent train formal --training-plan-id plan_xxx
itd-agent train evaluate --training-job-id job_xxx
itd-agent model promote --model-version-id modelver_xxx --to shadow
```

------

# 23. V3 完成后的完整闭环

V3 完成后，你的总体闭环应该变成：

```text
V1：
COCO / DOM 推理
→ 主模型错误发现
→ ROI 构建
→ 专家纠错
→ 融合 / 回滚
→ trajectory

V2：
trajectory 审查
→ memory 沉淀
→ skill draft
→ finetune_pool
→ routing_candidate
→ distillation_candidate

V3：
finetune_pool / distillation_candidate
→ training trigger
→ dataset bundle
→ pilot training
→ formal training
→ benchmark
→ replay guard
→ geometry guard
→ model candidate
→ shadow
→ active / specialized
→ capability profile
→ routing / skill / memory 回流

下一轮 V1：
使用更新后的模型和策略重新推理
→ 产生新的 trajectory
→ 继续进入 V2/V3
```

这时才是真正的：

```text
Main–Expert Adaptive Evolution Loop
主模型—专家模型自适应进化循环
```

不是只“记录经验”，也不是盲目“自动训练”，而是：

```text
推理产生证据；
审查筛选证据；
训练使用证据；
评估验证训练；
晋级控制风险；
策略回流系统；
下一轮继续闭环。
```

------

# 24. 最终一句话

```text
ITD_agent V3 应被定义为：
在 V1 已经完成真实主—专家推理闭环、V2 已经完成轨迹审查与经验资产沉淀的基础上，
以 V2 审查通过的 finetune_samples、distillation_candidates、routing_candidates 和 skill_records 为输入，
构建一个受控训练闭环；
通过训练触发审查、数据包构建、pilot 小规模训练、formal 正式训练、COCO benchmark、replay guard、DOM-only geometry guard、模型晋级和专家反哺主模型，
最终实现主模型更通用、专家模型更差异化、专家路由更可靠、训练触发更克制、经验资产可持续回流的完整自进化闭环。
```

最简判断标准：

```text
V1：发现问题、调用专家、记录轨迹。
V2：审查轨迹、沉淀经验、构建样本池。
V3：使用样本池训练模型、验证模型、晋级模型、回流策略。
```

到 V3 完成后，你的 ITD_agent 才真正从：

```text
一个会调用多个模型的推理系统
```

升级为：

```text
一个基于轨迹、评估、审查、训练和晋级机制持续进化的单木树冠检测与提取智能体系统。
```