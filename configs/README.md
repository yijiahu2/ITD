# Config Template Library

`configs/` 在当前架构中只承担“模板库”职责，不再保存运行期动态生成配置。

## Directory Layout

- `examples/`
  用户可直接复制和修改的示例入口配置。
- `templates/runtime/`
  主模型与子模型运行模板。
- `templates/finetune/`
  微调训练模板。
- `templates/benchmark/`
  benchmark 与最终评估模板。
- `expert_taxonomy/`
  专家族、领域标签和模板路由定义，供 `ITD_agent/planning/scheduler/expert_taxonomy.py` 读取。
- `generated/`
  历史/人工保留的实验配置快照。当前运行期新生成配置不应写回这里。

## Naming Rules

- 示例配置：
  `itd_agent_<scene>_example.yaml`
- 运行模板：
  `runtime_<scene>_<purpose>.yaml`
- 微调模板：
  `finetune_<scope>_<purpose>.yaml`
- benchmark 模板：
  `benchmark_<scope>_<purpose>.yaml`

其中：

- `<scene>` 表示区域、地块或数据域，例如 `dom177`、`dom197`
- `<scope>` 表示数据来源或任务范围，例如 `public_isprs_itd`
- `<purpose>` 表示模板用途，例如 `baseline`、`segmentation_cascade`、`data_processing`

## Runtime Relationship

运行期配置由 `ITD_agent/planning/scheduler` 基于这些模板生成，并落到：

- `output_dir/planning_scheduler/examples/...`
- `output_dir/planning_scheduler/templates/runtime/...`
- `output_dir/planning_scheduler/templates/finetune/...`
- `output_dir/planning_scheduler/templates/benchmark/...`

运行期生成配置不得回写到 `configs/`。已有 `configs/generated/` 仅作为历史或人工保留快照目录，不作为当前调度器的运行期输出目标。

## Config Blocks

当前入口配置使用以下顶层块：

- `runtime`: `run_name`、环境、工作目录，以及可选参考评估字段名。
- `inputs`: 遥感影像、DEM、CHM/DSM、可选调查/小班矢量、领域知识、公开数据集。
- `ITD_agent`: `planning`、`llm_gateway`、`data_processing`、`segmentation_models`。
- `segmentation`: 直接分割参数覆盖项，例如 `diam_list`、`tile`、`overlap`、`tile_overlap`、`bsize`、`augment`、`iou_merge_thr`。
- `evaluation`: 评估分析和最终报告设置。
- `outputs`: `root_dir` 或 `root_base_dir`、`cleanup_policy`、`temp_runtime`。

## Expert Model Templates

当前架构支持在运行模板或示例入口的 `ITD_agent.segmentation_models.expert_models`
中直接声明“独立专家模型模板”。

- 当还没有单独训练好的专家 checkpoint 时，可以先配置模板型专家模型。
- 模板型专家模型仍可复用主分割引擎，但拥有独立的名称、适用场景标签、失败类别标签和默认运行参数。
- 调度器会基于 ROI 评估结果、场景标签、地形标签和失败模式，对这些模板进行默认路由。
- 后续一旦有独立训练好的专家模型，只需要在对应条目中补充 `algorithm`、`checkpoint` 或 `script`，无需改主流程。

推荐的模板字段：

- `name`: 专家模型模板名，供调度器和执行器选择。
- `template_profile`: 设为 `true` 表示当前是模板型专家模型。
- `description`: 模板用途说明。
- `scene_tags`: 适用场景标签，例如 `dense_mixed`、`shadow`。
- `terrain_tags`: 适用地形标签，例如 `steep`、`ridge`、`north_shade`。
- `failure_categories`: 对应失败类别，例如 `dense_canopy_adhesion`。
- `target_error_patterns`: 对应误差模式，例如 `count_under`、`closure_low`。
- `routing_priority`: 默认路由优先级。
- `runtime_overrides`: 模板型子模型的默认运行参数覆盖项。
