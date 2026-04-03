# ITD_agent 规划调度模块规范

`规划调度` 是 `ITD_agent` 中唯一的配置与规则生成中心。

它不负责执行分割、训练或 ROI 裁剪，只负责在 `LLM网关` 推理结果和结构化上下文的基础上，生成可被其他模块消费的计划产物。

## 角色边界

- `orchestrator`
  负责流程编排、循环控制、模块调用。
- `LLM网关`
  负责认知推理和结构化决策建议。
- `规划调度`
  负责把模板、评估结果、知识上下文和 LLM 决策转成稳定的计划产物。
- `分割模型`
  只负责模型接入、推理入口、训练入口和计划消费。

## 统一产物

`规划调度` 统一生成以下计划：

- `runtime_plan`
  主模型或子模型当前轮的运行配置摘要。
- `roi_extraction_plan`
  ROI 提取与细化计划，包含候选区域筛选、缓冲区、轮次与停止规则。
- `child_model_call_plan`
  子模型候选列表、优先模型、调用路由规则和升级/回退规则。
- `finetune_training_plan`
  微调训练配置模板更新结果和训练参数。
- `knowledge_embedding_plan`
  先验知识如何嵌入 `backbone / neck / initial_prediction / head` 的规则与配置提示。

## 编排入口

`orchestrator` 只通过以下 planner 门面调用规划调度：

- `generate_main_model_plan(...)`
- `generate_child_model_plan(...)`
- `generate_finetune_plan(...)`

不再直接在 orchestrator 中拼装主/子模型 planning runtime cfg。

## 模板库与输出目录

`configs/` 在最新架构中被定位为“模板库”，而不是运行期配置堆放目录。

- `configs/examples/`
  保存示例入口配置。
- `configs/templates/runtime/`
  保存主模型和子模型运行模板。
- `configs/templates/finetune/`
  保存微调训练模板。
- `configs/templates/benchmark/`
  保存 benchmark 与最终评估模板。

`planning/scheduler` 基于这些模板生成本次运行的配置产物，并把输出写到 `output_dir/planning_scheduler/` 下，目录结构与模板目录保持对应关系：

- `planning_scheduler/examples/runtime/<template_name>/...`
- `planning_scheduler/templates/runtime/<template_name>/...`
- `planning_scheduler/templates/finetune/<template_name>/...`
- `planning_scheduler/templates/benchmark/<template_name>/...`

运行期生成配置不再回写到 `configs/`。

## 代码位置

- 门面入口：`ITD_agent/planning/scheduler/planner.py`
- 运行时计划生成：`ITD_agent/planning/scheduler/runtime_scheduler.py`
- 模板更新：`ITD_agent/planning/scheduler/template_manager.py`
- 调度上下文：`ITD_agent/planning/scheduler/context_builder.py`
- 契约定义：`ITD_agent/planning/contracts.py`

## 约束

- `规划调度` 可以生成规则，但不能执行分割或训练。
- ROI 区域裁剪属于 `data_processing` 执行范围，`planning/scheduler` 只生成 `roi_extraction` 规则与计划。
- `分割模型` 可以执行计划，但不能自行生成策略规则。
- `LLM网关` 输出必须是结构化决策，不能直接替代 planner 生成最终运行配置。
