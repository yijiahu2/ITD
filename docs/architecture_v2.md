# 三段式架构说明

主维护文档：

- [ITD_agent 现行架构与运行逻辑说明](/home/xth/forest_agent_project/docs/itd_agent_runtime_architecture_guide.md)

当前仓库已收口为三个对外部分，并且核心目录统一命名为 `ITD_agent`：

- `input_layer/`
  负责高分辨率遥感影像、DEM、样地调查数据、领域知识和公开数据集的统一接入与清单化。
- `ITD_agent/`
  作为核心智能体壳，内部继续组织 LLM 网关、规划调度、数据处理、评估分析、分割模型、记忆库和微调池。
- `output_layer/`
  负责统一整理树冠矢量、树木点位、可视化图件和评估报告等最终交付物。

## 当前物理目录映射

- `ITD_agent/planning/agent/`
  原 `agent/`
- `ITD_agent/planning/scheduler/`
  当前“规划调度”模块的实现目录，负责模板管理与运行期计划生成
- `ITD_agent/data_processing/`
  原 `geo_layer/`
- `ITD_agent/segmentation/finetuning/`
  原 `finetune_layer/`
- `ITD_agent/segmentation/model_training/`
  原 `segmentation_train/`
- `ITD_agent/segmentation/model_registry/`
  原 `segmentation_zoo/`
- `output_layer/reporting/`
  原 `reporting/`

## 兼容策略

- 当前仓库已以新目录和新入口为准，不再保留旧层架构的历史命名与历史脚本。
- 新架构优先通过适配器调用保留下来的底层实现，避免一次性重写算法本体。
- 新入口为 `python -m ITD_agent.orchestration.orchestrator --config <config>`。
- `scripts/` 中仅保留围绕当前架构的专项 CLI 包装。

## 阶段语义调整

- 原“第一阶段”逻辑在新架构中归入 `ITD_agent/data_processing`，作为语义先验与预处理的一部分。
- 原“第二阶段”逻辑已经被拆入整个 `ITD_agent` 运行链，不再单独作为阶段概念存在。
- 当前对外表达统一使用 `data_processing / evaluation_analysis / llm_gateway / planning / segmentation`。

## 规划调度职责

- 统一负责配置模板管理。
- 基于评估分析结果、记忆库和微调池上下文生成自适应配置。
- 负责主模型/子模型运行配置、微调训练配置、ROI 提取参数、先验知识嵌入规则和中间数据处理规则。
- 成功策略写入 `ITD_agent/memory_store/`，多次失败样本写入 `ITD_agent/finetune_pool/`。

## 新增存储

- `ITD_agent/memory_store/`
  存放执行日志和成功策略。
- `ITD_agent/finetune_pool/`
  存放待进入微调训练的数据候选与失败案例清单。
