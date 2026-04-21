# 三段式架构说明

主维护文档：

- [当前代码地图](codemap.md)
- [ITD_agent 现行架构与运行逻辑说明](itd_agent_runtime_architecture_guide.md)

当前仓库已收口为三个对外部分，并且核心目录统一命名为 `ITD_agent`：

- `input_layer/`
  负责高分辨率遥感影像、DEM、样地调查数据、领域知识和公开数据集的统一接入与清单化。
- `ITD_agent/`
  作为核心智能体壳，内部继续组织 LLM 网关、规划调度、数据处理、评估分析、分割模型、记忆库和微调池。
- `output_layer/`
  负责统一整理树冠矢量、树木点位、可视化图件和评估报告等最终交付物。

## 当前物理目录映射

- `ITD_agent/orchestration/`
  当前主运行编排目录，负责配置准备、单场景主链、grouped inference、运行期路径、输出同步与清理。
- `ITD_agent/evaluation_analysis/`
  当前评估分析目录，负责输入、主模型、ROI、子模型、微调效果、参考质量和最终评估。
- `ITD_agent/llm_gateway/`
  当前 LLM 网关目录，负责结构化 JSON 决策请求、prompt 组装和复盘输入压缩。
- `ITD_agent/planning/agent/`
  局部细化、按小班规划和旧式 agent 辅助逻辑。
- `ITD_agent/planning/scheduler/`
  当前“规划调度”模块的实现目录，负责模板管理、参数搜索、专家族路由与运行期计划生成。
- `ITD_agent/data_processing/`
  当前数据处理目录，负责影像、地形、小班、知识、公开数据集、ROI 和实例后处理。
- `ITD_agent/segmentation/finetuning/`
  微调数据准备、伪标签选择、数据处理微调回灌入口。
- `ITD_agent/segmentation/model_training/`
  分割模型训练、测试和 finetuned 推理入口。
- `ITD_agent/segmentation/model_registry/`
  分割算法注册、runner、adapter 和外部算法执行分发。
- `ITD_agent/memory_store/`
  执行轨迹、成功策略、失败模式和运行复盘的结构化记忆库。
- `ITD_agent/finetune_pool/`
  失败样本、回放样本、公开数据候选、训练触发快照和微调数据包导出。
- `output_layer/reporting/`
  最终报告与评估结果整理。
- `scripts/`
  当前命令行包装、评估、微调、benchmark、切片和清理脚本。
- `tools/`
  运行期缓存 worker/client、stage runner、进程执行和栅格辅助工具。

## 兼容策略

- 当前仓库以 `input_layer/`、`ITD_agent/`、`output_layer/`、`configs/`、`scripts/`、`tools/` 的真实目录为准。
- 新架构通过 adapter/runner 调用外部分割实现，避免把外部框架细节塞进主编排器。
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
