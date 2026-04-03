# ITD_agent 分割模型模块规范

`分割模型` 模块只负责模型接入与执行，不负责策略推理、规则生成或配置决策。

这些决策统一由：

- `LLM网关` 负责推理
- `规划调度` 负责生成结构化计划
- `orchestrator` 负责调用顺序与循环控制

## 当前边界

`分割模型` 保留三类职责：

- 模型注册与选择入口
- 分割推理执行入口
- 微调训练执行入口

它不再负责：

- 主模型/子模型配置生成
- ROI 细化规则生成
- 子模型调用路由规则生成
- 微调训练策略生成
- 先验知识嵌入规则生成

这些内容统一归 `planning/scheduler`。

## 目录职责

- `ITD_agent/segmentation/model_registry`
  模型注册、算法描述、运行器映射、推理入口。
- `ITD_agent/segmentation/executor.py`
  主模型、子模型、ROI 模型的统一执行门面。
- `ITD_agent/segmentation/model_training`
  主模型/子模型训练与测试入口。
- `ITD_agent/segmentation/finetuning`
  微调数据准备、伪标签选择、微调回灌入口。

## 与 orchestrator 的关系

`orchestrator` 只向分割模型模块发送执行请求，不再让其自行推理参数。

建议统一为两类执行对象：

- `SegmentationExecutionRequest`
  用于主模型/子模型推理执行
- `SegmentationFinetuneRequest`
  用于微调训练执行

对应结果：

- `SegmentationExecutionResult`
- `SegmentationFinetuneResult`

## 当前仓库中的边界问题

当前仍存在两类遗留问题：

1. `segmentation/finetuning` 中仍保留了一部分原“数据处理阶段”的训练/回灌逻辑  
例如 `train_data_processing_light.py`、`infer_data_processing_finetuned.py`。
按新架构，它们更接近 `data_processing` 的训练与回灌，而不是 `segmentation` 核心能力。

2. `model_registry` 仍使用 `segmentation_*` 命名和兼容导入  
这是兼容遗留流程，不代表当前架构的推荐对外表达。

## 建议的后续收口方向

第一阶段：

- 保持当前执行代码不动
- 先统一请求/响应契约
- 让 orchestrator 最终通过统一执行入口调用分割模型

第二阶段：

- 将 `segmentation/finetuning` 中与 `data_processing` 更贴近的内容迁出
- 将 `model_registry` 的内部历史阶段命名逐步替换为主模型/子模型表达
- 将训练入口显式区分为：
  - `main_model_training`
  - `child_model_training`
  - `main_model_finetune`
  - `child_model_finetune`

## 知识嵌入计划

当前 `knowledge_embedding_plan` 已由 `planning/scheduler` 统一生成。

在未完成深入研究前，分割模型模块只需要：

- 保留对该计划的输入接口
- 不在当前阶段强制执行

后续再将其逐步接入训练配置生成和推理配置生成。

## ROI 裁剪边界

ROI 区域提取不属于分割模型模块。

按当前架构：

- `评估分析` 发现质量差的区域
- `规划调度` 生成 `roi_extraction` 计划
- `数据处理` 执行 ROI 裁剪
- `分割模型` 只对裁剪后的 ROI 结果执行子模型分割
