# ITD Agent Data Input Boundary v2

本阶段冻结项目的数据输入边界如下。

## Core Online Inputs

- `DOM`
- `DEM`
- `CHM`

这三类输入构成默认主流程的核心在线观测输入。

### Role Definition

- `DOM`
  负责视觉纹理、颜色、语义先验、实例分割主体输入。
- `DEM`
  负责地形背景、坡度/坡向/坡位/地貌等弱地形约束。
- `CHM`
  负责冠层高度、局部峰值、边界高度梯度和高度一致性约束。

## Offline Capability Input

- `public_datasets`

公开数据集不作为默认在线推理主输入，而作为：

- 专家模型训练与微调资源
- benchmark 资源
- 参数模板先验来源
- 学习门槛与经验沉淀的离线证据源

## Optional Reference Input

- `industry_vectors`
- `survey_tables`

这些输入默认不要求存在，但必须保留接口。

### Optional Use Cases

- 参考评估
- 报告对照
- ROI 约束
- 质量诊断增强

## Runtime Requirements

- 没有 `industry_vectors / survey_tables` 时，主流程必须能完整运行。
- 有 `industry_vectors / survey_tables` 时，仍沿用当前 `field_mapping` 契约。
- 没有 `CHM` 时，可退化为 `DOM + DEM` 路径。
- 有 `DSM` 时，作为可选表面高程辅助输入，不替代 `CHM` 的冠层角色。
