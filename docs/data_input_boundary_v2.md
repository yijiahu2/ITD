# ITD Agent Data Input Boundary v2

本阶段冻结项目的数据输入边界：**只有一套统一 ITD_agent 全流程，主线 A/B 由 `mainline_profile` 控制输入能力，不拆分两套 pipeline。**

## Unified Pipeline Input Profiles

A/B 共享相同运行逻辑：

`input_layer → data_processing → input_assessment → planning_scheduler → llm_gateway → main_model_loop → expert_model_loop / ROI refinement → memory / finetune_pool → postprocess → final_report`

区别只在 profile 打开的 DEM/CHM 增强输入、可选外部知识接口和增强输出字段。

### `A_DOM_ONLY`

- 在线观测输入：`DOM`
- 训练/验证/推理数据：公开数据集与自制 COCO 数据集
- 经验上下文：`memory_store` 与 `finetune_pool`
- 外部知识层：默认关闭
- DEM / CHM / DSM：默认关闭
- 目标：公平对比 DOM-only SOTA，验证智能体框架在相同输入条件下的泛化提升。
- 运行要求：完整智能体全流程必须可运行，不能因为没有 DEM/CHM/小班/知识而降级为半流程。

### `B_DOM_DEM_CHM_KNOWLEDGE`

- 在线观测输入：`DOM + DEM + CHM`
- 训练/验证/推理数据：公开数据集与自制 COCO 数据集
- 经验上下文：`memory_store` 与 `finetune_pool`
- 外部知识层：默认关闭，保留可选接口；现阶段不作为主线优化重点
- 目标：在 A 的统一流程基础上进一步提升分割精度，并提取树高、冠高、结构信息。
- 运行要求：继承 A 的全流程逻辑，只增加 DEM/CHM 可用证据和增强输出。

## Input Role Definition

- `DOM`
  - A/B 都必须支持。
  - 负责视觉纹理、颜色、语义先验、实例分割主体输入。
- `DEM`
  - 仅 B profile 默认启用。
  - 负责地形背景、坡度/坡向/坡位/地貌等弱地形约束。
- `CHM`
  - 仅 B profile 默认启用。
  - 负责冠层高度、局部峰值、边界高度梯度、高度一致性约束、树高/冠高/结构属性提取。
- `DSM`
  - B profile 可选辅助输入，不替代 CHM 的冠层角色。

## Shared Learning And Dataset Layer

A/B 共享以下能力：

- `public_datasets`
- 自制 COCO 数据集
- `memory_store`
- `finetune_pool`

这些输入用于：

- 训练、验证和推理数据接入
- 动态系统提示词中的经验上下文补充
- 训练决策
- 路由决策
- 样本筛选
- 报告解释与诊断

它们不属于 A/B 差异项，也不作为 DEM/CHM 一类的在线增强输入。

## Optional External Knowledge Layer

外部知识层包括：

- `industry_vectors`
- `survey_tables`
- `domain_knowledge`

这些输入默认关闭。A profile 不保留外部知识接口；B profile 保留可选接口，但现阶段先不作为主线优化重点。开启后目标是：

- 动态系统提示词补充
- 训练决策
- 路由决策
- 样本筛选
- 后处理约束
- 报告解释与诊断

它们不作为默认模型 tensor 输入。除非后续明确开启多模态/知识嵌入模型实验，否则不得把外部知识描述为默认模型输入通道。

## Runtime Requirements

- 没有 `industry_vectors / survey_tables / domain_knowledge` 时，A/B 主流程必须完整运行。
- A/B profile 下公开数据集、自制 COCO 数据集、经验记忆和微调池均属于共享能力。
- A profile 下即使配置误填 DEM/CHM/小班/知识，也应通过 profile gate 忽略，避免污染 DOM-only benchmark。
- B profile 下 DEM/CHM 可参与场景分析、调度、路由、样本筛选、后处理、可信度增强和树高结构输出。
- A/B 结果必须分开报告：
  - `A_DOM_ONLY`：公平 SOTA 对比主结果。
  - `B_DOM_DEM_CHM_KNOWLEDGE`：增强输入收益结果。
