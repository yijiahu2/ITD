# ITD Agent Mainline Experiment Matrix

本矩阵用于固化统一 pipeline 下的 A/B 实验口径。A/B 共享同一套智能体全流程，只改变 `runtime.mainline_profile` 和对应输入能力。

## Mainline A: `A_DOM_ONLY`

目标：在 DOM-only 条件下公平对比 SOTA，证明智能体框架带来的泛化收益。

| 阶段 | 输入 | 智能体能力 | 证明点 |
|---|---|---|---|
| A0 | DOM | 固定参数/固定模型，不启用自适应规划 | DOM-only baseline |
| A1 | DOM | 启用输入质量/纹理画像与参数规划 | 参数规划是否提升主模型泛化 |
| A2 | DOM | 启用 ROI 诊断与专家路由 | 局部失败诊断和路由是否提升结果 |
| A3 | DOM | 启用 A-compatible 记忆/回顾，但不使用 B-only 经验 | 经验沉淀是否提升跨场景稳定性 |
| A4 | DOM + public/COCO dataset | 启用失败样本筛选与微调决策 | 样本闭环是否进一步提升泛化 |

A 线报告必须标注 `mainline_profile=A_DOM_ONLY`，禁止引用 DEM/CHM/小班/领域知识作为决策证据；公开/自制 COCO 数据集、经验记忆和微调池属于 A/B 共享能力。

## Mainline B: `B_DOM_DEM_CHM_KNOWLEDGE`

目标：在 A 的统一流程基础上，证明 DEM/CHM 带来的精度、可信度和林业属性输出增强。

| 阶段 | 输入 | 增强能力 | 证明点 |
|---|---|---|---|
| B0 | DOM | 复用 A 最佳全流程 | B 线增强前参照 |
| B1 | DOM + DEM | 地形背景、坡度/坡向/坡位进入调度/ROI/后处理 | DEM 是否改善地形相关误差 |
| B2 | DOM + CHM | 高度峰值、边界高度梯度、实例高度属性 | CHM 是否改善粘连分割和高度输出 |
| B3 | DOM + DEM + CHM | DEM/CHM 增强调度、路由、样本筛选、后处理与树高输出 | 多输入综合收益 |

B 线报告必须与 A 线分栏展示，不能把 B 的多输入结果作为 DOM-only 公平 SOTA 对比结论。

## Standard Config Templates

- A 标准配置：`configs/examples/itd_agent_dom_only_mainline_a.yaml`
- B 标准配置：`configs/examples/itd_agent_dom_dem_chm_minimal_example.yaml`
