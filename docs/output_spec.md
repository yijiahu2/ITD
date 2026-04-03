# ITD_agent Output Specification

`ITD_agent` 的输出分为两层：

## 1. Internal Runtime Outputs

运行中间数据统一保存在 `output_dir/` 下，供系统追溯和大模型调用。典型目录包括：

- `input_registry/`
- `prepared_inputs/`
- `planning_scheduler/`
- `roi_refinement/`
- `terrain_cache/`
- `final_outputs/`

说明：

- 顶层 `logs/` 目录不属于 `ITD_agent` 主流程输出体系
- `logs/` 仅用于保存外部服务级日志，例如 `mlflow_server.log`

当配置启用临时运行目录时：

- `persistent_output_dir`
  指向最终保留目录，例如 `/mnt/f/forest_agent_project/outputs/<run_name>`
- `output_dir`
  指向运行期临时目录，例如 `/tmp/itd_agent_runtime/<run_name>`

运行结束后，系统会将需要保留的内部结果同步回 `persistent_output_dir`，并按清理策略删除临时目录。

## 2. Final Deliverables

最终对外交付只保留以下文件：

- `tree_crowns.shp`
- `tree_points.shp`
- `segmentation_visualization.png`
- `final_evaluation_report.md`
- `final_evaluation_report.json`

## Final Evaluation Modes

### benchmark

当配置了 `evaluation.final_report.benchmark.gt_tree_crowns_shp` 且真值文件存在时，最终报告自动切换为 `benchmark` 模式，输出：

- `AP50`
- `AP75`
- `R2`

其中 `R2` 当前定义为 `IoU=0.50` 下匹配树冠面积的决定系数。

### inventory_consistency

当缺少可用真值时，最终报告自动退回 `inventory_consistency` 模式，输出融合后结果与样地/小班约束的一致性指标。

## Cleanup Policy

- `minimal`
  仅保留最终结果和必要摘要
- `standard`
  保留输入注册、数据处理摘要、规划配置、最终评估中间结果
- `debug`
  保留完整运行期目录，不删除临时目录
