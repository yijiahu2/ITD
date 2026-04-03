# External Service Logs

`logs/` 在当前项目中只用于保存外部服务级日志，不属于 `ITD_agent` 主架构的一部分。

## Scope

当前这类日志主要包括：

- `mlflow_server.log`

## Boundary

`ITD_agent` 主流程运行信息、评估结果、中间数据和最终输出，不写入 `logs/`，统一放在每次任务对应的 `output_dir/` 下。

主流程相关内容应保存在：

- `output_dir/input_registry/`
- `output_dir/data_processing/`
- `output_dir/evaluation_analysis/`
- `output_dir/planning_scheduler/`
- `output_dir/final_outputs/`

## Rule

- `logs/` 仅保留外部服务日志
- 不新增 `ITD_agent` 主流程日志到此目录
- 如某个外部服务不再使用，可连同对应日志一起清理
