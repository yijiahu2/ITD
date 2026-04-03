# ITD_agent 记忆库规范

`memory_store` 负责保存结构化经验记忆，不保存大体积原始数据本体。

## 角色边界

- `orchestrator`
  决定何时写入记忆库。
- `LLM网关`
  生成复盘内容与知识更新建议。
- `规划调度`
  只读取记忆上下文，不直接改写记忆。
- `memory_store`
  负责记忆对象落盘、索引和查询。

## 记忆类型

- `execution_trace`
  每次运行都写入，记录完整轨迹摘要。
- `successful_strategy`
  达到阈值时写入，记录成功策略。
- `failure_pattern`
  失败明显或重复失败时写入，记录失败模式。
- `run_retrospective`
  记录 LLM 网关的运行复盘。

## 存储结构

`ITD_agent/memory_store/records/`

- `execution_trace.jsonl`
- `successful_strategy.jsonl`
- `failure_pattern.jsonl`
- `run_retrospective.jsonl`

兼容旧文件：

- `execution_log.jsonl`
- `successful_strategies.jsonl`

`ITD_agent/memory_store/index/`

- `by_scene.json`
- `by_tag.json`

## 核心字段

每条记忆至少包含：

- `memory_id`
- `memory_type`
- `timestamp`
- `run_name`
- `scene_profile`
- `artifact_refs`
- `tags`

其中 `scene_profile` 优先描述：

- `forest_type`
- `terrain_type`
- `image_resolution_m`
- `knowledge_profile_types`
- `public_dataset_roles`

## 主编排写入点

- 任务结束后：写 `execution_trace`
- 结果达标后：写 `successful_strategy`
- 失败显著后：写 `failure_pattern`
- 复盘完成后：写 `run_retrospective`

## 规划调度读取上下文

当前规划调度读取：

- 最近成功策略
- 最近失败模式
- 最近执行轨迹
- 场景相似记忆

这样可以支持：

- 复用成功经验
- 规避重复失败
- 判断是否需要进入微调池或触发微调计划
