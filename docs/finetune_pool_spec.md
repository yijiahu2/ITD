**Finetune Pool**
`ITD_agent/finetune_pool` 是微调候选池，不直接执行训练。

职责边界：
- 接收运行中沉淀的失败 ROI、困难样本、成功回放样本
- 接收来自数据处理模块的公开数据集分类索引，并转成微调候选
- 形成失败类别聚类和训练触发快照
- 给 `规划调度` 和 `LLM网关` 提供结构化上下文

正式对象：
- `FinetunePoolSample`
- `PublicDatasetCandidate`
- `FinetunePoolCluster`
- `FinetuneTriggerSnapshot`

训练导出：
- `export_finetune_dataset_bundle(...)`
- 输出 `finetune_dataset_bundle.json`
- 将样本拆成：
  - `training_ready_samples`
  - `weak_supervision_candidates`
  - `label_preparation_queue`
  - `replay_samples`
  - `public_dataset_candidates`

样本类型：
- `failed_roi_sample`
- `hard_case_sample`
- `replay_good_sample`
- `public_dataset_candidate`

落盘结构：
- `records/samples.jsonl`
- `records/public_dataset_candidates.jsonl`
- `records/clusters.jsonl`
- `records/training_triggers.jsonl`
- `records/latest_trigger_snapshot.json`
- `index/by_failure_category.json`
- `index/by_target_model.json`

与主编排链的关系：
- `orchestrator` 在任务结束时调用 `register_finetune_pool_assets(...)`
- `orchestrator` 随后调用 `export_finetune_dataset_bundle(...)`
- `planning/scheduler` 读取 `load_finetune_pool_snapshot(...)`
- `LLM网关` 仍可读取 `load_recent_failed_cases(...)` 作为短期失败上下文

职责划分：
- 微调池负责“样本管理和触发判断”
- 规划调度负责“训练计划生成”
- 分割模型负责“训练执行”
