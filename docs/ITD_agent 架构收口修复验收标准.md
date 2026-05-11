# ITD_agent 架构收口修复验收标准

## 0. 总体验收结论标准

修复完成后，项目必须满足一句话：

> 用户只需要通过统一 CLI 入口启动完整智能体闭环；项目内部不存在 V1/V2/V3 阶段性入口、重复流程、重复职责、薄 shim 套娃和核心模块反向依赖 scripts 的问题；`llm_gateway` 明确作为规划顾问层参与闭环，但不能直接执行推理、训练或模型晋升。

如果达不到这句话，就不能算修复完成。

------

# 1. 统一入口验收标准

## 1.1 必须存在唯一正式 CLI 入口

应存在：

```text
ITD_agent/cli/main.py
```

并支持统一命令：

```bash
itd-agent run --config xxx.yaml
itd-agent review --config xxx.yaml
itd-agent train --config xxx.yaml
itd-agent state --db xxx.sqlite
itd-agent export --run-dir xxx --out xxx
```

## 1.2 不允许继续暴露阶段性入口

以下命令不应作为正式入口继续存在：

```bash
itd-agent evolve-infer
itd-agent evolve-preflight
itd-agent review run
itd-agent finetune-pool export
itd-agent train run
```

可以在迁移期保留到 release notes，但代码正式入口不应继续依赖这些命名。

## 1.3 不允许存在薄 shim 套娃

不应出现这种文件：

```python
from ITD_agent.cli.main import main

if __name__ == "__main__":
    main()
```

尤其是：

```text
ITD_agent/evolution/cli.py
ITD_agent/orchestrator.py
ITD_agent/orchestration/orchestrator.py
```

如果这些文件拆完只剩转发，就应删除，而不是保留“兼容壳”。

------

# 2. `orchestration/` 验收标准

## 2.1 orchestration 只能做编排

`orchestration/` 只能负责：

```text
构建 RunContext
调用 input_layer
调用 data_processing
调用 segmentation
调用 evaluation_analysis
调用 planning/llm_gateway
调用 evolution
调用 finetune_pool
调用 training_loop
调用 output_layer
汇总 final_summary
```

## 2.2 orchestration 不允许做具体业务计算

`orchestration/` 中不应直接实现：

| 禁止内容          | 应归属模块                  |
| ----------------- | --------------------------- |
| COCO 错误分解     | `evaluation_analysis`       |
| ROI 构建/聚类     | `evolution/roi`             |
| 专家模型路由评分  | `planning`                  |
| 专家结果融合/回滚 | `evolution/fusion`          |
| 样本入池规则      | `finetune_pool`             |
| 训练触发判断      | `training_loop`             |
| 输出成果渲染      | `output_layer`              |
| LLM prompt 构造   | `llm_gateway` 或 `planning` |

## 2.3 主流程函数应足够薄

最终主流程应类似：

```python
def run_workflow(config_path: str) -> dict:
    ctx = build_run_context(config_path)
    input_result = prepare_inputs(ctx)
    processing_result = prepare_data(ctx, input_result)
    main_result = run_main_model(ctx, processing_result)
    eval_result = evaluate_main_result(ctx, main_result)
    adaptive_result = run_adaptive_inference(ctx, eval_result)
    review_result = review_and_store_candidates(ctx, adaptive_result)
    training_result = maybe_run_training(ctx, review_result)
    publish_result = publish_outputs(ctx)
    return build_final_summary(...)
```

如果 `orchestration` 文件仍然超过大量业务逻辑，说明未收口完成。

------

# 3. `evolution/` 验收标准

## 3.1 evolution 只负责推理期自适应

`evolution/` 最终只允许负责：

```text
主模型结果后的错误区域识别
ROI candidate
ROI cluster
expert task builder
expert task runner
expert result comparator
fusion / rollback
trajectory writer
state writer
training candidate 标记
```

## 3.2 evolution 不允许负责训练

`evolution/` 不允许出现：

```text
真实训练执行
模型注册
模型晋升
active route_map 更新
active model 替换
formal training / pilot training 逻辑
```

这些必须属于 `training_loop/`。

## 3.3 evolution 不应再保留 V1 命名

不应继续存在正式函数名：

```python
run_evolve_infer_v1
preflight_evolve_config_v1
supervised_coco_evolve_v1
```

应替换为：

```python
run_adaptive_inference_stage
preflight_runtime_config
adaptive_inference
```

## 3.4 evolution 不应包含 CLI

`ITD_agent/evolution/cli.py` 应删除。
CLI 只能放在：

```text
ITD_agent/cli/
```

------

# 4. `finetune_pool/` 验收标准

## 4.1 finetune_pool 是样本池和审查层

它必须负责：

```text
接收 evolution 产生的候选样本
样本审查
样本质量筛选
样本聚类去重
accepted/rejected/deferred 状态记录
finetune dataset bundle 导出
distillation candidate 管理
```

## 4.2 V2 review 不应继续放在 evolution 下

以下结构不应继续作为正式结构存在：

```text
ITD_agent/evolution/review/
```

应迁移为：

```text
ITD_agent/finetune_pool/review/
```

因为 review 的本质是样本入池审查，不是推理期 evolution。

## 4.3 finetune_pool 不允许跑训练

`finetune_pool/` 中不应直接出现：

```python
run_training_plan(...)
register_model_version(...)
decide_model_promotion(...)
```

这些必须由 `training_loop/` 完成。

------

# 5. `training_loop/` 验收标准

## 5.1 training_loop 只负责受控训练闭环

必须包括：

```text
读取 finetune_pool 导出的训练数据
样本质量门控
训练触发判断
pilot training
formal training
post-train evaluation
replay guard
DOM-only geometry guard
model registry
promotion suggestion
routing update candidate
training feedback
```

当前 `training_runner.py` 方向是对的，但需要完成命名和输入语义收口。

## 5.2 不应再依赖 V2 阶段命名

配置中不应继续以这个作为正式字段：

```yaml
source:
  v2_review_dir: ...
```

应改为：

```yaml
source:
  finetune_pool_dir: ...
  review_asset_dir: ...
  replay_pool_dir: ...
```

## 5.3 training_loop 不允许自动替换 active 模型

除非未来你明确引入人工确认机制，否则验收标准是：

```text
可以注册 candidate model
可以生成 promotion suggestion
可以生成 routing update candidate
不允许自动替换 active model
不允许自动更新 active route_map
```

------

# 6. `llm_gateway/` 验收标准

这是这次修复的重点之一。

## 6.1 必须有正式模块

应存在：

```text
ITD_agent/llm_gateway/
  client.py
  schemas.py
  prompt_builder.py
  response_parser.py
  fallback.py
  audit.py
```

## 6.2 llm_gateway 必须参与闭环

它至少应参与以下一个或多个环节：

```text
主模型结果诊断后的专家模型选择建议
ROI refinement plan 建议
训练样本审查辅助建议
模型晋升说明生成
最终报告解释生成
```

但最重要的是：

```text
planning 调用 llm_gateway
llm_gateway 返回结构化建议
planning validator 校验
orchestration 执行校验后的 plan
```

## 6.3 llm_gateway 不能直接执行动作

`llm_gateway` 不允许：

```text
直接调用 segmentation
直接调用 training_loop
直接写 finetune_pool
直接更新配置文件
直接替换模型
直接更新 route_map
```

它只能输出结构化建议，例如：

```json
{
  "action": "call_expert_model",
  "preferred_expert_family": "under_segmentation_repair",
  "confidence": 0.73,
  "reason": "主模型存在明显漏检和欠分割区域"
}
```

## 6.4 必须有失败降级

当 LLM 不可用时，系统必须能回退到规则路由：

```text
LLM success → 使用 LLM 建议 + rule validator
LLM failed → 使用 rule-based planner
LLM invalid JSON → 使用 fallback planner
LLM timeout → 使用 fallback planner
```

## 6.5 必须有审计记录

每次 LLM 调用至少记录：

```text
run_id
stage
prompt_hash
input_context_path
raw_response_path
parsed_response_path
validation_status
fallback_used
latency_ms
provider
model
```

------

# 7. `scripts/` 依赖验收标准

## 7.1 核心模块不得依赖 scripts

不允许出现：

```python
from scripts.xxx import ...
```

尤其不允许导入 `_private_function`。

当前类似 `real_inference_adapter.py` 依赖 `scripts.benchmark_coco_instance_dataset` 私有函数的情况必须清理。

## 7.2 scripts 只能调用核心模块

允许：

```python
from ITD_agent.segmentation.xxx import ...
from ITD_agent.evolution.xxx import ...
```

不允许反过来：

```python
from scripts.xxx import ...
```

## 7.3 可用检查命令

```bash
grep -R "from scripts" -n ITD_agent input_layer output_layer
grep -R "import scripts" -n ITD_agent input_layer output_layer
```

验收结果必须为空。

------

# 8. 命名收口验收标准

最终正式代码中不应出现阶段性命名：

```text
v1
v2
v3
evolve_infer_v1
review_v2
training_v3
child_model
```

例外：

```text
tests 中可以保留历史测试名
migration notes 中可以保留
兼容性说明文档中可以保留
```

正式模块、正式函数、正式配置字段应统一为：

| 旧命名              | 新命名                               |
| ------------------- | ------------------------------------ |
| V1 evolve infer     | adaptive inference                   |
| V2 review           | finetune pool review                 |
| V3 training         | controlled training                  |
| child model         | expert model                         |
| v2_review_dir       | review_asset_dir / finetune_pool_dir |
| run_evolve_infer_v1 | run_adaptive_inference_stage         |

检查命令：

```bash
grep -R "run_evolve_infer_v1\|review_v2\|training_v3\|child_model\|v2_review_dir" -n ITD_agent input_layer output_layer configs
```

正式代码中不应再出现这些命名。

------

# 9. 配置体系验收标准

## 9.1 最终配置不应暴露阶段概念

最终配置应表达完整闭环，而不是 V1/V2/V3：

```yaml
pipeline:
  run_mode: full_closed_loop
  stages:
    input: true
    data_processing: true
    main_inference: true
    adaptive_inference: true
    review: true
    controlled_training: false
    publish: true
```

而不是：

```yaml
v1:
v2:
v3:
```

## 9.2 主模型和专家模型配置必须统一

应统一放在：

```yaml
ITD_agent:
  segmentation_models:
    main_model:
      name: ...
      algorithm: ...
      checkpoint: ...
    expert_models:
      - name: ...
        expert_family: ...
        target_failure_categories: ...
```

不应同时存在：

```yaml
child_models
expert_models
sub_models
```

如果为了兼容旧配置，兼容逻辑只能放在 config migration 层，不能污染正式 runtime。

------

# 10. 闭环运行验收标准

至少需要三类 smoke test。

## 10.1 DOM-only 主线运行

命令：

```bash
itd-agent run --config configs/examples/dom_only_closed_loop.yaml
```

必须产出：

```text
run_summary.json
state.sqlite
trajectory/*.json
main_model artifacts
evaluation artifacts
adaptive_inference artifacts
publish artifacts
```

## 10.2 有专家模型介入的运行

构造一个主模型明显错误的样本，必须验证：

```text
主模型评估发现问题
ROI candidate 非空
expert task 非空
expert result 被比较
fusion/rollback 有明确 decision
trajectory 记录完整
finetune candidate 被写入
```

## 10.3 LLM 失败降级运行

关闭 LLM 或配置错误 provider，系统必须：

```text
不中断主流程
记录 fallback_used=true
使用 rule-based planner
最终仍能完成 adaptive inference
```

------

# 11. 状态与产物验收标准

每次完整运行必须有统一产物结构：

```text
outputs/<run_id>/
  config/
  input/
  data_processing/
  main_inference/
  evaluation/
  adaptive_inference/
  finetune_pool/
  training/
  publish/
  llm_audit/
  state.sqlite
  run_summary.json
```

不能出现 V1/V2/V3 混乱目录，例如：

```text
outputs/evolve_coco_v1/
outputs/v2_review/
outputs/v3_training/
```

这些可以作为历史迁移目录，但不应作为最终默认输出目录。

------

# 12. 测试验收标准

至少应有以下测试：

```text
tests/test_cli_entrypoints.py
tests/test_orchestration_workflow.py
tests/test_evolution_adaptive_inference.py
tests/test_finetune_pool_review.py
tests/test_training_loop_controlled_training.py
tests/test_llm_gateway_fallback.py
tests/test_no_core_imports_from_scripts.py
tests/test_no_stage_naming_in_runtime.py
```

其中两个测试非常关键：

## 12.1 禁止核心模块依赖 scripts

```python
def test_no_core_imports_from_scripts():
    ...
```

扫描：

```text
ITD_agent/
input_layer/
output_layer/
```

不得出现：

```python
from scripts
import scripts
```

## 12.2 禁止阶段命名污染正式 runtime

扫描正式代码不得出现：

```text
run_evolve_infer_v1
review_v2
training_v3
v2_review_dir
child_model
```

------

# 13. 最终验收表

| 验收项        | 必须结果                                             |
| ------------- | ---------------------------------------------------- |
| 统一 CLI      | 只有 `ITD_agent/cli/main.py` 是正式入口              |
| 阶段入口      | `evolve-infer` 等阶段命令不再作为正式入口            |
| shim          | 不存在无意义转发 shim                                |
| orchestration | 只编排，不做业务计算                                 |
| evolution     | 只做推理期自适应，不训练                             |
| finetune_pool | 负责 review、入池、导出，不训练                      |
| training_loop | 负责受控训练、注册候选、晋升建议                     |
| llm_gateway   | 正式参与 planning，支持 fallback 和 audit            |
| scripts 依赖  | 核心模块不依赖 scripts                               |
| 命名          | 正式代码无 V1/V2/V3 阶段命名                         |
| 配置          | 配置表达完整闭环，不表达阶段试验                     |
| 产物          | 输出目录按最终闭环组织                               |
| 测试          | 有边界测试、入口测试、LLM fallback 测试              |
| 安全          | 不自动替换 active model，不自动更新 active route_map |

------

# 最终通过标准

只有同时满足下面 5 条，才算修复完成：

1. **从用户视角**：只需要一个 `itd-agent run/review/train/export/state` 入口，不需要知道 V1/V2/V3。
2. **从代码视角**：`cli/orchestration/evolution/finetune_pool/training_loop/llm_gateway` 边界清楚，没有重复职责。
3. **从工程视角**：核心模块不依赖 `scripts`，没有薄 shim 套娃。
4. **从闭环视角**：主模型推理、专家模型介入、样本入池、受控训练、模型候选注册、输出发布可以串起来。
5. **从安全视角**：LLM 只给建议，训练和模型晋升受规则、guard、人工确认约束，不会自动失控。