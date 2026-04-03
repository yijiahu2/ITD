# LLM Gateway Package

该目录用于承载 `ITD_agent` 的 LLM 网关逻辑，替代原来的单文件实现。

当前职责拆分如下：

- [__init__.py](/home/xth/forest_agent_project/ITD_agent/llm_gateway/__init__.py)
  对外兼容导出，保持 `from ITD_agent.llm_gateway import ...` 不变。
- [gateway.py](/home/xth/forest_agent_project/ITD_agent/llm_gateway/gateway.py)
  负责配置解析、客户端构建、JSON 调用、统一任务入口与响应封装。
- [prompts.py](/home/xth/forest_agent_project/ITD_agent/llm_gateway/prompts.py)
  负责各类 LLM 任务的 prompt 组装。
- [retrospective_input.py](/home/xth/forest_agent_project/ITD_agent/llm_gateway/retrospective_input.py)
  负责复盘阶段的模板化输入压缩。

维护约束：

- 新增能力时，优先落到职责对应模块，不要重新堆回 `__init__.py`
- 若新增新的 LLM 任务类型，优先新增 prompt builder，并在 `gateway.py` 中补统一入口
- 若新增新的模板化输入，优先单独建摘要/模板模块，而不是直接把逻辑塞进 prompt
