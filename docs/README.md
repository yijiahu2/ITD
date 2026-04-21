# ITD Agent Docs

本目录维护当前 `forest_agent_project` 运行链路和模块边界说明。核心同步面收敛为 5 个文件：先更新 codemap，再同步入口、架构、运行流程或配置说明。

## 推荐阅读顺序

1. [codemap.md](codemap.md)
   当前仓库代码地图、入口、配置表面、脚本和工具。
2. [architecture_v2.md](architecture_v2.md)
   三段式架构、历史目录迁移和当前边界概览。
3. [itd_agent_runtime_architecture_guide.md](itd_agent_runtime_architecture_guide.md)
   主流程、模块职责、关键入口和维护记录。
4. [../configs/README.md](../configs/README.md)
   配置目录、模板边界和顶层配置字段。

专题 spec 仅在对应模块接口发生实质变化时更新；普通结构同步优先落在上述 5 个核心文件。

## 收口文档

- [llm_gateway_input_template_spec.md](llm_gateway_input_template_spec.md)
  已收口为指向主维护文档的轻量入口；LLM 网关细节优先更新主文档的“LLM 网关”和“新增解释记录”部分。
