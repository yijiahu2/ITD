# 在WSL终端执行以下命令
# 临时使用
# export ARK_API_KEY="yourapikey"
# export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
# export ARK_MODEL="ep-yourmodelid"


# 永久使用
# echo 'export ARK_API_KEY="yourapikey"' >> ~/.bashrc
# echo 'export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"' >> ~/.bashrc
# echo 'export ARK_MODEL="ep-modelid"' >> ~/.bashrc
# source ~/.bashrc

from ITD_agent.llm_gateway import build_client, call_json, resolve_gateway_config


def get_doubao_client():
    cfg = resolve_gateway_config(provider="doubao")
    return build_client(cfg)


def call_doubao_json(prompt: str) -> dict:
    return call_json(
        prompt=prompt,
        cfg=resolve_gateway_config(provider="doubao"),
        system_prompt=(
            "你是森林单木分割参数优化智能体。"
            "你必须只输出合法JSON，不要输出解释性前缀、markdown代码块或额外文字。"
        ),
    )
