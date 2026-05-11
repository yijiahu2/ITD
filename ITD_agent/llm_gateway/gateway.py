from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from input_layer.mainline_profiles import (
    A_DOM_ONLY,
    B_DOM_DEM_CHM_KNOWLEDGE,
    get_mainline_capabilities,
    resolve_mainline_profile,
)

from .audit import build_audit_record, write_audit_record
from .client import build_openai_compatible_client
from .fallback import fallback_structured_plan
from .response_parser import parse_json_response, strip_json_fence
from .prompts import (
    _build_planning_prompt,
    _build_roi_candidate_selection_prompt,
    _build_retrospective_prompt,
    _build_roi_decision_prompt,
)


DEFAULT_SYSTEM_PROMPT = (
    "你是 ITD_agent 的认知决策中心。"
    "你负责围绕树冠分割主流程进行推理、给出结构化决策建议、"
    "总结成功策略与失败模式，并支持记忆更新和自主进化。"
    "你必须只输出合法 JSON，不要输出 markdown 或额外说明。"
)


def build_dynamic_system_prompt(
    *,
    runtime_cfg: dict[str, Any] | None = None,
    task_type: str | None = None,
    base_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    profile = resolve_mainline_profile(runtime_cfg or {})
    capabilities = (runtime_cfg or {}).get("_mainline_capabilities") or get_mainline_capabilities(profile)
    lines = [base_prompt, "", f"当前主线 profile: {profile}。", f"当前任务类型: {task_type or 'unknown'}。"]
    if profile == A_DOM_ONLY:
        lines.extend(
            [
                "主线 A 是完整智能体全流程的 DOM-only 模式，不是简化流程。",
                "可使用 DOM-derived 证据、公开/自制 COCO 数据集摘要、分割残差、ROI 诊断、经验记忆和微调池上下文做决策。",
                "禁止引用 DEM/CHM/DSM、领域知识、规则或 B-only 记忆作为决策证据。",
                "所有建议必须服务于公平 DOM-only SOTA 对比和智能体泛化能力验证。",
            ]
        )
    elif profile == B_DOM_DEM_CHM_KNOWLEDGE:
        lines.extend(
            [
                "主线 B 继承主线 A 的同一套完整智能体流程，并额外启用 DEM 和 CHM。",
                "DEM/CHM 可用于地形、高度、ROI、后处理、可信度增强和树高结构属性提取决策。",
                "公开/自制 COCO 数据集、经验记忆和微调池上下文是 A/B 共享能力。",
                "领域知识、规则等外部知识默认关闭；除非运行配置显式开启，不得作为决策证据或默认模型 tensor 输入。",
            ]
        )
    lines.append(f"可用能力: {json.dumps(capabilities, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass
class LLMGatewayConfig:
    provider: str = "doubao"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMGatewayResponse:
    task_type: str
    status: str
    provider: str | None = None
    model: str | None = None
    parsed_result: dict[str, Any] | None = None
    raw_text: str | None = None
    system_prompt: str | None = None
    prompt_chars: int | None = None
    error: str | None = None
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_gateway_block(runtime_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runtime_cfg, dict):
        return {}
    itd_cfg = runtime_cfg.get("ITD_agent") or {}
    gateway_cfg = itd_cfg.get("llm_gateway")
    return gateway_cfg if isinstance(gateway_cfg, dict) else {}


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _env_value(key: str) -> str | None:
    value = os.environ.get(key)
    if value:
        return _strip_wrapping_quotes(value)
    return None


@lru_cache(maxsize=1)
def _load_bashrc_exports() -> dict[str, str]:
    path = Path("~/.bashrc").expanduser()
    if not path.exists():
        return {}

    exports: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _EXPORT_RE.match(line)
            if not match:
                continue
            key, value = match.groups()
            exports[key] = _strip_wrapping_quotes(value)
    except Exception:
        return {}
    return exports


def _bashrc_or_env(key: str) -> str | None:
    value = _load_bashrc_exports().get(key)
    if value:
        return value
    return _env_value(key)


def resolve_gateway_config(
    *,
    runtime_cfg: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMGatewayConfig:
    gateway_block = _get_gateway_block(runtime_cfg)
    resolved_provider = (
        provider
        or gateway_block.get("provider")
        or _bashrc_or_env("LLM_PROVIDER")
        or "doubao"
    ).strip().lower()
    resolved_model = (
        model
        or gateway_block.get("model")
        or _bashrc_or_env("ARK_MODEL")
        or _bashrc_or_env("OPENAI_MODEL")
    )
    resolved_api_key = (
        api_key
        or gateway_block.get("api_key")
        or _bashrc_or_env("ARK_API_KEY")
        or _bashrc_or_env("OPENAI_API_KEY")
    )
    resolved_base_url = (
        base_url
        or gateway_block.get("base_url")
        or _bashrc_or_env("ARK_BASE_URL")
        or _bashrc_or_env("OPENAI_BASE_URL")
    )
    if resolved_provider == "doubao" and not resolved_base_url:
        resolved_base_url = "https://ark.cn-beijing.volces.com/api/v3"
    return LLMGatewayConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        temperature=float(gateway_block.get("temperature", 0.2)),
        enabled=bool(gateway_block.get("enabled", True)),
    )


def build_client(cfg: LLMGatewayConfig) -> Any:
    return build_openai_compatible_client(cfg)


def gateway_available(cfg: LLMGatewayConfig) -> bool:
    return bool(cfg.enabled and cfg.api_key and cfg.model)


def _strip_json_fence(content: str) -> str:
    return strip_json_fence(content)


def call_json(prompt: str, cfg: LLMGatewayConfig, system_prompt: str) -> dict[str, Any]:
    if not cfg.model:
        raise ValueError("LLM model is not configured.")
    client = build_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg.temperature,
    )
    content = _strip_json_fence((resp.choices[0].message.content or "").strip())
    return parse_json_response(content)


def _invoke_json_task(
    *,
    task_type: str,
    prompt: str,
    system_prompt: str,
    runtime_cfg: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    prompt_chars = len(prompt or "")
    start_time = time.perf_counter()
    cfg = resolve_gateway_config(
        runtime_cfg=runtime_cfg,
        provider=provider,
        model=model,
    )
    if not use_llm:
        response = LLMGatewayResponse(
            task_type=task_type,
            status="disabled",
            provider=cfg.provider,
            model=cfg.model,
            system_prompt=system_prompt,
            prompt_chars=prompt_chars,
        ).to_dict()
        _audit_if_configured(runtime_cfg=runtime_cfg, task_type=task_type, cfg=cfg, prompt=prompt, response=response, start_time=start_time)
        return response
    if not gateway_available(cfg):
        fallback = fallback_structured_plan(task_type=task_type, error="LLM gateway is not configured or disabled.")
        response = LLMGatewayResponse(
            task_type=task_type,
            status="unavailable",
            provider=cfg.provider,
            model=cfg.model,
            parsed_result=fallback,
            system_prompt=system_prompt,
            prompt_chars=prompt_chars,
            error="LLM gateway is not configured or disabled.",
            fallback_used=True,
        ).to_dict()
        _audit_if_configured(runtime_cfg=runtime_cfg, task_type=task_type, cfg=cfg, prompt=prompt, response=response, start_time=start_time)
        return response
    try:
        parsed = call_json(prompt=prompt, cfg=cfg, system_prompt=system_prompt)
        response = LLMGatewayResponse(
            task_type=task_type,
            status="completed",
            provider=cfg.provider,
            model=cfg.model,
            parsed_result=parsed,
            system_prompt=system_prompt,
            prompt_chars=prompt_chars,
        ).to_dict()
        _audit_if_configured(runtime_cfg=runtime_cfg, task_type=task_type, cfg=cfg, prompt=prompt, response=response, start_time=start_time)
        return response
    except Exception as exc:
        fallback = fallback_structured_plan(task_type=task_type, error=str(exc))
        response = LLMGatewayResponse(
            task_type=task_type,
            status="fallback",
            provider=cfg.provider,
            model=cfg.model,
            parsed_result=fallback,
            system_prompt=system_prompt,
            prompt_chars=prompt_chars,
            error=str(exc),
            fallback_used=True,
        ).to_dict()
        _audit_if_configured(runtime_cfg=runtime_cfg, task_type=task_type, cfg=cfg, prompt=prompt, response=response, start_time=start_time)
        return response


def _audit_if_configured(
    *,
    runtime_cfg: dict[str, Any] | None,
    task_type: str,
    cfg: LLMGatewayConfig,
    prompt: str,
    response: dict[str, Any],
    start_time: float,
) -> None:
    gateway_block = _get_gateway_block(runtime_cfg)
    audit_path = gateway_block.get("audit_jsonl")
    if not audit_path:
        return
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    record = build_audit_record(
        task_type=task_type,
        provider=cfg.provider,
        model=cfg.model,
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        run_id=str((runtime_cfg or {}).get("run_id") or (runtime_cfg or {}).get("run_name") or "") or None,
        stage=task_type,
        input_context_path=gateway_block.get("input_context_path"),
        raw_response_path=gateway_block.get("raw_response_path"),
        parsed_response_path=gateway_block.get("parsed_response_path"),
        validation_status="valid" if response.get("parsed_result") is not None else "not_validated",
    )
    write_audit_record(audit_path, record)


def request_planning_decision(
    *,
    planning_stage: str,
    template_cfg: dict[str, Any],
    scheduler_context: dict[str, Any],
    runtime_cfg: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    return _invoke_json_task(
        task_type=f"plan_{planning_stage}_config",
        prompt=_build_planning_prompt(
            planning_stage=planning_stage,
            template_cfg=template_cfg,
            scheduler_context=scheduler_context,
        ),
        system_prompt=build_dynamic_system_prompt(runtime_cfg=runtime_cfg, task_type=f"plan_{planning_stage}_config"),
        runtime_cfg=runtime_cfg,
        use_llm=use_llm,
    )


def request_roi_decision(
    *,
    roi_assessment: dict[str, Any],
    metrics: dict[str, Any],
    runtime_cfg: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    return _invoke_json_task(
        task_type="decide_roi_continuation",
        prompt=_build_roi_decision_prompt(
            roi_assessment=roi_assessment,
            metrics=metrics,
        ),
        system_prompt=build_dynamic_system_prompt(runtime_cfg=runtime_cfg, task_type="decide_roi_continuation"),
        runtime_cfg=runtime_cfg,
        use_llm=use_llm,
    )


def request_roi_candidate_selection(
    *,
    candidate_rois: list[dict[str, Any]],
    metrics: dict[str, Any],
    scene_analysis: dict[str, Any] | None = None,
    runtime_cfg: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    return _invoke_json_task(
        task_type="select_roi_candidates",
        prompt=_build_roi_candidate_selection_prompt(
            candidate_rois=candidate_rois,
            metrics=metrics,
            scene_analysis=scene_analysis,
        ),
        system_prompt=build_dynamic_system_prompt(runtime_cfg=runtime_cfg, task_type="select_roi_candidates"),
        runtime_cfg=runtime_cfg,
        use_llm=use_llm,
    )


def request_run_retrospective(
    *,
    run_summary: dict[str, Any],
    memory_context: list[dict[str, Any]] | None = None,
    finetune_context: list[dict[str, Any]] | None = None,
    runtime_cfg: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    return _invoke_json_task(
        task_type="summarize_run_retrospective",
        prompt=_build_retrospective_prompt(
            run_summary=run_summary,
            memory_context=memory_context,
            finetune_context=finetune_context,
        ),
        system_prompt=build_dynamic_system_prompt(runtime_cfg=runtime_cfg, task_type="summarize_run_retrospective"),
        runtime_cfg=runtime_cfg,
        use_llm=use_llm,
    )
