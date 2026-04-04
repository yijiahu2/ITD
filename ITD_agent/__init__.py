from __future__ import annotations

from typing import Any


__all__ = ["config_adapter", "contracts", "llm_gateway", "orchestration", "orchestrator", "run_itd_agent"]


def run_itd_agent(*args: Any, **kwargs: Any):
    from ITD_agent.orchestration.orchestrator import run_itd_agent as _run_itd_agent

    return _run_itd_agent(*args, **kwargs)
