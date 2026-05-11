from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_audit_record(
    *,
    task_type: str,
    provider: str | None,
    model: str | None,
    prompt: str,
    response: dict[str, Any],
    latency_ms: float | None = None,
    run_id: str | None = None,
    stage: str | None = None,
    input_context_path: str | None = None,
    raw_response_path: str | None = None,
    parsed_response_path: str | None = None,
    validation_status: str | None = None,
) -> dict[str, Any]:
    prompt_hash = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
    return {
        "decision_id": f"llm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "stage": stage or task_type,
        "task_type": task_type,
        "provider": provider,
        "model": model,
        "prompt_hash": prompt_hash,
        "prompt_chars": len(prompt or ""),
        "input_context_path": input_context_path,
        "raw_response_path": raw_response_path,
        "parsed_response_path": parsed_response_path,
        "validation_status": validation_status or ("valid" if response.get("parsed_result") is not None else "not_validated"),
        "latency_ms": latency_ms,
        "status": response.get("status"),
        "fallback_used": bool(response.get("fallback_used")),
        "error": response.get("error"),
    }


def write_audit_record(path: str | Path, record: dict[str, Any]) -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(dst)
