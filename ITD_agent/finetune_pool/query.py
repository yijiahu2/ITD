from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ITD_agent.finetune_pool.store import DEFAULT_FINETUNE_POOL_ROOT, LEGACY_FINETUNE_POOL_ROOT


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _pool_roots(finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT) -> list[Path]:
    root = Path(finetune_pool_root)
    roots = [root]
    if root == DEFAULT_FINETUNE_POOL_ROOT and LEGACY_FINETUNE_POOL_ROOT != DEFAULT_FINETUNE_POOL_ROOT:
        roots.append(LEGACY_FINETUNE_POOL_ROOT)
    return roots


def _load_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for item in _load_jsonl(path):
            sample_id = str(item.get("sample_id") or item.get("candidate_id") or "")
            dedupe_key = sample_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(item)
    return rows


def _load_json_first(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        payload = _load_json(path)
        if payload:
            return payload
    return {}


def load_recent_finetune_pool_samples(
    *,
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
    limit: int = 20,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _load_jsonl_many([root / "records" / "samples.jsonl" for root in _pool_roots(finetune_pool_root)])
    if source_types:
        allowed = {str(item) for item in source_types}
        rows = [row for row in rows if str(row.get("source_type")) in allowed]
    return rows[-limit:]


def load_recent_failed_cases(
    *,
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
    limit: int = 10,
) -> list[dict[str, Any]]:
    return load_recent_finetune_pool_samples(
        finetune_pool_root=finetune_pool_root,
        limit=limit,
        source_types=["failed_roi_sample", "hard_case_sample"],
    )


def load_public_dataset_candidates(
    *,
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = _load_jsonl_many([root / "records" / "public_dataset_candidates.jsonl" for root in _pool_roots(finetune_pool_root)])
    return rows[-limit:]


def load_finetune_pool_snapshot(
    *,
    finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT,
) -> dict[str, Any]:
    return _load_json_first([root / "records" / "latest_trigger_snapshot.json" for root in _pool_roots(finetune_pool_root)])
