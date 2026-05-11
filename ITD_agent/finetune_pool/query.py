from __future__ import annotations

from pathlib import Path
from typing import Any

from ITD_agent.common.json_store import load_json_first, load_jsonl_many
from ITD_agent.finetune_pool.store import (
    DEFAULT_FINETUNE_POOL_ROOT,
    LEGACY_FINETUNE_POOL_ROOT,
    SOURCE_LEGACY_FINETUNE_POOL_ROOT,
)


def _pool_roots(finetune_pool_root: str | Path = DEFAULT_FINETUNE_POOL_ROOT) -> list[Path]:
    root = Path(finetune_pool_root)
    roots = [root]
    if root == DEFAULT_FINETUNE_POOL_ROOT:
        for legacy_root in (SOURCE_LEGACY_FINETUNE_POOL_ROOT, LEGACY_FINETUNE_POOL_ROOT):
            if legacy_root != root:
                roots.append(legacy_root)
    return roots


def _load_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    return load_jsonl_many(paths, dedupe_key=lambda item: str(item.get("sample_id") or item.get("candidate_id") or ""))


def _load_json_first(paths: list[Path]) -> dict[str, Any]:
    return load_json_first(paths)


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
