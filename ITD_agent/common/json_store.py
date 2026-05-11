from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return str(out_path)


def load_json(path: str | Path, default: Any = None) -> Any:
    in_path = Path(path)
    if not in_path.exists():
        return {} if default is None else default
    try:
        with open(in_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {} if default is None else default


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(out_path)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load_jsonl_many(
    paths: Iterable[str | Path],
    *,
    dedupe_key: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for item in load_jsonl(path):
            key = dedupe_key(item) if dedupe_key else ""
            if not key:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    return rows


def load_json_first(paths: Iterable[str | Path]) -> dict[str, Any]:
    for path in paths:
        payload = load_json(path, default={})
        if payload:
            return payload
    return {}


def replace_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    out_path = Path(path)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(out_path)
    return str(out_path)
