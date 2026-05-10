from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def load_structured(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON config when PyYAML is unavailable.") from exc
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def write_json(path: str | Path, payload: Any) -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(dst)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return str(dst)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return str(dst)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(dst)


def read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))
