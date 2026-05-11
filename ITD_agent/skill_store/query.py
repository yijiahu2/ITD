from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ITD_agent.common.json_store import load_jsonl_many


def load_skill_records(
    *,
    db_path: str | Path | None = None,
    review_output_dir: str | Path | None = None,
    statuses: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    statuses = statuses or ["draft", "shadow", "active"]
    if db_path and Path(db_path).exists():
        return _load_from_sqlite(Path(db_path), statuses=statuses, limit=limit)
    if review_output_dir:
        return _load_from_jsonl(Path(review_output_dir) / "skills" / "skill_records.jsonl", statuses=statuses, limit=limit)
    return []


def _load_from_sqlite(db_path: Path, *, statuses: list[str], limit: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM skill_records
        WHERE status IN ({})
        ORDER BY created_at DESC
        LIMIT ?
        """.format(",".join("?" for _ in statuses)),
        [*statuses, limit],
    ).fetchall()
    conn.close()
    records: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in [
            "source_run_ids_json",
            "source_trajectory_ids_json",
            "trigger_conditions_json",
            "recommended_action_json",
            "evidence_summary_json",
            "safety_constraints_json",
        ]:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    item[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        records.append(item)
    return records


def _load_from_jsonl(path: Path, *, statuses: list[str], limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = load_jsonl_many([path], dedupe_key=lambda item: str(item.get("skill_id") or ""))
    filtered = [item for item in records if item.get("status") in statuses]
    return filtered[-limit:]
