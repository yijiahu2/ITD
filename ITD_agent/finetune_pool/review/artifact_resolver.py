from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def resolve_artifacts_for_trajectory(*, db_path: str | Path, trajectory_id: str) -> dict[str, dict[str, Any]]:
    db = Path(db_path)
    if not db.exists():
        return {}
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT artifact_id, artifact_type, path, format, metadata_json, sha256
            FROM artifacts
            WHERE trajectory_id = ?
            ORDER BY artifact_id
            """,
            (trajectory_id,),
        ).fetchall()
    artifacts: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = Path(row["path"])
        artifacts[str(row["artifact_type"])] = {
            "artifact_id": row["artifact_id"],
            "path": str(path),
            "exists": path.exists(),
            "format": row["format"],
            "sha256": row["sha256"],
        }
    return artifacts


def missing_artifacts(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, ref in sorted(artifacts.items()) if not ref.get("exists")]
