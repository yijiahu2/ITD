from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json


@dataclass(frozen=True)
class TrajectoryRef:
    trajectory_id: str
    run_id: str
    path: Path


def list_v1_trajectories(*, db_path: str | Path, run_id: str | None = None, artifact_root: str | Path | None = None) -> list[TrajectoryRef]:
    refs: list[TrajectoryRef] = []
    db = Path(db_path)
    if db.exists():
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            if run_id:
                rows = conn.execute(
                    "SELECT trajectory_id, run_id, trajectory_path FROM trajectories WHERE run_id = ? ORDER BY trajectory_id",
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT trajectory_id, run_id, trajectory_path FROM trajectories ORDER BY trajectory_id").fetchall()
        refs = [TrajectoryRef(str(row["trajectory_id"]), str(row["run_id"]), Path(row["trajectory_path"])) for row in rows]
    if refs or not artifact_root:
        return refs
    root = Path(artifact_root)
    for path in sorted((root / "trajectories").glob("*.json")):
        payload = read_json(path)
        refs.append(TrajectoryRef(str(payload.get("trajectory_id") or path.stem), str(payload.get("run_id") or run_id or "unknown"), path))
    return refs


def read_trajectory(ref: TrajectoryRef) -> dict[str, Any]:
    payload = read_json(ref.path)
    payload.setdefault("_source_path", str(ref.path))
    return payload
