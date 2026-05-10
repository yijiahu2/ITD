from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_trajectory(trajectory: dict[str, Any], output_dir: str | Path) -> str:
    path = Path(output_dir) / "trajectories" / f"{trajectory['trajectory_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
