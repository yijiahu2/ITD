from __future__ import annotations

from pathlib import Path


def get_phase_dir(runtime_cfg: dict, phase: str, round_idx: int | None = None) -> Path:
    root = Path(runtime_cfg["output_dir"]).resolve() / "evaluation_analysis" / phase
    if round_idx is not None:
        root = root / f"round_{int(round_idx):02d}"
    root.mkdir(parents=True, exist_ok=True)
    return root
