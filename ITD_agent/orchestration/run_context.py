from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    config_path: str | None = None
    run_dir: str | None = None
    db_path: str | None = None
    output_path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def build_config_context(config_path: str | Path) -> RunContext:
    return RunContext(config_path=str(config_path))


def build_state_context(db_path: str | Path) -> RunContext:
    return RunContext(db_path=str(db_path))


def build_export_context(run_dir: str | Path, output_path: str | Path) -> RunContext:
    return RunContext(run_dir=str(run_dir), output_path=str(output_path))
