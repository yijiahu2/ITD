from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.cli.main import build_parser


def test_final_cli_help_exposes_product_commands_without_stage_commands() -> None:
    help_text = build_parser().format_help()

    for command in ["run", "review", "train", "state", "export"]:
        assert command in help_text
    for stage_command in ["evolve-infer", "finetune-pool", "review run", "train run"]:
        assert stage_command not in help_text


def test_evolution_cli_file_removed() -> None:
    assert not (PROJECT_ROOT / "ITD_agent" / "evolution" / "cli.py").exists()


def test_core_modules_do_not_import_scripts_private_helpers() -> None:
    offenders: list[str] = []
    for path in (PROJECT_ROOT / "ITD_agent").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from scripts." in text or "import scripts." in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
