from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOTS = ["ITD_agent", "input_layer", "output_layer"]


def test_core_runtime_modules_do_not_import_scripts_package() -> None:
    offenders: list[str] = []
    for root in CORE_ROOTS:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from scripts" in text or "import scripts" in text:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
