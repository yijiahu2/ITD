from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_itd_agent_cli_exposes_only_formal_commands() -> None:
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts" / "itd-agent"), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "{run,evolve,evolve-infer,review,train,state,export}" in result.stdout
    assert "run_evolve_infer" not in result.stdout
    assert "review_v" not in result.stdout
    assert "training_v" not in result.stdout
