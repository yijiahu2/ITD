from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.orchestration.orchestrator import _build_input_assessment_compat


def test_missing_xiaoban_is_not_blocking_input_issue() -> None:
    assessment = _build_input_assessment_compat(
        {"metadata": {"input_modalities": {"image": True, "dem": True, "chm": False, "inventory": False, "knowledge": False, "public_datasets": False}}}
    )

    assert all("行业矢量边界" not in issue for issue in assessment["issues"])
    assert assessment["modality_status"]["inventory"] is False
