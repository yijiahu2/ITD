from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.input_assessment import assess_input_bundle


def test_missing_xiaoban_is_not_blocking_input_issue() -> None:
    assessment = assess_input_bundle(
        cfg={"input_image": "/tmp/dom.tif", "dem_tif": "/tmp/dem.tif"},
        input_manifest={"metadata": {"input_modalities": {"image": True, "dem": True, "chm": False, "inventory": False, "knowledge": False, "public_datasets": False}}},
        terrain_info={"dem_tif": "/tmp/dem.tif"},
        data_processing_summary={"image_profiles": []},
    )

    assert all("行业矢量边界" not in issue for issue in assessment["issues"])
    assert any("在线质量指标" in action for action in assessment["recommended_actions"])
