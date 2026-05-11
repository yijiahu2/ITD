from __future__ import annotations

from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.segmentation.executor import _resolve_role_entry


def test_executor_prefers_canonical_expert_models_block() -> None:
    cfg = {
        "ITD_agent": {
            "segmentation_models": {
                "expert_models": [
                    {"name": "boundary_calibration_template", "algorithm": "mmdet_mask_scoring_rcnn"},
                ],
            }
        }
    }

    entry = _resolve_role_entry(cfg, model_role="expert_model", preferred_model="boundary_calibration_template")

    assert entry["name"] == "boundary_calibration_template"


def test_runtime_template_registers_boundary_calibration_expert() -> None:
    template_path = Path("/home/xth/forest_agent_project/configs/templates/runtime/runtime_dom177_baseline.yaml")
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    expert_models = (((payload.get("ITD_agent") or {}).get("segmentation_models") or {}).get("expert_models")) or []
    expert_names = {str(item.get("name")) for item in expert_models if isinstance(item, dict)}

    assert "boundary_calibration_template" in expert_names
