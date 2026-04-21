from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.orchestration import runtime_steps
from ITD_agent.segmentation import executor


def test_run_semantic_prior_task_uses_worker_when_enabled() -> None:
    cfg = {
        "use_runtime_cache_worker": True,
        "output_dir": "/tmp/out",
        "input_image": "/tmp/in.tif",
        "semantic_prior_script": "/tmp/semantic.py",
    }

    with mock.patch.object(
        runtime_steps,
        "run_semantic_prior_task_via_worker",
        return_value={"m_sem_tif": "/tmp/out/M_sem.tif", "m_sem_png": "/tmp/out/M_sem.png"},
    ) as worker_mock, mock.patch.object(runtime_steps, "get_stage_output_paths", return_value={"m_sem_tif": "/tmp/out/M_sem.tif", "m_sem_png": "/tmp/out/M_sem.png"}):
        result = runtime_steps.run_semantic_prior_task(cfg)

    worker_mock.assert_called_once_with(cfg)
    assert result["m_sem_tif"] == "/tmp/out/M_sem.tif"


def test_execute_segmentation_model_uses_worker_when_enabled() -> None:
    cfg = {
        "use_runtime_cache_worker": True,
        "input_image": "/tmp/in.tif",
        "output_dir": "/tmp/out",
        "conda_sh": "/tmp/conda.sh",
        "conda_env": "test",
        "work_dir": "/tmp",
        "ITD_agent": {
            "segmentation_models": {
                "main_model": {
                    "name": "main_expert",
                    "algorithm": "mmdet_htc",
                }
            }
        },
    }

    with mock.patch.object(
        executor,
        "run_segmentation_task_via_worker",
        return_value={"y_inst_shp": "/tmp/out/Y_inst.shp", "y_inst_tif": "/tmp/out/Y_inst.tif"},
    ) as worker_mock:
        result = executor.execute_segmentation_model(
            cfg=cfg,
            m_sem_tif="/tmp/out/M_sem.tif",
            phase="segmentation_inference",
            model_role="main_model",
        )

    worker_mock.assert_called_once()
    assert result["y_inst_shp"] == "/tmp/out/Y_inst.shp"
    assert result["execution_result"]["status"] == "completed"
