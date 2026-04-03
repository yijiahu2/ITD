from __future__ import annotations

from ITD_agent.segmentation.model_registry.common import run_external_algorithm
from ITD_agent.segmentation.model_registry.mmdet_specs import get_mmdet_algorithm_spec


def run_mmdet_algorithm(algorithm_name: str, cfg: dict, m_sem_tif: str) -> dict:
    spec = get_mmdet_algorithm_spec(algorithm_name)
    return run_external_algorithm(
        cfg,
        m_sem_tif,
        default_driver_module=spec.driver_module,
        default_algorithm_cfg=spec.inference_defaults(),
    )
