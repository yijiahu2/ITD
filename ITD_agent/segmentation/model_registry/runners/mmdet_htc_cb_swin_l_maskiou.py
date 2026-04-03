from __future__ import annotations

from ITD_agent.segmentation.model_registry.runners.mmdet_common import run_mmdet_algorithm


def run(cfg: dict, m_sem_tif: str) -> dict:
    return run_mmdet_algorithm("mmdet_htc_cb_swin_l_maskiou", cfg, m_sem_tif)
