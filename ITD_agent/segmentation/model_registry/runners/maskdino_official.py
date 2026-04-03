from __future__ import annotations

from ITD_agent.segmentation.model_registry.common import run_external_algorithm


def run(cfg: dict, m_sem_tif: str) -> dict:
    return run_external_algorithm(
        cfg,
        m_sem_tif,
        default_driver_module="ITD_agent.segmentation.model_registry.adapters.maskdino_instance_adapter",
        default_algorithm_cfg={
            "conda_env": "maskdino",
            "repo_root": "/home/xth/MaskDINO",
            "cwd": "/home/xth/MaskDINO",
            "config_file": "/home/xth/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
            "checkpoint": "/home/xth/MaskDINO/weights/maskdino_r50.pth",
            "device": "cuda",
            "score_thr": 0.2,
            "min_area_px": 50,
            "min_sem_overlap_ratio": 0.01,
            "clip_to_msem": True,
            "required_outputs": ["y_inst_shp"],
        },
    )
