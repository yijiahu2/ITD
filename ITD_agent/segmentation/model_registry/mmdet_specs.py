from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_MMDET_REPO_ROOT = "/home/xth/mmdetection331"
_MMDET_CONDA_SH = "/home/xth/anaconda3/etc/profile.d/conda.sh"


@dataclass(frozen=True)
class MMDetAlgorithmSpec:
    name: str
    description: str
    runner_module: str
    train_base_config: str
    infer_config_file: str
    init_checkpoint: str
    config_style: str = "roi_head_instance"
    train_pipeline_style: str = "generic_instance"
    train_loop_style: str = "epoch_sgd"
    extra_num_class_keys: tuple[str, ...] = ()
    conda_sh: str = _MMDET_CONDA_SH
    conda_env: str = "mmdetection"
    repo_root: str = _MMDET_REPO_ROOT
    driver_module: str = "ITD_agent.segmentation.model_registry.adapters.mmdet_instance_adapter"
    device: str = "cuda:0"
    score_thr: float = 0.2
    min_area_px: int = 50
    min_sem_overlap_ratio: float = 0.01
    clip_to_msem: bool = True
    required_outputs: tuple[str, ...] = ("y_inst_shp",)
    trainer_module: str = "ITD_agent.segmentation.model_training.train_mmdet_instance"

    def training_defaults(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "conda_sh": self.conda_sh,
            "conda_env": self.conda_env,
            "train_base_config": self.train_base_config,
            "init_checkpoint": self.init_checkpoint,
            "driver_module": self.driver_module,
        }

    def inference_defaults(self) -> dict[str, Any]:
        return {
            "conda_sh": self.conda_sh,
            "conda_env": self.conda_env,
            "repo_root": self.repo_root,
            "cwd": self.repo_root,
            "config_file": self.infer_config_file,
            "checkpoint": self.init_checkpoint,
            "driver_module": self.driver_module,
            "device": self.device,
            "score_thr": self.score_thr,
            "min_area_px": self.min_area_px,
            "min_sem_overlap_ratio": self.min_sem_overlap_ratio,
            "clip_to_msem": self.clip_to_msem,
            "required_outputs": list(self.required_outputs),
        }

    def num_class_target_keys(self) -> tuple[str, ...]:
        return ("bbox_head", "mask_head", *self.extra_num_class_keys)


MMDET_ALGORITHM_SPECS: dict[str, MMDetAlgorithmSpec] = {
    "mmdet_cascade_mask_rcnn": MMDetAlgorithmSpec(
        name="mmdet_cascade_mask_rcnn",
        description="Cascade Mask R-CNN via MMDetection official config/runtime.",
        runner_module="ITD_agent.segmentation.model_registry.runners.mmdet_cascade_mask_rcnn",
        train_base_config=f"{_MMDET_REPO_ROOT}/configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
        infer_config_file=f"{_MMDET_REPO_ROOT}/configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
        init_checkpoint=f"{_MMDET_REPO_ROOT}/checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth",
    ),
    "mmdet_htc": MMDetAlgorithmSpec(
        name="mmdet_htc",
        description="Hybrid Task Cascade via MMDetection official config/runtime.",
        runner_module="ITD_agent.segmentation.model_registry.runners.mmdet_htc",
        train_base_config=f"{_MMDET_REPO_ROOT}/configs/htc/htc-without-semantic_r50_fpn_1x_coco.py",
        infer_config_file=f"{_MMDET_REPO_ROOT}/configs/htc/htc-without-semantic_r50_fpn_1x_coco.py",
        init_checkpoint=f"{_MMDET_REPO_ROOT}/checkpoints/htc_r50_fpn_1x_coco_20200317-7332cf16.pth",
    ),
    "mmdet_htc_cb_swin_l_maskiou": MMDetAlgorithmSpec(
        name="mmdet_htc_cb_swin_l_maskiou",
        description="Hybrid Task Cascade with project-side CB-Swin-L backbone and MaskIoU head.",
        runner_module="ITD_agent.segmentation.model_registry.runners.mmdet_htc_cb_swin_l_maskiou",
        train_base_config="/home/xth/forest_agent_project/configs/mmdet_custom/htc_cb_swin_l_maskiou.py",
        infer_config_file="/home/xth/forest_agent_project/configs/mmdet_custom/htc_cb_swin_l_maskiou.py",
        init_checkpoint="",
        train_loop_style="epoch_adamw",
        extra_num_class_keys=("mask_iou_head",),
    ),
    "mmdet_mask_scoring_rcnn": MMDetAlgorithmSpec(
        name="mmdet_mask_scoring_rcnn",
        description="Mask Scoring R-CNN via MMDetection official config/runtime.",
        runner_module="ITD_agent.segmentation.model_registry.runners.mmdet_mask_scoring_rcnn",
        train_base_config=f"{_MMDET_REPO_ROOT}/configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py",
        infer_config_file=f"{_MMDET_REPO_ROOT}/configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py",
        init_checkpoint=(
            "https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/"
            "ms_rcnn_r50_caffe_fpn_1x_coco/"
            "ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"
        ),
        extra_num_class_keys=("mask_iou_head",),
    ),
    "mmdet_mask2former": MMDetAlgorithmSpec(
        name="mmdet_mask2former",
        description="Mask2Former instance segmentation via MMDetection official config/runtime.",
        runner_module="ITD_agent.segmentation.model_registry.runners.mmdet_mask2former",
        train_base_config=f"{_MMDET_REPO_ROOT}/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
        infer_config_file=f"{_MMDET_REPO_ROOT}/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
        init_checkpoint=(
            "https://download.openmmlab.com/mmdetection/v3.0/mask2former/"
            "mask2former_r50_8xb2-lsj-50e_coco/"
            "mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
        ),
        config_style="mask2former_instance",
        train_pipeline_style="mask2former_lsj_instance",
        train_loop_style="iter_adamw",
    ),
}


def is_mmdet_algorithm(name: str) -> bool:
    return str(name).strip().lower() in MMDET_ALGORITHM_SPECS


def get_mmdet_algorithm_spec(name: str) -> MMDetAlgorithmSpec:
    key = str(name).strip().lower()
    spec = MMDET_ALGORITHM_SPECS.get(key)
    if spec is None:
        known = ", ".join(sorted(MMDET_ALGORITHM_SPECS))
        raise KeyError(f"Unsupported MMDetection segmentation algorithm: {name}. Known: [{known}]")
    return spec


def list_mmdet_algorithm_names() -> list[str]:
    return sorted(MMDET_ALGORITHM_SPECS)
