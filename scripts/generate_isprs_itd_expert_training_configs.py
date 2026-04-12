from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.segmentation.finetuning.io_utils import dump_json, dump_yaml, load_yaml


MODEL_SPECS = {
    "dense_adhesion_htc": {
        "template": "/home/xth/forest_agent_project/configs/templates/finetune/finetune_expert_dense_adhesion_htc.yaml",
        "family": "dense_adhesion",
        "init_checkpoint": "/home/xth/mmdetection331/checkpoints/htc_r50_fpn_1x_coco_20200317-7332cf16.pth",
    },
    "shadow_topography_mask2former": {
        "template": "/home/xth/forest_agent_project/configs/templates/finetune/finetune_expert_shadow_topography_mask2former.yaml",
        "family": "shadow_topography",
        "init_checkpoint": "/home/xth/mmdetection331/checkpoints/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth",
    },
    "large_crown_cascade": {
        "template": "/home/xth/forest_agent_project/configs/templates/finetune/finetune_expert_large_crown_cascade.yaml",
        "family": "large_crown_over_split",
        "init_checkpoint": "/home/xth/mmdetection331/checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth",
    },
    "boundary_mask_scoring": {
        "template": "/home/xth/forest_agent_project/configs/templates/finetune/finetune_expert_boundary_mask_scoring.yaml",
        "family": "boundary_calibration",
        "init_checkpoint": "/home/xth/mmdetection331/checkpoints/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth",
    },
    "generalist_maskdino": {
        "template": "/home/xth/forest_agent_project/configs/templates/finetune/finetune_expert_generalist_maskdino.yaml",
        "family": "cross_domain_generalist",
        "init_checkpoint": "/home/xth/MaskDINO/weights/maskdino_r50.pth",
    },
}

DATASET_GROUPS = {
    "dense_adhesion_htc": {
        "train_ids": [4, 5, 9],
        "val_ids": [4, 5],
        "rationale": "亚热带常绿阔叶林与高密小冠幅混交场景，优先强化闭冠粘连与漏分割修复。",
    },
    "shadow_topography_mask2former": {
        "train_ids": [2, 3, 8],
        "val_ids": [3],
        "rationale": "热带雨林、湿润森林、山地森林优先，利用大感受野建模阴影和复杂地形背景。",
    },
    "large_crown_cascade": {
        "train_ids": [3, 6, 7],
        "val_ids": [3, 5],
        "rationale": "湿润森林大冠幅、城市混交林与温带阔叶林更容易出现大冠幅过分裂，适合 Cascade 分阶段细化。",
    },
    "boundary_mask_scoring": {
        "train_ids": [1, 3, 7, 9],
        "val_ids": [3, 4, 5],
        "rationale": "温带混交林、湿润森林和高分辨率阔叶林边界形态复杂，适合用 Mask Scoring 做边界质量校准。",
    },
    "generalist_maskdino": {
        "train_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "val_ids": [3, 4, 5],
        "rationale": "跨国家、跨气候带、跨林型的全域训练，用于建立未见域泛化保底专家。",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expert-split-root")
    parser.add_argument("--config-out-dir", required=True)
    parser.add_argument("--run-suffix", default="20260405_clean_gpu_full")
    parser.add_argument("--output-root", default="/mnt/f/forest_agent_project/outputs/experts_full_clean")
    parser.add_argument("--epochs-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_root = Path(args.expert_split_root).expanduser().resolve() if args.expert_split_root else None
    config_out_dir = Path(args.config_out_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for model_name, spec in MODEL_SPECS.items():
        family = spec["family"]
        family_ann_dir = split_root / family / "annotations" if split_root else None
        cfg = load_yaml(spec["template"])
        cfg["run_name"] = f"{model_name}_{args.run_suffix}"
        cfg["output_dir"] = str(output_root / f"{model_name}_{args.run_suffix}")
        cfg["public_dataset_root"] = str(dataset_root)
        group = DATASET_GROUPS[model_name]
        if family_ann_dir and family_ann_dir.exists():
            cfg["public_dataset_annotation_files_by_role"] = {
                "train": str((family_ann_dir / "instances_train.json").resolve()),
                "val": str((family_ann_dir / "instances_val.json").resolve()),
                "test": str((family_ann_dir / "instances_test.json").resolve()),
            }
        else:
            cfg.pop("public_dataset_annotation_files_by_role", None)
            cfg["public_dataset_include_dataset_ids_by_role"] = {
                "train": list(group["train_ids"]),
                "val": list(group["val_ids"]),
                "test": [],
            }
            cfg["public_dataset_test_splits"] = []
            cfg["public_dataset_holdout_test_fraction"] = 0.10
            cfg["public_dataset_holdout_test_source_roles"] = ["train", "val"]
            cfg["public_dataset_holdout_test_seed"] = 42
        cfg["segmentation_dataset_force_rebuild"] = True
        cfg["segmentation_eval_after_train"] = True
        cfg["segmentation_train_num_gpus"] = 1
        cfg["segmentation_train_device"] = "cuda"
        cfg["segmentation_train_amp"] = False if str(cfg["segmentation_algorithm"]) == "mmdet_mask2former" else True
        cfg["segmentation_train_pin_memory"] = True
        cfg["segmentation_train_init_checkpoint"] = spec["init_checkpoint"]
        cfg["segmentation_train_epochs"] = max(1, int(round(float(cfg.get("segmentation_train_epochs", 8)) * args.epochs_multiplier)))
        cfg["clean_start_policy"] = {
            "require_official_or_empty_checkpoint": True,
            "allow_finetuned_checkpoint": False,
        }
        cfg["expert_dataset_grouping"] = {
            "mode": "metadata_guided_dataset_id_grouping",
            "target_expert_family": family,
            "train_dataset_ids": list(group["train_ids"]),
            "val_dataset_ids": list(group["val_ids"]),
            "rationale": group["rationale"],
            "metadata_source": str((PROJECT_ROOT / "data" / "isprs_itd_dataset_metadata.yaml").resolve()),
            "holdout_policy": "10% train + 10% val into local GT test",
        }
        config_path = config_out_dir / f"{model_name}_{args.run_suffix}.yaml"
        dump_yaml(cfg, config_path)
        manifest[model_name] = {
            "family": family,
            "config_path": str(config_path),
            "output_dir": str(cfg["output_dir"]),
            "init_checkpoint": str(cfg["segmentation_train_init_checkpoint"]),
            "algorithm": str(cfg["segmentation_algorithm"]),
            "train_dataset_ids": list(group["train_ids"]),
            "val_dataset_ids": list(group["val_ids"]),
            "grouping_mode": str(cfg["expert_dataset_grouping"]["mode"]),
        }

    dump_json(
        {
            "dataset_root": str(dataset_root),
            "expert_split_root": str(split_root) if split_root else None,
            "configs": manifest,
        },
        config_out_dir / f"expert_training_manifest_{args.run_suffix}.json",
    )
    print(f"[OK] expert configs generated: {config_out_dir}")


if __name__ == "__main__":
    main()
