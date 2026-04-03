from __future__ import annotations

import argparse
import math
import pprint
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, ensure_dir, load_json, load_yaml, to_bool
from ITD_agent.segmentation.model_registry.common import resolve_algorithm_cfg
from ITD_agent.segmentation.model_registry.mmdet_specs import MMDetAlgorithmSpec, get_mmdet_algorithm_spec, list_mmdet_algorithm_names


SUPPORTED_ALGORITHMS = set(list_mmdet_algorithm_names())
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _algorithm_defaults(algorithm_name: str) -> dict[str, Any]:
    return get_mmdet_algorithm_spec(algorithm_name).training_defaults()


def _resolve_training_env(cfg: dict[str, Any]) -> dict[str, Any]:
    algorithm_name = str(cfg.get("segmentation_algorithm", "")).strip().lower()
    if algorithm_name not in SUPPORTED_ALGORITHMS:
        known = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(f"segmentation_algorithm must be one of [{known}], got: {algorithm_name}")

    merged = _algorithm_defaults(algorithm_name)
    merged.update(resolve_algorithm_cfg(cfg))

    repo_root = str(cfg.get("segmentation_train_repo_root") or merged.get("repo_root") or "").strip()
    conda_sh = str(cfg.get("segmentation_train_conda_sh") or merged.get("conda_sh") or "").strip()
    conda_env = str(cfg.get("segmentation_train_conda_env") or merged.get("conda_env") or "").strip()
    train_base_config = str(cfg.get("segmentation_train_base_config") or merged.get("train_base_config") or "").strip()
    init_checkpoint = str(cfg.get("segmentation_train_init_checkpoint") or merged.get("checkpoint") or merged.get("init_checkpoint") or "").strip()
    driver_module = str(
        merged.get("driver_module") or "ITD_agent.segmentation.model_registry.adapters.mmdet_instance_adapter"
    ).strip()

    if not repo_root:
        raise ValueError("无法解析 segmentation_train_repo_root / repo_root")
    if not conda_sh or not conda_env:
        raise ValueError("无法解析 segmentation_train_conda_sh / segmentation_train_conda_env")
    if not train_base_config:
        raise ValueError("无法解析 segmentation_train_base_config")

    return {
        "algorithm_name": algorithm_name,
        "repo_root": repo_root,
        "conda_sh": conda_sh,
        "conda_env": conda_env,
        "train_base_config": train_base_config,
        "init_checkpoint": init_checkpoint,
        "driver_module": driver_module,
        "resolved_algorithm_cfg": merged,
    }


def _find_best_ckpt(search_root: Path) -> Path | None:
    patterns = ["best*.pth", "best*.pt", "latest*.pth", "epoch_*.pth", "*.pth", "*.pt"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_root.rglob(pattern))

    if not candidates:
        return None

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        priority = 0
        if "best" in name:
            priority += 100
        if "coco" in name and "segm" in name:
            priority += 20
        if "latest" in name:
            priority += 10
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return priority, mtime

    return sorted(candidates, key=score, reverse=True)[0]


def _ensure_public_dataset(args_config: str, cfg: dict[str, Any]) -> tuple[Path, Path]:
    dataset_dir = Path(cfg["output_dir"]) / cfg.get("segmentation_dataset_dirname", "external_segmentation_dataset")
    summary_path = dataset_dir / "prepare_summary.json"
    force_rebuild = to_bool(cfg.get("segmentation_dataset_force_rebuild"), default=False)

    if dataset_dir.exists() and summary_path.exists() and not force_rebuild:
        return dataset_dir, summary_path

    if dataset_dir.exists() and force_rebuild:
        print(f"[INFO] removing existing segmentation dataset dir: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    cmd = [
        sys.executable,
        "-m",
        "ITD_agent.segmentation.model_training.prepare_public_coco_segmentation_dataset",
        "--config",
        args_config,
    ]
    print("[RUN segmentation dataset prepare]")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ITD_agent.segmentation.model_training.prepare_public_coco_segmentation_dataset failed")

    if not summary_path.exists():
        raise RuntimeError(f"未找到 segmentation 数据准备摘要: {summary_path}")

    return dataset_dir, summary_path


def _scheduler_milestones(max_epochs: int) -> list[int]:
    if max_epochs <= 2:
        return [1]
    first = max(1, int(math.floor(max_epochs * 0.67)))
    second = max(first + 1, max_epochs - 1)
    if second >= max_epochs:
        second = max_epochs - 1
    if second <= first:
        return [first]
    return [first, second]


def _literal(value: Any) -> str:
    return pprint.pformat(value, width=120, sort_dicts=False)


def _default_lr(spec: MMDetAlgorithmSpec) -> float:
    if spec.train_loop_style in {"iter_adamw", "epoch_adamw"}:
        return 6.25e-6
    return 0.0025


def _default_weight_decay(spec: MMDetAlgorithmSpec) -> float:
    if spec.train_loop_style in {"iter_adamw", "epoch_adamw"}:
        return 0.05
    return 1.0e-4


def _generic_instance_pipelines() -> tuple[str, str]:
    return (
        """[
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]""",
        """[
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]""",
    )


def _mask2former_instance_pipelines() -> tuple[str, str]:
    return (
        """[
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(1024, 1024),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs'),
]""",
        """[
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]""",
    )


def _pipeline_strings(spec: MMDetAlgorithmSpec) -> tuple[str, str]:
    if spec.train_pipeline_style == "mask2former_lsj_instance":
        return _mask2former_instance_pipelines()
    return _generic_instance_pipelines()


def _model_update_code(spec: MMDetAlgorithmSpec, num_classes: int) -> str:
    if spec.config_style == "mask2former_instance":
        class_weight = [1.0] * num_classes + [0.1]
        return f"""model = _base_.model

model['panoptic_head']['num_things_classes'] = {num_classes}
model['panoptic_head']['num_stuff_classes'] = 0
model['panoptic_head']['loss_cls']['class_weight'] = {_literal(class_weight)}
model['panoptic_fusion_head']['num_things_classes'] = {num_classes}
model['panoptic_fusion_head']['num_stuff_classes'] = 0

del _base_.model
"""

    return f"""model = _base_.model

def _set_num_classes(_module_cfg, _num_classes):
    if _module_cfg is None:
        return
    if isinstance(_module_cfg, list):
        for _item in _module_cfg:
            _item['num_classes'] = _num_classes
    else:
        _module_cfg['num_classes'] = _num_classes

for _target_key in {_literal(list(spec.num_class_target_keys()))}:
    _set_num_classes(model['roi_head'].get(_target_key), {num_classes})

if 'mask_iou_head' in {_literal(list(spec.num_class_target_keys()))}:
    _rcnn_train_cfg = model.get('train_cfg', {{}}).get('rcnn')
    if isinstance(_rcnn_train_cfg, list):
        for _stage_train_cfg in _rcnn_train_cfg:
            _stage_train_cfg.setdefault('mask_thr_binary', 0.5)

del _base_.model
"""


def _training_blocks(
    *,
    spec: MMDetAlgorithmSpec,
    num_train_images: int,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    val_interval: int,
    use_amp: bool,
) -> tuple[str, str, str]:
    if spec.train_loop_style == "iter_adamw":
        iters_per_epoch = max(1, math.ceil(num_train_images / max(batch_size, 1)))
        max_iters = iters_per_epoch * max_epochs
        eval_interval = max(1, iters_per_epoch * max(val_interval, 1))
        milestones = [
            max(1, int(math.floor(max_iters * 0.89))),
            max(1, int(math.floor(max_iters * 0.96))),
        ]
        scheduler = f"""dict(
    type='MultiStepLR',
    begin=0,
    end={max_iters},
    by_epoch=False,
    milestones={_literal(milestones)},
    gamma=0.1,
)"""
        wrapper_type = "AmpOptimWrapper" if use_amp else "OptimWrapper"
        loss_scale_line = "    loss_scale='dynamic',\n" if use_amp else ""
        optimizer = f"""dict(
    type='{wrapper_type}',
{loss_scale_line}    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr={lr},
        weight_decay={weight_decay},
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={{
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
        }},
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2),
)"""
        train_cfg = f"""train_cfg = dict(type='IterBasedTrainLoop', max_iters={max_iters}, val_interval={eval_interval})
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')"""
        return scheduler, optimizer, train_cfg

    if spec.train_loop_style == "epoch_adamw":
        milestones = _scheduler_milestones(max_epochs)
        scheduler = f"""[
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end={max_epochs},
        by_epoch=True,
        milestones={_literal(milestones)},
        gamma=0.1,
    ),
]"""
        wrapper_type = "AmpOptimWrapper" if use_amp else "OptimWrapper"
        loss_scale_line = "    loss_scale='dynamic',\n" if use_amp else ""
        optimizer = f"""dict(
    type='{wrapper_type}',
{loss_scale_line}    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr={lr},
        betas=(0.9, 0.999),
        weight_decay={weight_decay}),
    paramwise_cfg=dict(
        custom_keys={{
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }},
    ),
)"""
        train_cfg = f"""train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={max_epochs}, val_interval={val_interval})
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')"""
        return scheduler, optimizer, train_cfg

    milestones = _scheduler_milestones(max_epochs)
    scheduler = f"""[
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end={max_epochs},
        by_epoch=True,
        milestones={_literal(milestones)},
        gamma=0.1,
    ),
]"""
    wrapper_type = "AmpOptimWrapper" if use_amp else "OptimWrapper"
    loss_scale_line = "    loss_scale='dynamic',\n" if use_amp else ""
    optimizer = f"""dict(
    type='{wrapper_type}',
{loss_scale_line}    optimizer=dict(type='SGD', lr={lr}, momentum=0.9, weight_decay={weight_decay}),
)"""
    train_cfg = f"""train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={max_epochs}, val_interval={val_interval})
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')"""
    return scheduler, optimizer, train_cfg


def _write_generated_config(
    *,
    out_path: Path,
    spec: MMDetAlgorithmSpec,
    env_cfg: dict[str, Any],
    dataset_root: Path,
    train_json: Path,
    val_json: Path,
    test_json: Path,
    training_work_dir: Path,
    num_train_images: int,
    num_classes: int,
    class_names: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    val_interval: int,
    use_amp: bool,
    pin_memory: bool,
    prefetch_factor: int | None,
) -> None:
    train_pipeline, test_pipeline = _pipeline_strings(spec)
    model_update = _model_update_code(spec, num_classes)
    param_scheduler, optim_wrapper, loop_cfg = _training_blocks(
        spec=spec,
        num_train_images=num_train_images,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        val_interval=val_interval,
        use_amp=use_amp,
    )
    prefetch_line = f"    prefetch_factor={prefetch_factor},\n" if prefetch_factor and num_workers > 0 else ""
    checkpoint_by_epoch = spec.train_loop_style != "iter_adamw"
    checkpoint_interval = "1" if checkpoint_by_epoch else "max(1, train_cfg['val_interval'])"
    content = f"""_base_ = {_literal(str(Path(env_cfg["train_base_config"]).resolve()))}

classes = {_literal(class_names)}
metainfo = dict(classes=classes)
data_root = {_literal(str(dataset_root.resolve()) + "/")}
backend_args = None

train_pipeline = {train_pipeline}

test_pipeline = {test_pipeline}

{model_update}

train_dataloader = dict(
    batch_size={batch_size},
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    pin_memory={str(pin_memory)},
{prefetch_line}    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file={_literal(str(train_json.resolve()))},
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    pin_memory={str(pin_memory)},
{prefetch_line}    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file={_literal(str(val_json.resolve()))},
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers={num_workers},
    persistent_workers={str(num_workers > 0)},
    pin_memory={str(pin_memory)},
{prefetch_line}    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file={_literal(str(test_json.resolve()))},
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file={_literal(str(val_json.resolve()))},
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator
test_evaluator = dict(
    type='CocoMetric',
    ann_file={_literal(str(test_json.resolve()))},
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
)

{loop_cfg}

param_scheduler = {param_scheduler}

optim_wrapper = {optim_wrapper}

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval={checkpoint_interval},
        by_epoch={str(checkpoint_by_epoch)},
        save_last=True,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=3,
    ),
    logger=dict(type='LoggerHook', interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

train_cfg_seed = {seed}
randomness = dict(seed=train_cfg_seed)
env_cfg = dict(cudnn_benchmark=True)
load_from = {_literal(env_cfg["init_checkpoint"] if env_cfg["init_checkpoint"] else None)}
resume = False
work_dir = {_literal(str(training_work_dir.resolve()))}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_cfg = _resolve_training_env(cfg)
    algorithm_spec = get_mmdet_algorithm_spec(env_cfg["algorithm_name"])

    output_dir = Path(cfg["output_dir"])
    training_dir = output_dir / cfg.get("segmentation_train_out_dirname", "segmentation_training")
    trainer_root = ensure_dir(training_dir / "external_trainer")
    work_dir = ensure_dir(trainer_root / "work_dir")
    dataset_dir, dataset_summary_json = _ensure_public_dataset(args.config, cfg)
    dataset_summary = load_json(dataset_summary_json)

    ann_dir = dataset_dir / "annotations"
    train_json = ann_dir / "instances_train.json"
    val_json = ann_dir / "instances_val.json"
    test_json = ann_dir / "instances_test.json"
    if not train_json.exists():
        raise FileNotFoundError(f"缺少训练标注文件: {train_json}")
    if not val_json.exists():
        raise FileNotFoundError(f"缺少验证标注文件: {val_json}")
    if not test_json.exists():
        raise FileNotFoundError(f"缺少测试标注文件: {test_json}")

    class_names_raw = cfg.get("segmentation_class_names", ["crown"])
    if isinstance(class_names_raw, str):
        class_names = tuple(x.strip() for x in class_names_raw.split(",") if x.strip())
    else:
        class_names = tuple(str(x).strip() for x in class_names_raw if str(x).strip())
    if not class_names:
        class_names = ("crown",)
    num_classes = int(cfg.get("segmentation_num_classes", len(class_names)))

    batch_size = int(cfg.get("segmentation_train_batch_size", 1))
    num_workers = int(cfg.get("segmentation_train_num_workers", 4))
    max_epochs = int(cfg.get("segmentation_train_epochs", 8))
    lr = float(cfg.get("segmentation_train_lr", _default_lr(algorithm_spec)))
    weight_decay = float(cfg.get("segmentation_train_weight_decay", _default_weight_decay(algorithm_spec)))
    seed = int(cfg.get("segmentation_train_seed", 42))
    val_interval = int(cfg.get("segmentation_train_val_interval", 1))
    use_amp = to_bool(cfg.get("segmentation_train_amp"), default=False)
    pin_memory = to_bool(cfg.get("segmentation_train_pin_memory"), default=True)
    prefetch_factor_raw = cfg.get("segmentation_train_prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw not in {None, ""} else 4
    num_train_images = int(dataset_summary["counts"]["train_images"])
    if num_train_images <= 0:
        raise RuntimeError("segmentation public dataset has no train images")

    generated_config = training_dir / "generated_configs" / f"{env_cfg['algorithm_name']}_public_itd.py"
    _write_generated_config(
        out_path=generated_config,
        spec=algorithm_spec,
        env_cfg=env_cfg,
        dataset_root=Path(cfg["public_dataset_root"]),
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        training_work_dir=work_dir,
        num_train_images=num_train_images,
        num_classes=num_classes,
        class_names=class_names,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        val_interval=val_interval,
        use_amp=use_amp,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    summary: dict[str, Any] = {
        "status": "config_built" if args.build_only else "pending",
        "segmentation_algorithm": env_cfg["algorithm_name"],
        "dataset_dir": str(dataset_dir),
        "dataset_summary_json": str(dataset_summary_json),
        "repo_root": env_cfg["repo_root"],
        "conda_sh": env_cfg["conda_sh"],
        "conda_env": env_cfg["conda_env"],
        "train_base_config": env_cfg["train_base_config"],
        "generated_config": str(generated_config),
        "init_checkpoint": env_cfg["init_checkpoint"],
        "trainer_output_dir": str(trainer_root),
        "work_dir": str(work_dir),
        "class_names": list(class_names),
        "num_classes": num_classes,
        "train_batch_size": batch_size,
        "train_num_workers": num_workers,
        "train_epochs": max_epochs,
        "train_lr": lr,
        "train_weight_decay": weight_decay,
        "train_seed": seed,
        "train_amp": use_amp,
        "train_pin_memory": pin_memory,
        "train_prefetch_factor": prefetch_factor,
        "driver_module": env_cfg["driver_module"],
        "test_json": str(test_json),
    }

    if args.build_only:
        dump_json(summary, training_dir / "train_summary.json")
        print(f"[OK] segmentation training config built only: {generated_config}")
        return

    repo_root = Path(env_cfg["repo_root"]).resolve()
    train_py = repo_root / "tools" / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"未找到 MMDetection train.py: {train_py}")

    bash_cmd = (
        f"source {env_cfg['conda_sh']} && "
        f"conda activate {env_cfg['conda_env']} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export PYTHONPATH={repo_root}:{PROJECT_ROOT}:${{PYTHONPATH:-}} && "
        f"python {train_py} {generated_config} --work-dir {work_dir}"
    )
    print("[RUN segmentation mmdet trainer]")
    print(bash_cmd)
    result = subprocess.run(["bash", "-lc", bash_cmd], cwd=str(repo_root))

    best_ckpt = _find_best_ckpt(work_dir)
    if result.returncode != 0 and best_ckpt is None:
        raise RuntimeError("segmentation mmdet trainer failed and no checkpoint was recovered")
    if best_ckpt is None:
        raise RuntimeError("segmentation mmdet trainer finished but no checkpoint was found")

    summary["status"] = "completed" if result.returncode == 0 else "recovered_ckpt"
    summary["trainer_returncode"] = int(result.returncode)
    summary["trainer_failed_but_ckpt_recovered"] = bool(result.returncode != 0)
    summary["best_ckpt"] = str(best_ckpt)
    summary["generated_infer_config"] = str(generated_config)

    eval_after_train = to_bool(cfg.get("segmentation_eval_after_train"), default=False)
    summary["eval_after_train"] = eval_after_train
    if eval_after_train:
        summary["test_summary_json"] = str(training_dir / "evaluation" / "test_summary.json")
        dump_json(summary, training_dir / "train_summary.json")
        eval_cmd = [
            sys.executable,
            "-m",
            "ITD_agent.segmentation.model_training.test_mmdet_instance",
            "--config",
            args.config,
            "--checkpoint",
            str(best_ckpt),
        ]
        print("[RUN segmentation mmdet evaluation]")
        print(" ".join(eval_cmd))
        eval_result = subprocess.run(eval_cmd)
        summary["eval_returncode"] = int(eval_result.returncode)
        if eval_result.returncode != 0:
            dump_json(summary, training_dir / "train_summary.json")
            raise RuntimeError("segmentation mmdet evaluation failed")

    dump_json(summary, training_dir / "train_summary.json")
    print(f"[OK] segmentation mmdet training done: {best_ckpt}")


if __name__ == "__main__":
    main()
