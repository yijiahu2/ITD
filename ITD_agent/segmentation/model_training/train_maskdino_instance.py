from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, ensure_dir, load_json, load_yaml, to_bool
from ITD_agent.segmentation.model_training.expert_injection import build_training_injection_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _algorithm_defaults() -> dict[str, Any]:
    return {
        "repo_root": "/home/xth/MaskDINO",
        "conda_sh": "/home/xth/anaconda3/etc/profile.d/conda.sh",
        "conda_env": "maskdino",
        "train_base_config": "/home/xth/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        "init_checkpoint": "/home/xth/MaskDINO/weights/maskdino_r50.pth",
        "driver_module": "ITD_agent.segmentation.model_registry.adapters.maskdino_instance_adapter",
    }


def _find_best_ckpt(search_root: Path) -> Path | None:
    patterns = ["model_final.pth", "model_*.pth", "*.pth"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_root.rglob(pattern))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        priority = 0
        if name == "model_final.pth":
            priority += 100
        if name.startswith("model_"):
            priority += 20
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
        # Rebuild in place so reruns do not depend on deleting large output trees first.
        print(f"[INFO] rebuilding segmentation dataset in place: {dataset_dir}", flush=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "ITD_agent.segmentation.model_training.prepare_public_coco_segmentation_dataset",
        "--config",
        args_config,
    ]
    print("[RUN segmentation dataset prepare]", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ITD_agent.segmentation.model_training.prepare_public_coco_segmentation_dataset failed")
    if not summary_path.exists():
        raise RuntimeError(f"未找到 segmentation 数据准备摘要: {summary_path}")
    return dataset_dir, summary_path


def _scheduler_steps(max_iter: int) -> tuple[int, ...]:
    if max_iter <= 2:
        return (1,)
    first = max(1, int(math.floor(max_iter * 0.67)))
    second = max(first + 1, int(math.floor(max_iter * 0.92)))
    if second >= max_iter:
        second = max_iter - 1
    if second <= first:
        return (first,)
    return (first, second)


def _build_dataset_names(algorithm_name: str) -> tuple[str, str, str]:
    train_name = f"forest_agent_{algorithm_name}_public_itd_train"
    val_name = f"forest_agent_{algorithm_name}_public_itd_val"
    test_name = f"forest_agent_{algorithm_name}_public_itd_test"
    return train_name, val_name, test_name


def _grad_accum_steps(cfg: dict[str, Any]) -> int:
    raw = cfg.get("segmentation_train_grad_accum_steps", 1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def _write_generated_config(
    *,
    out_path: Path,
    base_config: str,
    train_dataset_name: str,
    val_dataset_name: str,
    output_dir: Path,
    init_checkpoint: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    max_iter: int,
    eval_period: int,
    checkpoint_period: int,
    base_lr: float,
    weight_decay: float,
    detections_per_image: int,
    device: str,
    use_amp: bool,
) -> None:
    steps = _scheduler_steps(max_iter)
    content = f"""_BASE_: {base_config}

MODEL:
  WEIGHTS: {init_checkpoint}
  DEVICE: "{device}"
  SEM_SEG_HEAD:
    NUM_CLASSES: {num_classes}
  MaskDINO:
    TEST:
      SEMANTIC_ON: False
      INSTANCE_ON: True
      PANOPTIC_ON: False
      OBJECT_MASK_THRESHOLD: 0.25
      OVERLAP_THRESHOLD: 0.8

DATASETS:
  TRAIN: ("{train_dataset_name}",)
  TEST: ("{val_dataset_name}",)

DATALOADER:
  NUM_WORKERS: {num_workers}
  FILTER_EMPTY_ANNOTATIONS: True

SOLVER:
  IMS_PER_BATCH: {batch_size}
  BASE_LR: {base_lr}
  MAX_ITER: {max_iter}
  STEPS: {steps}
  CHECKPOINT_PERIOD: {checkpoint_period}
  WEIGHT_DECAY: {weight_decay}
  AMP:
    ENABLED: {str(use_amp)}

TEST:
  EVAL_PERIOD: {eval_period}
  DETECTIONS_PER_IMAGE: {detections_per_image}

OUTPUT_DIR: "{str(output_dir.resolve())}"
VERSION: 2
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    algorithm_name = str(cfg.get("segmentation_algorithm", "")).strip().lower()
    if algorithm_name != "maskdino_official":
        raise ValueError(f"segmentation_algorithm must be maskdino_official, got: {algorithm_name}")

    defaults = _algorithm_defaults()
    repo_root = str(cfg.get("segmentation_train_repo_root") or defaults["repo_root"]).strip()
    conda_sh = str(cfg.get("segmentation_train_conda_sh") or defaults["conda_sh"]).strip()
    conda_env = str(cfg.get("segmentation_train_conda_env") or defaults["conda_env"]).strip()
    train_base_config = str(cfg.get("segmentation_train_base_config") or defaults["train_base_config"]).strip()
    init_checkpoint = str(cfg.get("segmentation_train_init_checkpoint") or defaults["init_checkpoint"]).strip()
    driver_module = str(defaults["driver_module"]).strip()

    output_dir = Path(cfg["output_dir"])
    training_dir = output_dir / cfg.get("segmentation_train_out_dirname", "segmentation_training")
    trainer_root = ensure_dir(training_dir / "external_trainer")
    work_dir = ensure_dir(trainer_root / "work_dir")
    dataset_dir, dataset_summary_json = _ensure_public_dataset(args.config, cfg)
    dataset_summary = load_json(dataset_summary_json)
    injection_manifest = build_training_injection_manifest(
        cfg=cfg,
        dataset_summary=dataset_summary,
        output_dir=training_dir,
    )

    num_train_images = int(dataset_summary["counts"]["train_images"])
    num_val_images = int(dataset_summary["counts"]["val_images"])
    if num_train_images <= 0:
        raise RuntimeError("segmentation public dataset has no train images")
    if num_val_images <= 0:
        raise RuntimeError("segmentation public dataset has no val images")

    class_names_raw = cfg.get("segmentation_class_names", ["crown"])
    if isinstance(class_names_raw, str):
        class_names = [x.strip() for x in class_names_raw.split(",") if x.strip()]
    else:
        class_names = [str(x).strip() for x in class_names_raw if str(x).strip()]
    if not class_names:
        class_names = ["crown"]
    num_classes = int(cfg.get("segmentation_num_classes", len(class_names)))

    batch_size = int(cfg.get("segmentation_train_batch_size", 1))
    num_workers = int(cfg.get("segmentation_train_num_workers", 4))
    epochs = int(cfg.get("segmentation_train_epochs", 8))
    base_lr = float(cfg.get("segmentation_train_lr", 6.25e-6))
    weight_decay = float(cfg.get("segmentation_train_weight_decay", 0.05))
    detections_per_image = int(cfg.get("segmentation_train_detections_per_image", 300))
    device = str(cfg.get("segmentation_train_device") or "cuda")
    num_gpus = int(cfg.get("segmentation_train_num_gpus", 1))
    use_amp = to_bool(cfg.get("segmentation_train_amp"), default=True)
    resume_train = to_bool(cfg.get("segmentation_train_resume"), default=False)
    grad_accum_steps = _grad_accum_steps(cfg)
    val_interval = int(cfg.get("segmentation_train_val_interval", 1))

    iters_per_epoch = max(1, math.ceil(num_train_images / max(batch_size, 1)))
    max_iter = int(cfg.get("segmentation_train_max_iter") or (iters_per_epoch * epochs))
    eval_period = int(cfg.get("segmentation_train_eval_period") or (iters_per_epoch * max(val_interval, 1)))
    checkpoint_period = int(cfg.get("segmentation_train_checkpoint_period") or eval_period)

    train_dataset_name, val_dataset_name, test_dataset_name = _build_dataset_names(algorithm_name)
    generated_config = training_dir / "generated_configs" / "maskdino_public_itd.yaml"
    _write_generated_config(
        out_path=generated_config,
        base_config=train_base_config,
        train_dataset_name=train_dataset_name,
        val_dataset_name=val_dataset_name,
        output_dir=work_dir,
        init_checkpoint=init_checkpoint,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        max_iter=max_iter,
        eval_period=eval_period,
        checkpoint_period=checkpoint_period,
        base_lr=base_lr,
        weight_decay=weight_decay,
        detections_per_image=detections_per_image,
        device=device,
        use_amp=use_amp,
    )

    train_json = dataset_dir / "annotations" / "instances_train.json"
    val_json = dataset_dir / "annotations" / "instances_val.json"
    test_json = dataset_dir / "annotations" / "instances_test.json"
    summary: dict[str, Any] = {
        "status": "config_built" if args.build_only else "pending",
        "segmentation_algorithm": algorithm_name,
        "dataset_dir": str(dataset_dir),
        "dataset_summary_json": str(dataset_summary_json),
        "repo_root": repo_root,
        "conda_sh": conda_sh,
        "conda_env": conda_env,
        "train_base_config": train_base_config,
        "generated_config": str(generated_config),
        "init_checkpoint": init_checkpoint,
        "trainer_output_dir": str(trainer_root),
        "work_dir": str(work_dir),
        "class_names": class_names,
        "num_classes": num_classes,
        "train_batch_size": batch_size,
        "train_num_workers": num_workers,
        "train_epochs": epochs,
        "train_lr": base_lr,
        "train_weight_decay": weight_decay,
        "train_grad_accum_steps": grad_accum_steps,
        "train_effective_batch_size": batch_size * grad_accum_steps,
        "train_amp": use_amp,
        "train_resume": resume_train,
        "driver_module": driver_module,
        "train_dataset_name": train_dataset_name,
        "val_dataset_name": val_dataset_name,
        "test_dataset_name": test_dataset_name,
        "train_json": str(train_json),
        "val_json": str(val_json),
        "test_json": str(test_json),
        "image_root": str(Path(cfg["public_dataset_root"]).resolve()),
        "num_gpus": num_gpus,
        "max_iter": max_iter,
        "val_interval": val_interval,
        "eval_period": eval_period,
        "checkpoint_period": checkpoint_period,
        "expert_injection_manifest": str(injection_manifest.get("manifest_path") or ""),
        "expert_injection": injection_manifest,
    }

    if args.build_only:
        dump_json(summary, training_dir / "train_summary.json")
        print(f"[OK] segmentation MaskDINO training config built only: {generated_config}", flush=True)
        return

    repo_root_path = Path(repo_root).resolve()
    entry_module = "ITD_agent.segmentation.model_training.maskdino_train_entry"
    bash_cmd = (
        f"source {conda_sh} && "
        f"conda activate {conda_env} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export PYTHONUNBUFFERED=1 && "
        f"export MASKDINO_REPO_ROOT={repo_root_path} && "
        f"export PYTHONPATH={PROJECT_ROOT}:{repo_root_path}:${{PYTHONPATH:-}} && "
        f"python -u -m {entry_module} "
        f"--num-gpus {num_gpus} "
        f"--grad-accum-steps {grad_accum_steps} "
        f"{'--resume ' if resume_train else ''}"
        f"--config-file {generated_config} "
        f"--train-json {train_json} "
        f"--val-json {val_json} "
        f"--test-json {test_json} "
        f"--image-root {Path(cfg['public_dataset_root']).resolve()} "
        f"--train-dataset-name {train_dataset_name} "
        f"--val-dataset-name {val_dataset_name} "
        f"--test-dataset-name {test_dataset_name} "
        f"--thing-classes {','.join(class_names)}"
    )
    print("[RUN segmentation MaskDINO trainer]", flush=True)
    print(bash_cmd, flush=True)
    result = subprocess.run(["bash", "-lc", bash_cmd], cwd=str(repo_root_path))

    best_ckpt = _find_best_ckpt(work_dir)
    if result.returncode != 0 and best_ckpt is None:
        raise RuntimeError("segmentation MaskDINO trainer failed and no checkpoint was recovered")
    if best_ckpt is None:
        raise RuntimeError("segmentation MaskDINO trainer finished but no checkpoint was found")

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
            "-u",
            "-m",
            "ITD_agent.segmentation.model_training.test_maskdino_instance",
            "--config",
            args.config,
            "--checkpoint",
            str(best_ckpt),
        ]
        print("[RUN segmentation MaskDINO evaluation]", flush=True)
        print(" ".join(eval_cmd), flush=True)
        eval_result = subprocess.run(eval_cmd)
        summary["eval_returncode"] = int(eval_result.returncode)
        if eval_result.returncode != 0:
            dump_json(summary, training_dir / "train_summary.json")
            raise RuntimeError("segmentation MaskDINO evaluation failed")
    dump_json(summary, training_dir / "train_summary.json")
    print(f"[OK] segmentation MaskDINO training done: {best_ckpt}", flush=True)


if __name__ == "__main__":
    main()
