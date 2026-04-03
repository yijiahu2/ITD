from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.finetuning.io_utils import dump_json, dump_yaml, load_json, load_yaml, to_bool


def _base_selected_algorithm_cfg(base_cfg: dict[str, Any], selected_algorithm: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    candidate_cfgs = base_cfg.get("segmentation_candidate_cfgs")
    if isinstance(candidate_cfgs, dict):
        candidate = candidate_cfgs.get(selected_algorithm)
        if isinstance(candidate, dict):
            merged.update(candidate)

    if str(base_cfg.get("segmentation_algorithm", "")).strip().lower() == selected_algorithm:
        inline_cfg = base_cfg.get("segmentation_algorithm_cfg")
        if isinstance(inline_cfg, dict):
            merged.update(inline_cfg)

    return merged


def _has_config_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"]) / "finetuned_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_config": cfg.get("base_config"),
        "ckpt": args.ckpt,
        "can_rerun": False,
        "reason": "",
        "exp_finetuned_yaml": None,
    }

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        summary["reason"] = "训练权重不存在，无法进入分割模型回灌阶段。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] segmentation ckpt not found.")
        return

    if not to_bool(cfg.get("enable_rerun_after_finetune", False), default=False):
        summary["reason"] = "enable_rerun_after_finetune=false；当前仅执行分割模型训练，不回灌 rerun。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] rerun disabled by config.")
        return

    base_cfg_path = cfg.get("base_config")
    if not base_cfg_path:
        summary["reason"] = "缺少 base_config，无法生成 segmentation rerun 配置。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] missing base_config.")
        return

    base_cfg = load_yaml(base_cfg_path)
    train_summary_path = Path(cfg["output_dir"]) / cfg.get("segmentation_train_out_dirname", "segmentation_training") / "train_summary.json"
    if not train_summary_path.exists():
        summary["reason"] = f"未找到 segmentation 训练摘要: {train_summary_path}"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] missing segmentation train_summary.")
        return

    train_summary = load_json(train_summary_path)
    selected_algorithm = str(cfg.get("segmentation_algorithm", "")).strip().lower()
    if not selected_algorithm:
        summary["reason"] = "segmentation_algorithm 为空，无法回灌。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] empty segmentation_algorithm.")
        return

    generated_config = train_summary.get("generated_infer_config") or train_summary.get("generated_config")
    if not generated_config:
        summary["reason"] = "train_summary 中缺少 generated_config，无法回灌。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] missing generated config.")
        return

    generated_config_path = Path(generated_config)
    if not generated_config_path.exists():
        summary["reason"] = f"generated_config 不存在: {generated_config_path}"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] generated config missing.")
        return

    new_cfg = copy.deepcopy(base_cfg)
    suffix = str(cfg.get("finetuned_suffix", f"_{selected_algorithm}_ft_v1"))
    old_run_name = str(base_cfg.get("run_name", "run"))
    new_cfg["run_name"] = old_run_name if old_run_name.endswith(suffix) else old_run_name + suffix

    rerun_out_dir = Path(cfg["output_dir"]) / "rerun_after_finetune"
    rerun_out_dir.mkdir(parents=True, exist_ok=True)
    new_cfg["output_dir"] = str(rerun_out_dir)
    new_cfg["metrics_json"] = str(rerun_out_dir / "metrics.json")
    new_cfg["details_csv"] = str(rerun_out_dir / "details.csv")

    selected_cfg = _base_selected_algorithm_cfg(base_cfg, selected_algorithm)
    selected_cfg["conda_sh"] = str(cfg.get("segmentation_train_conda_sh") or train_summary.get("conda_sh") or selected_cfg.get("conda_sh") or "")
    selected_cfg["conda_env"] = str(cfg.get("segmentation_train_conda_env") or train_summary.get("conda_env") or selected_cfg.get("conda_env") or "")
    selected_cfg["repo_root"] = str(cfg.get("segmentation_train_repo_root") or train_summary.get("repo_root") or selected_cfg.get("repo_root") or "")
    if selected_cfg.get("repo_root") and not selected_cfg.get("cwd"):
        selected_cfg["cwd"] = selected_cfg["repo_root"]
    selected_cfg["config_file"] = str(generated_config_path.resolve())
    selected_cfg["checkpoint"] = str(ckpt_path.resolve())
    selected_cfg["driver_module"] = str(
        train_summary.get("driver_module")
        or selected_cfg.get("driver_module")
        or "ITD_agent.segmentation.model_registry.adapters.mmdet_instance_adapter"
    )

    if cfg.get("segmentation_finetuned_device") not in {None, ""}:
        selected_cfg["device"] = str(cfg.get("segmentation_finetuned_device"))

    override_fields = [
        "score_thr",
        "min_area_px",
        "min_sem_overlap_ratio",
        "clip_to_msem",
        "tile_size",
        "tile_overlap",
        "tile_batch_size",
        "merge_iou_thr",
        "required_outputs",
    ]
    for field in override_fields:
        cfg_key = f"segmentation_finetuned_{field}"
        if cfg_key in cfg and _has_config_value(cfg.get(cfg_key)):
            selected_cfg[field] = cfg[cfg_key]

    new_cfg["segmentation_algorithm"] = selected_algorithm
    new_cfg["segmentation_algorithm_module"] = None
    new_cfg["segmentation_algorithm_cfg"] = selected_cfg

    out_cfg = out_dir / "exp_segmentation_finetuned.yaml"
    dump_yaml(new_cfg, out_cfg)

    summary["can_rerun"] = True
    summary["reason"] = "已生成可用于 segmentation rerun 的 finetuned 配置。"
    summary["exp_finetuned_yaml"] = str(out_cfg)
    summary["rerun_output_dir"] = str(rerun_out_dir)
    summary["rerun_metrics_json"] = new_cfg["metrics_json"]
    summary["rerun_details_csv"] = new_cfg["details_csv"]
    summary["segmentation_algorithm"] = selected_algorithm
    summary["generated_config"] = str(generated_config_path.resolve())

    dump_json(summary, out_dir / "integration_summary.json")
    print(f"[OK] segmentation finetuned config written: {out_cfg}")


if __name__ == "__main__":
    main()
