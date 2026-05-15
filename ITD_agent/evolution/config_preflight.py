from __future__ import annotations

from pathlib import Path
from typing import Any

from .adaptive_inference import _inject_real_only_default_templates, _load_structured
from .real_inference_adapter import derive_dataset_input, resolve_image_path_for_coco


def _path_status(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    text = str(path)
    if text.startswith(("http://", "https://")):
        return {"path": text, "exists": None, "remote": True}
    return {"path": text, "exists": Path(text).exists(), "remote": False}


def preflight_runtime_config(config_path: str | Path) -> dict[str, Any]:
    cfg = _inject_real_only_default_templates(_load_structured(config_path))
    input_cfg = derive_dataset_input(cfg.get("input") or {})
    annotation = _path_status(input_cfg.get("annotation_json"))
    image_root = _path_status(input_cfg.get("image_root"))
    base_config = _path_status((cfg.get("runtime") or {}).get("base_config"))

    model_checks: dict[str, Any] = {}
    for model_id, model_cfg in (cfg.get("model_configs") or {}).items():
        algorithm_cfg = model_cfg.get("segmentation_algorithm_cfg") or {}
        model_checks[str(model_id)] = {
            "segmentation_algorithm": model_cfg.get("segmentation_algorithm"),
            "config_file": _path_status(algorithm_cfg.get("config_file")),
            "checkpoint": _path_status(algorithm_cfg.get("checkpoint")),
        }

    failures: list[str] = []
    for label, status in [("annotation_json", annotation), ("image_root", image_root), ("runtime.base_config", base_config)]:
        if status["exists"] is False:
            failures.append(f"{label} not found: {status['path']}")
    for model_id, check in model_checks.items():
        for key in ["config_file", "checkpoint"]:
            status = check[key]
            if status["exists"] is False:
                failures.append(f"model_configs.{model_id}.{key} not found: {status['path']}")

    image_count = 0
    annotation_image_count = 0
    resolvable_annotation_image_count = 0
    if image_root["exists"]:
        image_count = sum(1 for path in Path(str(image_root["path"])).iterdir() if path.is_file())
    if annotation["exists"]:
        payload = _load_structured(str(annotation["path"]))
        images = list(payload.get("images") or [])
        annotation_image_count = len(images)
        for image in images:
            try:
                resolve_image_path_for_coco(image, str(image_root["path"]))
            except FileNotFoundError:
                continue
            resolvable_annotation_image_count += 1

    return {
        "config_path": str(config_path),
        "ok": not failures,
        "failures": failures,
        "input": {
            "annotation_json": annotation,
            "image_root": image_root,
            "image_count": image_count,
            "annotation_image_count": annotation_image_count,
            "resolvable_annotation_image_count": resolvable_annotation_image_count,
            "max_images": input_cfg.get("max_images"),
        },
        "runtime": {"base_config": base_config},
        "main_model": cfg.get("main_model") or {},
        "expert_models": cfg.get("expert_models") or {},
        "model_checks": model_checks,
    }
