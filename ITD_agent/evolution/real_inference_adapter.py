from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from ITD_agent.segmentation.coco_utils import build_image_index, resolve_image_path
from ITD_agent.segmentation.instance_label_io import instances_from_label_image
from ITD_agent.segmentation.semantic_prior_outputs import write_semantic_prior_outputs
from tools.cached_stage_runners import predict_semantic_prior_cached, run_segmentation_cached


def load_yaml_or_json(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"PyYAML is required to load YAML config: {src}") from exc
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def derive_dataset_input(input_cfg: dict[str, Any]) -> dict[str, Any]:
    if input_cfg.get("annotation_json") and input_cfg.get("image_root"):
        return dict(input_cfg)

    dataset_root = Path(str(input_cfg["dataset_root"])).expanduser()
    split_name = str(input_cfg.get("split") or "validation")
    split_mapping = input_cfg.get("split_mapping") or {
        "train": "Dataset_4_train",
        "train_4": "Dataset_4_train",
        "validation": "Validation_set",
        "val": "Validation_set",
    }
    split_dir = dataset_root / str(split_mapping.get(split_name, split_name))
    annotation_dir = split_dir / str(input_cfg.get("annotation_dirname") or "annotation")
    json_files = sorted(annotation_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No COCO annotation json found under {annotation_dir}")
    return {
        **input_cfg,
        "annotation_json": str(json_files[0]),
        "image_root": str(split_dir / str(input_cfg.get("image_dirname") or "images")),
        "resolved_split_dir": str(split_dir),
    }


def resolve_image_path_for_coco(image: dict[str, Any], image_root: str | Path) -> Path:
    image_dir = Path(image_root)
    by_name, by_stem = build_image_index(image_dir)
    return resolve_image_path(str(image["file_name"]), by_name, by_stem, image_dir)


def _read_label_image(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int32)


def _instances_from_segmentation_result(
    *,
    image_id: int,
    y_inst_tif: str,
    score_map: np.ndarray | None,
    score_mode: str,
) -> list[dict[str, Any]]:
    instances = instances_from_label_image(
        label_image=_read_label_image(y_inst_tif),
        image_id=image_id,
        score_map=score_map,
        score_mode=score_mode,
    )
    normalized: list[dict[str, Any]] = []
    for item in instances:
        normalized.append(
            {
                "id": item.get("pred_id"),
                "pred_id": item.get("pred_id"),
                "image_id": image_id,
                "bbox": item.get("bbox"),
                "area": item.get("area"),
                "score": item.get("score", 1.0),
            }
        )
    return normalized


def build_runtime_cfg(
    *,
    base_config_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    model_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = load_yaml_or_json(base_config_path)
    cfg = {**base, **(model_cfg or {}).get("runtime_overrides", {})}
    cfg["input_image"] = str(image_path)
    cfg["output_dir"] = str(output_dir)
    if "segmentation_algorithm" in (model_cfg or {}):
        cfg["segmentation_algorithm"] = str(model_cfg["segmentation_algorithm"])
    if "segmentation_algorithm_cfg" in (model_cfg or {}):
        cfg["segmentation_algorithm_cfg"] = dict(model_cfg["segmentation_algorithm_cfg"] or {})
    if "segmentation_candidate_cfgs" in (model_cfg or {}):
        cfg["segmentation_candidate_cfgs"] = dict(model_cfg["segmentation_candidate_cfgs"] or {})
    return cfg


def run_real_segmentation_for_sample(
    *,
    base_config_path: str | Path,
    image: dict[str, Any],
    image_path: str | Path,
    output_dir: str | Path,
    model_cfg: dict[str, Any] | None = None,
    score_mode: str = "semantic_prior_mean_prob",
) -> dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    runtime_cfg = build_runtime_cfg(
        base_config_path=base_config_path,
        image_path=image_path,
        output_dir=output_dir,
        model_cfg=model_cfg,
    )
    semantic_prior_pred = predict_semantic_prior_cached(runtime_cfg)
    semantic_outputs = write_semantic_prior_outputs(
        semantic_prior_pred,
        Path(output_dir),
        save_prob_tif=bool((model_cfg or {}).get("save_semantic_prior_probability_tif", False)),
    )
    segmentation_info = run_segmentation_cached(runtime_cfg, semantic_outputs["m_sem_tif"])
    instances = _instances_from_segmentation_result(
        image_id=int(image["id"]),
        y_inst_tif=segmentation_info["y_inst_tif"],
        score_map=semantic_prior_pred["probability"],
        score_mode=score_mode,
    )
    return {
        "status": "completed",
        "model_id": (model_cfg or {}).get("model_id") or runtime_cfg.get("segmentation_algorithm") or "legacy_cellpose_sam",
        "runtime_cfg": {
            "base_config_path": str(base_config_path),
            "segmentation_algorithm": runtime_cfg.get("segmentation_algorithm", "legacy_cellpose_sam"),
            "output_dir": str(output_dir),
        },
        "artifacts": {
            **semantic_outputs,
            **segmentation_info,
        },
        "instances": instances,
    }
