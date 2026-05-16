from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np

from tools.process_runner import run_streaming
from ITD_agent.segmentation.model_registry.output_utils import build_label_image_from_masks, load_prediction_npz, write_instance_color_png


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONDA_SH = "/home/xth/anaconda3/etc/profile.d/conda.sh"
DEFAULT_DRIVER_BY_FRAMEWORK = {
    "mmdetection": "ITD_agent.segmentation.model_registry.adapters.mmdet_instance_adapter",
    "detectron2_mask2former": "ITD_agent.segmentation.model_registry.adapters.mmdet_instance_adapter",
    "maskdino_detectron2": "ITD_agent.segmentation.model_registry.adapters.maskdino_instance_adapter",
}
DEFAULT_CONDA_ENV_BY_FRAMEWORK = {
    "mmdetection": "mmdetection",
    "detectron2_mask2former": "mmdetection",
    "maskdino_detectron2": "maskdino",
}
DEFAULT_REPO_ROOT_BY_FRAMEWORK = {
    "mmdetection": "/home/xth/mmdetection331",
    "detectron2_mask2former": "/home/xth/mmdetection331",
    "maskdino_detectron2": "/home/xth/MaskDINO",
}
EXPERT_BY_ERROR = {
    "under_segmentation": "htc",
    "over_segmentation": "mask2former",
    "false_positive_cleanup": "cascade_mask_rcnn",
    "missed_crown_recall": "maskdino",
    "boundary_quality": "htc",
}


def _write_json(path: str | Path, payload: Any) -> str:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _require_existing_file(path: Any, desc: str) -> str:
    text = str(path or "").strip()
    if not text:
        raise ValueError(f"{desc} is required for PNG expert real inference")
    resolved = Path(text).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{desc} not found: {resolved}")
    return str(resolved)


def _select_expert_template(cfg: dict[str, Any], dominant_error_type: str) -> tuple[str, dict[str, Any]]:
    expert_cfg = dict(cfg.get("expert_models") or {})
    templates = expert_cfg.get("default_templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("expert_models.default_templates is required for PNG expert real inference")
    routing = ((cfg.get("expert_routing_policy") or {}).get("expert_map") or {})
    route = routing.get(dominant_error_type) if isinstance(routing, dict) else None
    selected = None
    if isinstance(route, dict):
        selected = route.get("primary_expert")
    selected = selected or EXPERT_BY_ERROR.get(dominant_error_type) or next(iter(templates))
    if selected not in templates:
        raise ValueError(f"Routed expert {selected!r} is not configured in expert_models.default_templates")
    return str(selected), dict(templates[selected] or {})


def _run_conda_module(*, conda_sh: str, conda_env: str, module: str, args: list[str], cwd: str) -> None:
    cmd = " ".join([shlex.quote("python"), shlex.quote("-m"), shlex.quote(module), *[shlex.quote(str(item)) for item in args]])
    bash_cmd = (
        f"source {shlex.quote(conda_sh)} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT))}:${{PYTHONPATH:-}} && "
        f"{cmd}"
    )
    result = run_streaming(["bash", "-lc", bash_cmd], cwd=cwd, print_cmd=True, cmd_label="===== PNG EXPERT CMD =====")
    if result.returncode != 0:
        raise RuntimeError(f"PNG expert inference failed: module={module}, returncode={result.returncode}")


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _rle_from_mask(mask: np.ndarray) -> dict[str, Any] | None:
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": rle["counts"]}


def _label_to_coco_instances(label_img: np.ndarray, scores_by_id: dict[int, float], image_id: int | str) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for inst_id in sorted(int(value) for value in np.unique(label_img) if int(value) > 0):
        mask = label_img == inst_id
        area = int(mask.sum())
        record: dict[str, Any] = {
            "image_id": int(image_id) if str(image_id).isdigit() else image_id,
            "category_id": 1,
            "bbox": _bbox_from_mask(mask),
            "area": float(area),
            "score": float(scores_by_id.get(inst_id, 0.0)),
            "instance_id": inst_id,
        }
        rle = _rle_from_mask(mask)
        if rle:
            record["segmentation"] = rle
        instances.append(record)
    return instances


def _materialize_from_npz(
    *,
    npz_path: str | Path,
    semantic_mask_path: str | Path,
    output_dir: str | Path,
    image_id: int | str,
    score_thr: float,
    min_area_px: int,
    min_sem_overlap_ratio: float,
    clip_to_msem: bool,
    max_instances: int | None,
) -> dict[str, Any]:
    semantic_mask = np.load(semantic_mask_path).astype(np.uint8)
    masks, scores = load_prediction_npz(str(npz_path))
    label_img, records = build_label_image_from_masks(
        masks,
        scores,
        semantic_mask,
        score_thr=score_thr,
        min_area_px=min_area_px,
        min_sem_overlap_ratio=min_sem_overlap_ratio,
        clip_to_msem=clip_to_msem,
        max_instances=max_instances,
    )
    out_dir = Path(output_dir).expanduser().resolve()
    label_npy = out_dir / "expert_instance_mask.npy"
    color_png = out_dir / "expert_instance_mask.png"
    np.save(label_npy, label_img.astype(np.int32))
    write_instance_color_png(str(color_png), label_img)
    scores_by_id = {int(item["instance_id"]): float(item["score"]) for item in records}
    instances = _label_to_coco_instances(label_img, scores_by_id, image_id)
    return {
        "instances": instances,
        "instance_mask_npy": str(label_npy),
        "instance_mask_png": str(color_png),
        "kept_instance_count": len(instances),
    }


def run_png_expert_inference(
    *,
    cfg: dict[str, Any],
    selected_images: list[dict[str, Any]],
    main_per_image: dict[str, dict[str, Any]],
    dominant_error_type: str,
    output_dir: str | Path,
    max_expert_rounds: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_name, template = _select_expert_template(cfg, dominant_error_type)
    expert_model = dict(template.get("expert_model") or {})
    inference_cfg = dict(template.get("inference") or {})
    input_cfg = dict(template.get("input") or {})
    postprocess_cfg = dict(template.get("postprocess") or {})
    framework = str(expert_model.get("framework") or "").strip()
    driver_module = str(expert_model.get("driver_module") or DEFAULT_DRIVER_BY_FRAMEWORK.get(framework) or "").strip()
    if not driver_module:
        raise ValueError(f"No PNG expert driver_module configured for framework={framework!r}")
    config_file = _require_existing_file(expert_model.get("config_file"), f"{selected_name}.config_file")
    checkpoint_file = _require_existing_file(expert_model.get("checkpoint_file"), f"{selected_name}.checkpoint_file")
    conda_sh = str(expert_model.get("conda_sh") or DEFAULT_CONDA_SH)
    conda_env = str(expert_model.get("conda_env") or DEFAULT_CONDA_ENV_BY_FRAMEWORK.get(framework) or "").strip()
    repo_root = str(Path(expert_model.get("repo_root") or DEFAULT_REPO_ROOT_BY_FRAMEWORK.get(framework) or PROJECT_ROOT).expanduser().resolve())
    if not conda_env:
        raise ValueError(f"conda_env is required for PNG expert {selected_name}")
    if not Path(conda_sh).exists():
        raise FileNotFoundError(f"conda_sh not found for PNG expert {selected_name}: {conda_sh}")
    if not Path(repo_root).exists():
        raise FileNotFoundError(f"repo_root not found for PNG expert {selected_name}: {repo_root}")

    all_instances: list[dict[str, Any]] = []
    per_image_outputs: dict[str, Any] = {}
    rounds = max(int(max_expert_rounds), 0)
    if rounds <= 0:
        prediction_path = out_dir / "expert_prediction_coco.json"
        _write_json(prediction_path, [])
        selected_path = _write_json(out_dir / "selected_expert.json", {"selected_expert": selected_name, "skipped": "max_expert_rounds<=0"})
        return {"selected_expert_json": selected_path, "expert_prediction_path": str(prediction_path), "instances": []}

    for image in selected_images:
        image_id = str(image.get("id"))
        image_path = _require_existing_file(image.get("path") or image.get("resolved_path"), f"input image for image_id={image_id}")
        main_artifacts = ((main_per_image.get(image_id) or {}).get("artifacts") or {})
        semantic_mask_npy = _require_existing_file(main_artifacts.get("semantic_mask_npy"), f"main semantic mask for image_id={image_id}")
        image_out = out_dir / "per_image" / image_id
        image_out.mkdir(parents=True, exist_ok=True)
        resolved_cfg = {
            "repo_root": repo_root,
            "config_file": config_file,
            "checkpoint": checkpoint_file,
            "device": str(device or expert_model.get("device") or "cuda:0"),
            "score_thr": inference_cfg.get("instance_score_thr", inference_cfg.get("score_thr", 0.25)),
            "tile_size": input_cfg.get("tile_size", 1024),
            "tile_overlap": input_cfg.get("tile_overlap", 0),
            "tile_batch_size": inference_cfg.get("batch_size", 1),
            "merge_iou_thr": postprocess_cfg.get("merge_tile_iou_thr", inference_cfg.get("nms_iou_thr", 0.45)),
        }
        cfg_json = (image_out / "expert_algorithm_cfg.resolved.json").resolve()
        pred_npz = (image_out / "expert_predictions.npz").resolve()
        _write_json(cfg_json, resolved_cfg)
        _run_conda_module(
            conda_sh=conda_sh,
            conda_env=conda_env,
            module=driver_module,
            args=[
                "--config_json",
                str(cfg_json),
                "--input_png",
                image_path,
                "--pred_npz",
                str(pred_npz),
                "--input_image",
                image_path,
                "--msem_tif",
                str(semantic_mask_npy),
                "--output_dir",
                str(image_out.resolve()),
                "--algorithm_name",
                selected_name,
                "--y_inst_tif",
                str((image_out / "unused_y_inst.tif").resolve()),
                "--y_inst_shp",
                str((image_out / "unused_y_inst.shp").resolve()),
                "--y_inst_color_png",
                str((image_out / "unused_y_inst_color.png").resolve()),
            ],
            cwd=repo_root,
        )
        materialized = _materialize_from_npz(
            npz_path=pred_npz,
            semantic_mask_path=semantic_mask_npy,
            output_dir=image_out,
            image_id=image_id,
            score_thr=float(resolved_cfg["score_thr"]),
            min_area_px=int(postprocess_cfg.get("min_area_px") or 20),
            min_sem_overlap_ratio=float(postprocess_cfg.get("min_sem_overlap_ratio", 0.0)),
            clip_to_msem=bool(postprocess_cfg.get("clip_to_msem", True)),
            max_instances=(
                None
                if inference_cfg.get("max_instances", inference_cfg.get("max_per_img", inference_cfg.get("topk_per_image"))) in (None, "", 0)
                else int(inference_cfg.get("max_instances", inference_cfg.get("max_per_img", inference_cfg.get("topk_per_image"))))
            ),
        )
        all_instances.extend(materialized["instances"])
        per_image_outputs[image_id] = materialized

    prediction_path = out_dir / "expert_prediction_coco.json"
    selected_path = out_dir / "selected_expert.json"
    _write_json(prediction_path, all_instances)
    _write_json(
        selected_path,
        {
            "selected_expert": selected_name,
            "dominant_error_type": dominant_error_type,
            "framework": framework,
            "driver_module": driver_module,
            "config_file": config_file,
            "checkpoint_file": checkpoint_file,
            "image_count": len(selected_images),
            "instance_count": len(all_instances),
            "per_image": per_image_outputs,
        },
    )
    return {
        "selected_expert_json": str(selected_path),
        "expert_prediction_path": str(prediction_path),
        "instances": all_instances,
        "per_image": per_image_outputs,
    }
