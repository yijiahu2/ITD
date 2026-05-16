from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from input_layer.adapters import normalize_agent_runtime_config
from output_layer.contracts import FinalTreeCrownResult
from output_layer.publisher import publish_final_tree_crown_outputs

from ITD_agent.evaluation_analysis.coco_instance_evaluator import evaluate_coco_instances
from ITD_agent.segmentation.coco_utils import build_image_index, resolve_image_path
from ITD_agent.segmentation.png_expert_executor import run_png_expert_inference
from ITD_agent.segmentation.png_fusion import fuse_coco_predictions
from runtime_entrypoints.zstreeseg_png_main_model import infer_one_image


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _write_json(path: str | Path, payload: Any) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _write_yaml(path: str | Path, payload: Any) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(out_path)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _with_cli_overrides(
    cfg: dict[str, Any],
    *,
    dataset_root: str | Path,
    image_root: str | Path,
    annotation: str | Path,
    output_dir: str | Path,
    run_name: str,
    split: str,
    max_images: int | None,
    image_ids: list[str] | None,
    image_names: list[str] | None,
    max_expert_rounds: int,
    device: str | None,
) -> dict[str, Any]:
    runtime_cfg = deepcopy(cfg)
    runtime_cfg["input_type"] = "coco_dataset"
    runtime_cfg["output_dir"] = str(output_dir)
    runtime_cfg["run_name"] = run_name
    input_cfg = dict(runtime_cfg.get("input") or {})
    input_cfg.update(
        {
            "dataset_root": str(dataset_root),
            "image_root": str(image_root),
            "annotation_json": str(annotation),
            "split": split,
        }
    )
    if max_images is not None:
        input_cfg["max_images"] = int(max_images)
    if image_ids:
        input_cfg["image_ids"] = [str(item) for item in image_ids]
    if image_names:
        input_cfg["image_names"] = [str(item) for item in image_names]
    runtime_cfg["input"] = input_cfg
    runtime_cfg["inputs"] = {
        "public_datasets": {
            "datasets": [
                {
                    "id": run_name,
                    "format": "coco",
                    "root": str(dataset_root),
                    "image_root": str(image_root),
                    "annotation_path": str(annotation),
                    "required": True,
                }
            ]
        }
    }
    adaptive_cfg = dict(runtime_cfg.get("adaptive_inference") or {})
    adaptive_cfg["max_expert_rounds"] = int(max_expert_rounds)
    runtime_cfg["adaptive_inference"] = adaptive_cfg
    if device:
        main_cfg = dict(runtime_cfg.get("main_model_runtime_config") or {})
        main_runtime = dict(main_cfg.get("runtime") or {})
        main_runtime["device"] = device
        main_cfg["runtime"] = main_runtime
        runtime_cfg["main_model_runtime_config"] = main_cfg
        expert_cfg = dict(runtime_cfg.get("expert_models") or {})
        templates = dict(expert_cfg.get("default_templates") or {})
        for name, template in templates.items():
            template_cfg = dict(template or {})
            expert_model = dict(template_cfg.get("expert_model") or {})
            expert_model["device"] = device
            template_cfg["expert_model"] = expert_model
            templates[name] = template_cfg
        expert_cfg["default_templates"] = templates
        runtime_cfg["expert_models"] = expert_cfg
    return runtime_cfg


def _select_images(coco: dict[str, Any], image_root: str | Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    by_name, by_stem = build_image_index(image_root)
    candidates = [dict(image) for image in coco.get("images") or []]
    image_ids = cfg.get("image_ids")
    if image_ids:
        allowed = {str(item) for item in image_ids}
        candidates = [image for image in candidates if str(image.get("id")) in allowed]
    image_names = cfg.get("image_names")
    if image_names:
        allowed_names = {str(item) for item in image_names}
        candidates = [image for image in candidates if str(image.get("file_name")) in allowed_names or Path(str(image.get("file_name"))).name in allowed_names]
    max_images = cfg.get("max_images")
    limit = None if max_images in (None, "", 0) else int(max_images)
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for image in candidates:
        try:
            image_path = resolve_image_path(str(image.get("file_name")), by_name, by_stem, image_root)
        except FileNotFoundError as exc:
            unresolved.append({"id": image.get("id"), "file_name": image.get("file_name"), "error": str(exc)})
            continue
        resolved.append({**image, "path": str(image_path), "resolved_path": str(image_path)})
        if limit is not None and len(resolved) >= limit:
            break
    if unresolved:
        cfg["_unresolved_image_count"] = len(unresolved)
        cfg["_unresolved_image_examples"] = unresolved[:20]
    return resolved


def _run_main_model(
    *,
    selected_images: list[dict[str, Any]],
    cfg: dict[str, Any],
    working_dir: Path,
    device: str | None,
) -> dict[str, Any]:
    main_dir = working_dir / "main_model"
    per_image_dir = main_dir / "per_image"
    main_dir.mkdir(parents=True, exist_ok=True)
    main_template = dict(cfg.get("main_model_runtime_config") or {})
    all_instances: list[dict[str, Any]] = []
    per_image: dict[str, dict[str, Any]] = {}
    total = len(selected_images)
    print(f"[coco-png-infer] main_model start images={total}", flush=True)
    for index, image in enumerate(selected_images, start=1):
        image_id = str(image.get("id"))
        print(
            f"[coco-png-infer] main_model image {index}/{total} "
            f"image_id={image_id} file={Path(str(image.get('file_name'))).name}",
            flush=True,
        )
        result = infer_one_image(
            image_path=str(image["path"]),
            output_dir=per_image_dir / image_id,
            image_id=image_id,
            model_cfg=main_template,
            device=device,
        )
        per_image[image_id] = result
        all_instances.extend(result.get("instances") or [])
        print(
            f"[coco-png-infer] main_model done {index}/{total} "
            f"image_id={image_id} instances={len(result.get('instances') or [])}",
            flush=True,
        )
    prediction_path = main_dir / "main_prediction_coco.json"
    _write_json(prediction_path, all_instances)
    print(f"[coco-png-infer] main_model completed instances={len(all_instances)}", flush=True)
    return {"prediction_path": str(prediction_path), "instances": all_instances, "per_image": per_image}


def _inject_coco_area_profile(cfg: dict[str, Any], coco: dict[str, Any], selected_images: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_cfg = deepcopy(cfg)
    selected_ids = {str(image.get("id")) for image in selected_images}
    areas = [
        float(ann.get("area") or 0.0)
        for ann in coco.get("annotations") or []
        if str(ann.get("image_id")) in selected_ids and float(ann.get("area") or 0.0) > 0.0
    ]
    if not areas:
        return runtime_cfg
    areas_sorted = sorted(areas)

    def percentile(p: float) -> float:
        if not areas_sorted:
            return 0.0
        idx = min(len(areas_sorted) - 1, max(0, int(round((len(areas_sorted) - 1) * p))))
        return float(areas_sorted[idx])

    min_area = max(1, int(percentile(0.05) * 0.35))
    max_area = max(min_area, int(percentile(0.95) * 2.5))
    main_template = dict(runtime_cfg.get("main_model_runtime_config") or {})
    postprocess = dict(main_template.get("postprocess") or {})
    if postprocess.get("min_area_px") in {"auto_from_coco_area_profile", "auto", None, ""}:
        postprocess["min_area_px"] = min_area
    if postprocess.get("max_area_px") in {"auto_from_coco_area_profile", "auto", None, ""}:
        postprocess["max_area_px"] = max_area
    main_template["postprocess"] = postprocess
    runtime_cfg["main_model_runtime_config"] = main_template
    runtime_cfg["_coco_area_profile"] = {
        "annotation_count": len(areas),
        "min_area_px": min_area,
        "max_area_px": max_area,
        "p05_area_px": percentile(0.05),
        "p50_area_px": percentile(0.5),
        "p95_area_px": percentile(0.95),
    }
    return runtime_cfg


def _publish_outputs(
    *,
    run_name: str,
    output_dir: Path,
    annotation_path: str | Path,
    fused_prediction_path: str,
    fused_metrics: dict[str, Any],
    selected_images: list[dict[str, Any]],
    coco: dict[str, Any],
    instances: list[dict[str, Any]],
    main_result: dict[str, Any],
    expert_result: dict[str, Any],
) -> dict[str, Any]:
    width = int(selected_images[0].get("width") or 0) if selected_images else None
    height = int(selected_images[0].get("height") or 0) if selected_images else None
    instance_mask_paths: list[str] = []
    expert_per_image = dict(expert_result.get("per_image") or {})
    main_per_image = dict(main_result.get("per_image") or {})
    for image in selected_images:
        image_id = str(image.get("id"))
        expert_mask = (expert_per_image.get(image_id) or {}).get("instance_mask_png")
        main_mask = ((main_per_image.get(image_id) or {}).get("artifacts") or {}).get("instance_mask_png")
        selected_mask = expert_mask or main_mask
        if selected_mask:
            instance_mask_paths.append(str(selected_mask))
    result = FinalTreeCrownResult(
        run_id=run_name,
        output_dir=str(output_dir),
        input_type="coco_dataset",
        has_gt=True,
        input_dom_path=str(selected_images[0].get("path")) if selected_images else None,
        instances=instances,
        coco_predictions_path=fused_prediction_path,
        instance_mask_paths=instance_mask_paths,
        coordinate_mode="pixel",
        image_width=width,
        image_height=height,
        categories=list(coco.get("categories") or [{"id": 1, "name": "crown", "supercategory": "crown"}]),
        gt_metrics=fused_metrics,
        visualization_config={
            "max_sample_overlays": min(len(selected_images), 20),
            "max_instance_masks": min(len(selected_images), 20),
            "save_error_examples": True,
        },
        metadata={
            "source_adapter": "coco_png_infer",
            "output_type": "coco_gt",
            "dataset_type": "coco_instance_segmentation_with_gt",
            "image_root": str(Path(str(selected_images[0].get("path"))).parent) if selected_images else None,
            "annotation_path": str(annotation_path),
            "coco_images": selected_images,
            "images": selected_images,
            "dataset_name": run_name,
            "prediction_path": fused_prediction_path,
        },
    )
    return publish_final_tree_crown_outputs(result=result, publish_root=output_dir)


def run_coco_png_pipeline(
    *,
    template: str | Path,
    dataset_root: str | Path,
    image_root: str | Path,
    annotation: str | Path,
    output_dir: str | Path,
    run_name: str,
    split: str = "validation",
    max_images: int | None = None,
    image_ids: list[str] | None = None,
    image_names: list[str] | None = None,
    max_expert_rounds: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    working_dir = out_dir / "working"
    evaluation_dir = working_dir / "evaluation"
    working_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print(f"[coco-png-infer] load template={template}", flush=True)
    raw_cfg = _load_yaml(template)
    merged_cfg = _with_cli_overrides(
        raw_cfg,
        dataset_root=dataset_root,
        image_root=image_root,
        annotation=annotation,
        output_dir=output_dir,
        run_name=run_name,
        split=split,
        max_images=max_images,
        image_ids=image_ids,
        image_names=image_names,
        max_expert_rounds=max_expert_rounds,
        device=device,
    )
    print("[coco-png-infer] normalize runtime config", flush=True)
    normalized_cfg, manifest = normalize_agent_runtime_config(merged_cfg, config_path=str(template))
    resolved_config_path = _write_yaml(working_dir / "resolved_input_config.yaml", normalized_cfg)

    annotation_path = Path(annotation).expanduser()
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation not found: {annotation_path}")
    print(f"[coco-png-infer] load annotation={annotation_path}", flush=True)
    coco = _load_json(annotation_path)
    input_cfg = dict(normalized_cfg.get("input") or {})
    selected_images = _select_images(coco, image_root, input_cfg)
    if not selected_images:
        raise ValueError("No COCO images selected for inference")
    print(f"[coco-png-infer] selected images={len(selected_images)}", flush=True)
    normalized_cfg = _inject_coco_area_profile(normalized_cfg, coco, selected_images)
    resolved_config_path = _write_yaml(working_dir / "resolved_input_config.yaml", normalized_cfg)
    selected_images_path = _write_json(
        working_dir / "selected_images.json",
        {
            "selected": selected_images,
            "unresolved_image_count": input_cfg.get("_unresolved_image_count", 0),
            "unresolved_image_examples": input_cfg.get("_unresolved_image_examples", []),
        },
    )

    main_result = _run_main_model(selected_images=selected_images, cfg=normalized_cfg, working_dir=working_dir, device=device)
    selected_ids = [image.get("id") for image in selected_images]
    matching_cfg = dict((normalized_cfg.get("evaluation") or {}).get("matching") or {})
    print("[coco-png-infer] evaluate main_model predictions", flush=True)
    main_metrics = evaluate_coco_instances(
        annotation_path=annotation_path,
        prediction_path=main_result["prediction_path"],
        output_path=evaluation_dir / "main_metrics.json",
        image_ids=selected_ids,
        iou_threshold=float(matching_cfg.get("iou_threshold", 0.5)),
        weak_overlap_threshold=float(matching_cfg.get("weak_overlap_threshold", 0.1)),
    )
    dominant_error_type = str(main_metrics.get("dominant_error_type") or "boundary_quality")

    expert_result: dict[str, Any] = {"instances": [], "expert_prediction_path": None}
    if int(max_expert_rounds) > 0:
        print(
            f"[coco-png-infer] expert_model start rounds={int(max_expert_rounds)} "
            f"dominant_error_type={dominant_error_type}",
            flush=True,
        )
        expert_result = run_png_expert_inference(
            cfg=normalized_cfg,
            selected_images=selected_images,
            main_per_image=main_result["per_image"],
            dominant_error_type=dominant_error_type,
            output_dir=working_dir / "expert_model",
            max_expert_rounds=max_expert_rounds,
            device=device,
        )
        print(
            f"[coco-png-infer] expert_model completed "
            f"instances={len(expert_result.get('instances') or [])}",
            flush=True,
        )
    else:
        print("[coco-png-infer] expert_model skipped rounds=0", flush=True)
    print("[coco-png-infer] fusion start", flush=True)
    fusion_result = fuse_coco_predictions(
        main_prediction_path=main_result["prediction_path"],
        expert_prediction_path=expert_result.get("expert_prediction_path"),
        dominant_error_type=dominant_error_type,
        output_dir=working_dir / "fusion",
    )
    fused_instances = _load_json(fusion_result["fused_prediction_path"])
    print(f"[coco-png-infer] fusion completed instances={len(fused_instances)}", flush=True)
    print("[coco-png-infer] evaluate fused predictions", flush=True)
    fused_metrics = evaluate_coco_instances(
        annotation_path=annotation_path,
        prediction_path=fusion_result["fused_prediction_path"],
        output_path=evaluation_dir / "fused_metrics.json",
        image_ids=selected_ids,
        iou_threshold=float(matching_cfg.get("iou_threshold", 0.5)),
        weak_overlap_threshold=float(matching_cfg.get("weak_overlap_threshold", 0.1)),
    )
    print("[coco-png-infer] publish final outputs", flush=True)
    published = _publish_outputs(
        run_name=run_name,
        output_dir=out_dir,
        annotation_path=annotation_path,
        fused_prediction_path=fusion_result["fused_prediction_path"],
        fused_metrics=fused_metrics,
        selected_images=selected_images,
        coco=coco,
        instances=fused_instances,
        main_result=main_result,
        expert_result=expert_result,
    )
    final_summary = {
        "status": "completed",
        "run_name": run_name,
        "input_type": "coco_dataset",
        "resolved_input_config": resolved_config_path,
        "selected_images_json": selected_images_path,
        "main_prediction_coco": main_result["prediction_path"],
        "main_metrics_json": str(evaluation_dir / "main_metrics.json"),
        "selected_expert_json": expert_result.get("selected_expert_json"),
        "expert_prediction_coco": expert_result.get("expert_prediction_path"),
        "fused_prediction_coco": fusion_result["fused_prediction_path"],
        "fused_metrics_json": str(evaluation_dir / "fused_metrics.json"),
        "output_layer": published,
        "input_manifest": normalized_cfg.get("_input_manifest") or manifest.to_dict(),
    }
    _write_json(out_dir / "final_summary.json", final_summary)
    print(f"[coco-png-infer] completed output_dir={out_dir}", flush=True)
    return final_summary
