from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from input_layer.contracts import (
    InputManifest,
)
from input_layer.common import first_non_empty, resolve_path, safe_float
from input_layer.dom import build_dom_prepared_input_index, compile_dom_input_contract, prepare_dom_runtime_assets
from input_layer.height import parse_dem_sources, parse_height_rasters
from input_layer.mainline_profiles import (
    filter_manifest_sources_for_profile,
    get_mainline_capabilities,
    resolve_mainline_profile,
)
from input_layer.prior_data import parse_prior_data_knowledge_items, parse_prior_data_tables
from input_layer.public_dataset import parse_public_datasets
from input_layer.remote_sensing import parse_remote_sensing_sources
from input_layer.vector import parse_vector_sources
from input_layer.validators import validate_input_manifest


ALLOWED_INPUT_TYPES = {"dom_image", "coco_dataset"}
FORBIDDEN_INPUT_FIELDS = {
    "dem_tif",
    "chm_tif",
    "dsm_tif",
    "xiaoban_shp",
    "terrain",
    "canopy",
    "surface",
    "mainline_profile",
}
_PRODUCT_BOUNDARY_MANIFEST_KEYS = {
    "terrain_dem",
    "canopy_height",
    "surface_models",
    "dem_paths",
    "chm_paths",
    "dsm_paths",
}
_PRODUCT_BOUNDARY_MODALITY_KEYS = {"dem", "chm", "dsm"}


def _config_dir(config_path: str | None) -> Path | None:
    if not config_path:
        return None
    return Path(config_path).expanduser().resolve().parent


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _resolve_output_path(path: Any, config_dir: Path | None) -> str | None:
    return resolve_path(path, config_dir)


def _enforce_minimal_retention(runtime_cfg: dict[str, Any]) -> None:
    runtime_cfg["cleanup_policy"] = "minimal"
    runtime_cfg["keep_debug_outputs"] = False
    runtime_cfg["keep_semantic_prior_artifacts"] = False
    runtime_cfg["cleanup_temp_runtime"] = True


def _iter_forbidden_field_hits(mapping: Any, *, prefix: str = "") -> list[str]:
    if not isinstance(mapping, dict):
        return []
    hits: list[str] = []
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in FORBIDDEN_INPUT_FIELDS:
            hits.append(path)
        if isinstance(value, dict):
            hits.extend(_iter_forbidden_field_hits(value, prefix=path))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                hits.extend(_iter_forbidden_field_hits(item, prefix=f"{path}[{idx}]"))
    return hits


def _validate_supported_input_contract(cfg: dict[str, Any]) -> str:
    input_type = str(
        cfg.get("input_type")
        or (cfg.get("runtime") or {}).get("input_type")
        or (cfg.get("inputs") or {}).get("input_type")
        or ""
    ).strip()
    if not input_type:
        inputs = cfg.get("inputs") or {}
        if inputs.get("public_datasets"):
            input_type = "coco_dataset"
        else:
            input_type = "dom_image"
    if input_type not in ALLOWED_INPUT_TYPES:
        raise ValueError(f"Unsupported input_type={input_type!r}. Allowed values: {sorted(ALLOWED_INPUT_TYPES)}")

    hits = _iter_forbidden_field_hits(cfg.get("inputs") or {})
    if hits:
        raise ValueError("Unsupported input fields for current product boundary: " + ", ".join(sorted(hits)))
    return input_type


def _manifest_to_product_boundary_dict(manifest: InputManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    for key in _PRODUCT_BOUNDARY_MANIFEST_KEYS:
        payload.pop(key, None)
    metadata = dict(payload.get("metadata") or {})
    modalities = dict(metadata.get("input_modalities") or {})
    for key in _PRODUCT_BOUNDARY_MODALITY_KEYS:
        modalities.pop(key, None)
    if modalities:
        metadata["input_modalities"] = modalities
    payload["metadata"] = metadata
    return payload


def _modalities_to_product_boundary_dict(manifest: InputManifest) -> dict[str, bool]:
    modalities = dict(manifest.input_modalities)
    for key in _PRODUCT_BOUNDARY_MODALITY_KEYS:
        modalities.pop(key, None)
    return modalities


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def _map_framework_to_algorithm_name(framework: str | None, model_name: str | None) -> str:
    normalized_framework = str(framework or "").strip().lower()
    normalized_model_name = str(model_name or "").strip().lower()
    framework_map = {
        "mmdetection": {
            "htc": "mmdet_htc",
            "cascade_mask_rcnn": "mmdet_cascade_mask_rcnn",
            "mask_scoring_rcnn": "mmdet_mask_scoring_rcnn",
        },
        "detectron2_mask2former": {
            "mask2former": "mmdet_mask2former",
        },
        "maskdino_detectron2": {
            "maskdino": "maskdino_official",
        },
    }
    if normalized_framework in framework_map and normalized_model_name in framework_map[normalized_framework]:
        return framework_map[normalized_framework][normalized_model_name]
    fallback_by_name = {
        "htc": "mmdet_htc",
        "cascade_mask_rcnn": "mmdet_cascade_mask_rcnn",
        "mask2former": "mmdet_mask2former",
        "maskdino": "maskdino_official",
    }
    return fallback_by_name.get(normalized_model_name, normalized_model_name or "legacy_cellpose_sam")


def _materialize_main_model_runtime_template(template_cfg: dict[str, Any]) -> dict[str, Any]:
    execution_mode = str(template_cfg.get("execution_mode") or "").strip().lower()
    if execution_mode != "real":
        raise ValueError(f"main_model_runtime_config.execution_mode must be 'real', got: {template_cfg.get('execution_mode')!r}")

    runtime_section = dict(template_cfg.get("runtime") or {})
    stage1_section = dict(template_cfg.get("stage1_semantic_prior") or {})
    stage2_section = dict(template_cfg.get("stage2_instance") or {})
    output_section = dict(template_cfg.get("output") or {})
    runtime_overrides: dict[str, Any] = {}
    for source in [runtime_section, stage1_section, stage2_section, output_section]:
        runtime_overrides.update({key: value for key, value in source.items() if value not in (None, "")})

    return {
        "main_model": {
            "model_id": str(template_cfg.get("model_id") or "legacy_cellpose_sam"),
            "execution_mode": "real",
            "segmentation_algorithm": str(template_cfg.get("segmentation_algorithm") or "legacy_cellpose_sam"),
            "runtime_overrides": runtime_overrides,
            "device": runtime_section.get("device"),
        },
        "runtime_updates": {
            key: value
            for key, value in {
                "work_dir": runtime_section.get("work_dir"),
                "conda_env": runtime_section.get("conda_env"),
                "prediction_score_mode": runtime_section.get("prediction_score_mode"),
            }.items()
            if value not in (None, "")
        },
    }


def _materialize_expert_model_template(template_cfg: dict[str, Any]) -> dict[str, Any]:
    expert_model_cfg = dict(template_cfg.get("expert_model") or {})
    input_cfg = dict(template_cfg.get("input") or {})
    inference_cfg = dict(template_cfg.get("inference") or {})
    postprocess_cfg = dict(template_cfg.get("postprocess") or {})
    expert_name = str(expert_model_cfg.get("name") or "")
    segmentation_algorithm = _map_framework_to_algorithm_name(expert_model_cfg.get("framework"), expert_name)
    segmentation_algorithm_cfg = {
        "config_file": expert_model_cfg.get("config_file"),
        "checkpoint": expert_model_cfg.get("checkpoint_file"),
        "device": expert_model_cfg.get("device"),
        "score_thr": inference_cfg.get("instance_score_thr", inference_cfg.get("score_thr")),
        "tile_size": input_cfg.get("tile_size"),
        "tile_overlap": input_cfg.get("tile_overlap"),
        "tile_batch_size": inference_cfg.get("batch_size"),
        "merge_iou_thr": postprocess_cfg.get("merge_tile_iou_thr", inference_cfg.get("nms_iou_thr")),
        "min_area_px": postprocess_cfg.get("min_area_px"),
        "max_instances": inference_cfg.get("max_instances", inference_cfg.get("max_per_img", inference_cfg.get("topk_per_image"))),
        "max_area_px": postprocess_cfg.get("max_area_px"),
        "mask_thr_binary": inference_cfg.get("mask_thr_binary"),
        "object_mask_thr": inference_cfg.get("object_mask_thr"),
        "overlap_thr": inference_cfg.get("overlap_thr"),
        "min_compactness": postprocess_cfg.get("min_compactness"),
        "max_aspect_ratio": postprocess_cfg.get("max_aspect_ratio"),
        "mode": postprocess_cfg.get("mode"),
    }
    return {
        "model_id": expert_name,
        "segmentation_algorithm": segmentation_algorithm,
        "segmentation_algorithm_cfg": {key: value for key, value in segmentation_algorithm_cfg.items() if value not in (None, "")},
    }


def _inject_default_inference_templates(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = deepcopy(cfg)
    main_template = runtime_cfg.get("main_model_runtime_config")
    expert_models_raw = runtime_cfg.get("expert_models")
    expert_templates = None
    if isinstance(expert_models_raw, dict) and "default_templates" in expert_models_raw:
        expert_templates = expert_models_raw.get("default_templates")
    if not isinstance(main_template, dict) and expert_templates is None:
        return runtime_cfg

    materialized: dict[str, Any] = {"main_model": {}, "expert_models": {"execution_mode": "real"}, "model_configs": {}}
    if isinstance(main_template, dict):
        main_payload = _materialize_main_model_runtime_template(main_template)
        runtime_cfg["main_model"] = main_payload["main_model"]
        runtime_block = dict(runtime_cfg.get("runtime") or {})
        runtime_block.update(main_payload["runtime_updates"])
        runtime_cfg["runtime"] = runtime_block
        materialized["main_model"] = dict(main_payload["main_model"])

    expert_models_cfg = dict(runtime_cfg.get("expert_models") or {})
    if expert_models_cfg:
        execution_mode = str(expert_models_cfg.get("execution_mode") or "").strip().lower()
        if execution_mode and execution_mode != "real":
            raise ValueError(f"expert_models.execution_mode must be 'real', got: {expert_models_cfg.get('execution_mode')!r}")
        expert_models_cfg["execution_mode"] = "real"

    model_configs = dict(runtime_cfg.get("model_configs") or {})
    if isinstance(expert_templates, dict):
        if not expert_templates:
            raise ValueError("expert_models.default_templates must define at least one expert template")
        for model_name, template in expert_templates.items():
            if not isinstance(template, dict):
                continue
            materialized_template = _materialize_expert_model_template(template)
            model_configs[str(model_name)] = {
                **model_configs.get(str(model_name), {}),
                **materialized_template,
            }
            materialized["model_configs"][str(model_name)] = dict(materialized_template)
    if model_configs:
        runtime_cfg["model_configs"] = model_configs
    if expert_models_cfg:
        runtime_cfg["expert_models"] = expert_models_cfg
        materialized["expert_models"] = dict(expert_models_cfg)

    runtime_cfg["_default_inference_templates"] = materialized
    return runtime_cfg


def build_input_manifest(
    cfg: dict[str, Any],
    config_path: str | None = None,
) -> InputManifest:
    inputs = cfg.get("inputs") or {}
    config_dir = _config_dir(config_path)
    mainline_profile = resolve_mainline_profile(cfg)

    remote_sensing_cfg = inputs.get("remote_sensing") or {}
    terrain_cfg = inputs.get("terrain") or {}
    canopy_cfg = inputs.get("canopy") or {}
    surface_cfg = inputs.get("surface") or {}
    survey_cfg = inputs.get("survey_data") or {}
    inventory_cfg = inputs.get("inventory") or {}
    vector_cfg = inputs.get("industry_vectors") or {}
    knowledge_cfg = inputs.get("domain_knowledge") or inputs.get("knowledge") or {}
    public_cfg = inputs.get("public_datasets") or {}

    manifest = InputManifest(
        config_path=config_path,
        remote_sensing=parse_remote_sensing_sources(remote_sensing_cfg, cfg, config_dir),
        terrain_dem=parse_dem_sources(terrain_cfg, cfg, config_dir),
        canopy_height=parse_height_rasters(canopy_cfg, cfg, config_dir, role="chm", fallback_keys=("chm", "chm_tif")),
        surface_models=parse_height_rasters(surface_cfg, cfg, config_dir, role="dsm", fallback_keys=("dsm", "dsm_tif")),
        survey_tables=parse_prior_data_tables(survey_cfg, inventory_cfg, cfg, config_dir),
        industry_vectors=parse_vector_sources(vector_cfg, survey_cfg, inventory_cfg, cfg, config_dir),
        domain_knowledge_items=parse_prior_data_knowledge_items(knowledge_cfg, cfg, config_dir),
        public_datasets=parse_public_datasets(public_cfg, cfg, config_dir),
        metadata={
            "schema_version": "itd_input_v1",
            "config_dir": _display_path(config_dir),
            "mainline_profile": mainline_profile,
            "mainline_capabilities": get_mainline_capabilities(mainline_profile),
        },
    )
    if manifest.remote_sensing:
        manifest.dom_input_contract = compile_dom_input_contract(
            source=manifest.remote_sensing[0],
            runtime_cfg={
                "mainline_profile": mainline_profile,
                **cfg,
            },
            config_path=config_path,
        )
        prepared_dom_paths = prepare_dom_runtime_assets(manifest.dom_input_contract)
        if prepared_dom_paths:
            manifest.dom_input_contract.working_dom_path = prepared_dom_paths.get("working_dom_path") or manifest.dom_input_contract.working_dom_path
            manifest.dom_input_contract.valid_mask_path = prepared_dom_paths.get("valid_mask_path") or manifest.dom_input_contract.valid_mask_path
        manifest.metadata["dom_input_contract"] = manifest.dom_input_contract.to_dict()
    manifest.metadata["profile_gate"] = filter_manifest_sources_for_profile(manifest, mainline_profile)
    manifest.validation = validate_input_manifest(manifest)
    manifest.preparation = build_dom_prepared_input_index(manifest.dom_input_contract, cfg, config_path=config_path)
    return manifest


def normalize_agent_runtime_config(
    cfg: dict[str, Any],
    config_path: str | None = None,
) -> tuple[dict[str, Any], InputManifest]:
    runtime_cfg = _inject_default_inference_templates(cfg)
    input_type = _validate_supported_input_contract(runtime_cfg)
    config_dir = _config_dir(config_path)
    mainline_profile = resolve_mainline_profile(runtime_cfg)
    mainline_capabilities = get_mainline_capabilities(mainline_profile)
    manifest = build_input_manifest(runtime_cfg, config_path=config_path)

    runtime_cfg["input_type"] = input_type
    runtime_cfg["mainline_profile"] = mainline_profile
    runtime_cfg["_mainline_capabilities"] = mainline_capabilities

    if "inputs" not in runtime_cfg:
        _enforce_minimal_retention(runtime_cfg)
        runtime_cfg["_input_manifest"] = _manifest_to_product_boundary_dict(manifest)
        runtime_cfg["_input_validation"] = manifest.validation.to_dict() if manifest.validation else None
        runtime_cfg["_prepared_input_index"] = manifest.preparation.to_dict() if manifest.preparation else None
        return runtime_cfg, manifest

    inputs = runtime_cfg.get("inputs") or {}
    core_cfg = runtime_cfg.get("ITD_agent") or runtime_cfg.get("agent") or {}
    segmentation = runtime_cfg.get("segmentation") or {}
    evaluation = runtime_cfg.get("evaluation") or {}
    outputs = runtime_cfg.get("outputs") or {}
    runtime = runtime_cfg.get("runtime") or {}

    remote_sensing = inputs.get("remote_sensing") or {}
    terrain = inputs.get("terrain") or {}
    canopy = inputs.get("canopy") or {}
    surface = inputs.get("surface") or {}
    survey_data = inputs.get("survey_data") or {}
    inventory = inputs.get("inventory") or {}
    industry_vectors = inputs.get("industry_vectors") or {}
    planning = core_cfg.get("planning") or {}
    llm_gateway = core_cfg.get("llm_gateway") or {}
    data_processing = core_cfg.get("data_processing") or {}
    runtime_cache_worker = core_cfg.get("runtime_cache_worker") or {}
    segmentation_models = core_cfg.get("segmentation_models") or core_cfg.get("segmentation_model") or {}
    if isinstance(segmentation_models, dict):
        segmentation_models.setdefault("expert_models", [])
    model_cfg = segmentation.get("model") or {}
    semantic_prior_cfg = (
        data_processing.get("semantic_prior")
        or data_processing.get("semantic_mask")
        or {}
    )
    main_model_cfg = (
        segmentation_models.get("main_model")
        or segmentation_models.get("primary_model")
        or {}
    )
    eval_cfg = evaluation.get("analysis") or {}
    temp_runtime_cfg = outputs.get("temp_runtime") or {}

    first_image = first_non_empty(
        remote_sensing.get("image"),
        remote_sensing.get("rgb_image"),
        manifest.remote_sensing_images[0] if manifest.remote_sensing_images else None,
        runtime_cfg.get("input_image"),
    )
    if first_image:
        runtime_cfg["input_image"] = str(first_image)

    first_dem = first_non_empty(
        terrain.get("dem_tif"),
        manifest.dem_paths[0] if manifest.dem_paths else None,
        runtime_cfg.get("dem_tif"),
    )
    if first_dem and mainline_capabilities.get("allow_dem"):
        runtime_cfg["dem_tif"] = str(first_dem)
    else:
        runtime_cfg.pop("dem_tif", None)

    first_chm = first_non_empty(
        canopy.get("chm_tif"),
        manifest.chm_paths[0] if manifest.chm_paths else None,
        runtime_cfg.get("chm_tif"),
    )
    if first_chm and mainline_capabilities.get("allow_chm"):
        runtime_cfg["chm_tif"] = str(first_chm)
    else:
        runtime_cfg.pop("chm_tif", None)

    first_dsm = first_non_empty(
        surface.get("dsm_tif"),
        manifest.dsm_paths[0] if manifest.dsm_paths else None,
        runtime_cfg.get("dsm_tif"),
    )
    if first_dsm and mainline_capabilities.get("allow_dsm"):
        runtime_cfg["dsm_tif"] = str(first_dsm)
    else:
        runtime_cfg.pop("dsm_tif", None)

    first_vector = first_non_empty(
        survey_data.get("survey_vector"),
        inventory.get("survey_vector"),
        industry_vectors.get("default_vector"),
        manifest.survey_vector,
        runtime_cfg.get("xiaoban_shp"),
    )
    if first_vector and mainline_capabilities.get("allow_inventory"):
        runtime_cfg["xiaoban_shp"] = str(first_vector)
        runtime_cfg["reference_vector_path"] = str(first_vector)
        runtime_cfg["inventory_vector_path"] = str(first_vector)
    else:
        runtime_cfg.pop("xiaoban_shp", None)
        runtime_cfg.pop("reference_vector_path", None)
        runtime_cfg.pop("inventory_vector_path", None)

    runtime_cfg["grouped_inference_enabled"] = bool(
        first_non_empty(
            planning.get("grouped_inference_enabled"),
            (planning.get("grouped_inference") or {}).get("enabled"),
            runtime_cfg.get("grouped_inference_enabled", False),
        )
    )
    runtime_cfg["grouped_inference_use_llm"] = bool(
        first_non_empty(
            (planning.get("grouped_inference") or {}).get("use_llm"),
            llm_gateway.get("enabled"),
            runtime_cfg.get("grouped_inference_use_llm", True),
        )
    )
    if (planning.get("grouped_inference") or {}).get("buffer_m") is not None:
        runtime_cfg["grouped_inference_buffer_m"] = (planning.get("grouped_inference") or {}).get("buffer_m")

    if llm_gateway.get("provider"):
        runtime_cfg["llm_provider"] = llm_gateway.get("provider")
    if llm_gateway.get("model"):
        runtime_cfg["llm_model"] = llm_gateway.get("model")
    if llm_gateway.get("base_url"):
        runtime_cfg["llm_base_url"] = llm_gateway.get("base_url")

    semantic_prior_script = first_non_empty(
        (runtime_cfg.get("main_model") or {}).get("runtime_overrides", {}).get("semantic_prior_script"),
        semantic_prior_cfg.get("script"),
        semantic_prior_cfg.get("runner"),
        model_cfg.get("semantic_prior_script"),
        runtime_cfg.get("semantic_prior_script"),
    )
    if semantic_prior_script:
        runtime_cfg["semantic_prior_script"] = str(semantic_prior_script)

    segmentation_script = first_non_empty(
        (runtime_cfg.get("main_model") or {}).get("runtime_overrides", {}).get("segmentation_script"),
        main_model_cfg.get("script"),
        main_model_cfg.get("runner"),
        model_cfg.get("segmentation_script"),
        runtime_cfg.get("segmentation_script"),
    )
    if segmentation_script:
        runtime_cfg["segmentation_script"] = str(segmentation_script)

    for key in ["segmentation_algorithm", "segmentation_algorithm_module", "semantic_prior_ckpt"]:
        value = first_non_empty(
            (runtime_cfg.get("main_model") or {}).get(key),
            main_model_cfg.get(key),
            (runtime_cfg.get("main_model") or {}).get("runtime_overrides", {}).get(key),
            semantic_prior_cfg.get(key),
            model_cfg.get(key),
            segmentation_models.get(key),
            runtime_cfg.get(key),
        )
        if value is not None:
            runtime_cfg[key] = value

    for key in [
        "diam_list",
        "tile",
        "overlap",
        "tile_overlap",
        "bsize",
        "augment",
        "iou_merge_thr",
    ]:
        value = first_non_empty(segmentation.get(key), runtime_cfg.get(key))
        if value is None:
            value = (runtime_cfg.get("main_model") or {}).get("runtime_overrides", {}).get(key)
        if value is not None:
            runtime_cfg[key] = value

    if runtime_cfg.get("_default_inference_templates"):
        missing_required_template_keys = [
            key
            for key in [
                "semantic_prior_script",
                "segmentation_script",
                "segmentation_algorithm",
                "diam_list",
                "tile",
                "overlap",
                "tile_overlap",
                "bsize",
                "augment",
                "iou_merge_thr",
            ]
            if runtime_cfg.get(key) in (None, "")
        ]
        if missing_required_template_keys:
            raise ValueError(
                "default inference template is missing required runtime fields after normalization: "
                + ", ".join(missing_required_template_keys)
            )

    run_name = str(first_non_empty(runtime_cfg.get("run_name"), runtime.get("run_name"), "agent_system_run"))
    persistent_root = first_non_empty(
        _resolve_output_path(outputs.get("root_dir"), config_dir),
        (
            str(Path(_resolve_output_path(outputs.get("root_base_dir"), config_dir)) / run_name)
            if _resolve_output_path(outputs.get("root_base_dir"), config_dir)
            else None
        ),
        runtime_cfg.get("persistent_output_dir"),
        runtime_cfg.get("output_dir"),
        str(Path("outputs") / run_name),
    )
    persistent_root_path = Path(str(persistent_root))
    use_temp_runtime = bool(first_non_empty(temp_runtime_cfg.get("enabled"), False))
    if use_temp_runtime:
        temp_root = first_non_empty(
            _resolve_output_path(temp_runtime_cfg.get("root_dir"), config_dir),
            str(Path("/tmp") / "itd_agent_runtime"),
        )
        runtime_root_path = Path(str(temp_root)) / run_name
    else:
        runtime_root_path = persistent_root_path

    runtime_cfg["persistent_output_dir"] = str(persistent_root_path)
    runtime_cfg["output_dir"] = str(runtime_root_path)
    runtime_cfg["metrics_json"] = str(runtime_root_path / "evaluation_metrics.json")
    runtime_cfg["details_csv"] = str(runtime_root_path / "evaluation_details.csv")
    runtime_cfg["cleanup_policy"] = str(first_non_empty(outputs.get("cleanup_policy"), runtime_cfg.get("cleanup_policy"), "minimal"))
    runtime_cfg["cleanup_temp_runtime"] = bool(first_non_empty(temp_runtime_cfg.get("cleanup_after_run"), True))
    runtime_cfg["use_temp_runtime"] = use_temp_runtime

    if eval_cfg.get("flat_slope_threshold_deg") is not None:
        runtime_cfg["flat_slope_threshold_deg"] = eval_cfg.get("flat_slope_threshold_deg")
    if eval_cfg.get("plain_relief_threshold_m") is not None:
        runtime_cfg["plain_relief_threshold_m"] = eval_cfg.get("plain_relief_threshold_m")

    for key in [
        "experiment_name",
        "run_name",
        "conda_sh",
        "conda_env",
        "work_dir",
        "use_runtime_cache_worker",
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "density_field",
        "area_ha_field",
    ]:
        value = first_non_empty(runtime.get(key), runtime_cfg.get(key))
        if value is not None:
            runtime_cfg[key] = value

    if runtime_cfg.get("xiaoban_id_field") is not None:
        runtime_cfg["reference_id_field"] = runtime_cfg["xiaoban_id_field"]
        runtime_cfg["inventory_id_field"] = runtime_cfg["xiaoban_id_field"]
    if runtime_cache_worker.get("enabled") is not None and "use_runtime_cache_worker" not in runtime_cfg:
        runtime_cfg["use_runtime_cache_worker"] = bool(runtime_cache_worker.get("enabled"))

    runtime_cfg["_input_manifest"] = _manifest_to_product_boundary_dict(manifest)
    runtime_cfg["_input_profile"] = {
        "mainline_profile": mainline_profile,
        "capabilities": mainline_capabilities,
        "modalities": _modalities_to_product_boundary_dict(manifest),
        "profile_gate": manifest.metadata.get("profile_gate") or {},
    }
    runtime_cfg["_input_validation"] = manifest.validation.to_dict() if manifest.validation else None
    runtime_cfg["_prepared_input_index"] = manifest.preparation.to_dict() if manifest.preparation else None
    runtime_cfg["_dom_input_contract"] = manifest.dom_input_contract.to_dict() if manifest.dom_input_contract else None
    _enforce_minimal_retention(runtime_cfg)
    return runtime_cfg, manifest
