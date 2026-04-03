from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from input_layer.contracts import (
    DEMSource,
    DatasetSource,
    DomainKnowledgeItem,
    IndustryVectorSource,
    InputManifest,
    PublicDatasetSource,
    RemoteSensingImageSource,
    SurveyTableSource,
)
from input_layer.preparers import build_prepared_input_index
from input_layer.validators import validate_input_manifest


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        if value.strip():
            return [value.strip()]
    return [str(value).strip()]


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _infer_table_format(path: str | None) -> str | None:
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".parquet":
        return "parquet"
    return suffix.lstrip(".") or None


def _infer_knowledge_type(path: str | None, explicit_type: str | None = None) -> str:
    if explicit_type:
        return str(explicit_type)
    suffix = Path(path or "").suffix.lower()
    if suffix in {".csv", ".xlsx", ".xls", ".parquet"}:
        return "table"
    if suffix in {".txt", ".md", ".rst"}:
        return "text"
    if suffix in {".json", ".yaml", ".yml"}:
        return "rule"
    return "text"


def _infer_dataset_format(path: str | None, explicit_format: str | None = None) -> str:
    if explicit_format:
        return str(explicit_format)
    suffix = Path(path or "").suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".json":
        return "coco"
    return "unknown"


def _resolve_path(path: Any, config_dir: Path | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    p = Path(text).expanduser()
    if p.is_absolute() or config_dir is None:
        return str(p)
    return str((config_dir / p).resolve())


def _config_dir(config_path: str | None) -> Path | None:
    if not config_path:
        return None
    return Path(config_path).expanduser().resolve().parent


def _resolve_output_path(path: Any, config_dir: Path | None) -> str | None:
    return _resolve_path(path, config_dir)


def _default_inventory_field_mapping(cfg: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "xiaoban_id": cfg.get("xiaoban_id_field"),
        "tree_count": cfg.get("tree_count_field"),
        "crown_width": cfg.get("crown_field"),
        "closure": cfg.get("closure_field"),
        "density": cfg.get("density_field"),
        "area_ha": cfg.get("area_ha_field"),
    }
    return {key: str(value) for key, value in mapping.items() if value}


def _parse_remote_sensing_sources(
    remote_sensing_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[RemoteSensingImageSource]:
    sources: list[RemoteSensingImageSource] = []
    seen_paths: set[str] = set()
    raw_items = _as_list(remote_sensing_cfg.get("images"))
    first_image = _first_non_empty(
        remote_sensing_cfg.get("image"),
        remote_sensing_cfg.get("rgb_image"),
        cfg.get("input_image"),
    )
    if first_image and not raw_items:
        raw_items = [first_image]
    elif first_image:
        raw_items = [first_image] + raw_items

    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = _resolve_path(item.get("path") or item.get("image"), config_dir)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            sources.append(
                RemoteSensingImageSource(
                    id=str(item.get("id") or f"image_{idx:03d}"),
                    path=path,
                    sensor=item.get("sensor"),
                    resolution_m=_safe_float(item.get("resolution_m")),
                    crs=item.get("crs"),
                    bands=_as_string_list(item.get("bands")),
                    nodata=item.get("nodata"),
                    acquired_at=item.get("acquired_at"),
                    required=bool(item.get("required", True)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k
                        not in {
                            "id",
                            "path",
                            "image",
                            "sensor",
                            "resolution_m",
                            "crs",
                            "bands",
                            "nodata",
                            "acquired_at",
                            "required",
                        }
                    },
                )
            )
            continue
        path = _resolve_path(item, config_dir)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        sources.append(
            RemoteSensingImageSource(
                id=f"image_{idx:03d}",
                path=path,
            )
        )
    return sources


def _parse_dem_sources(
    terrain_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[DEMSource]:
    sources: list[DEMSource] = []
    raw_dem = terrain_cfg.get("dem")
    raw_items = _as_list(raw_dem if isinstance(raw_dem, list) else [raw_dem] if isinstance(raw_dem, dict) else raw_dem)
    single_dem = _first_non_empty(terrain_cfg.get("dem_tif"), cfg.get("dem_tif"))
    if single_dem and not raw_items:
        raw_items = [single_dem]
    elif single_dem and all(not isinstance(item, str) or item != single_dem for item in raw_items):
        raw_items = [single_dem] + raw_items

    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = _resolve_path(item.get("path") or item.get("dem"), config_dir)
            if not path:
                continue
            sources.append(
                DEMSource(
                    id=str(item.get("id") or f"dem_{idx:03d}"),
                    path=path,
                    resolution_m=_safe_float(item.get("resolution_m")),
                    crs=item.get("crs"),
                    vertical_unit=item.get("vertical_unit"),
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "path", "dem", "resolution_m", "crs", "vertical_unit", "required"}
                    },
                )
            )
            continue
        path = _resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(
            DEMSource(
                id=f"dem_{idx:03d}",
                path=path,
            )
        )
    return sources


def _parse_survey_tables(
    survey_cfg: dict[str, Any],
    inventory_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[SurveyTableSource]:
    sources: list[SurveyTableSource] = []
    default_mapping = _default_inventory_field_mapping(cfg)
    raw_items = _as_list(_first_non_empty(survey_cfg.get("tables"), inventory_cfg.get("tables")))
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = _resolve_path(item.get("path"), config_dir)
            if not path:
                continue
            mapping = dict(default_mapping)
            mapping.update({str(k): str(v) for k, v in (item.get("field_mapping") or {}).items() if v})
            sources.append(
                SurveyTableSource(
                    id=str(item.get("id") or f"survey_table_{idx:03d}"),
                    path=path,
                    format=item.get("format") or _infer_table_format(path),
                    sheet_name=item.get("sheet_name"),
                    key_fields=_as_string_list(item.get("key_fields")),
                    field_mapping=mapping,
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "path", "format", "sheet_name", "key_fields", "field_mapping", "required"}
                    },
                )
            )
            continue
        path = _resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(
            SurveyTableSource(
                id=f"survey_table_{idx:03d}",
                path=path,
                format=_infer_table_format(path),
                field_mapping=default_mapping,
            )
        )
    return sources


def _parse_industry_vectors(
    vector_cfg: dict[str, Any],
    survey_cfg: dict[str, Any],
    inventory_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[IndustryVectorSource]:
    sources: list[IndustryVectorSource] = []
    default_mapping = _default_inventory_field_mapping(cfg)
    raw_items = _as_list(vector_cfg.get("vectors"))
    fallback_vector = _first_non_empty(
        survey_cfg.get("survey_vector"),
        survey_cfg.get("sample_plot_vector"),
        inventory_cfg.get("survey_vector"),
        inventory_cfg.get("sample_plot_vector"),
        cfg.get("xiaoban_shp"),
    )
    if fallback_vector and not raw_items:
        raw_items = [fallback_vector]
    elif fallback_vector:
        raw_items = [fallback_vector] + raw_items

    seen_paths: set[str] = set()
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = _resolve_path(item.get("path") or item.get("vector"), config_dir)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            mapping = dict(default_mapping)
            mapping.update({str(k): str(v) for k, v in (item.get("field_mapping") or {}).items() if v})
            sources.append(
                IndustryVectorSource(
                    id=str(item.get("id") or f"industry_vector_{idx:03d}"),
                    path=path,
                    geometry_type=item.get("geometry_type"),
                    crs=item.get("crs"),
                    key_fields=_as_string_list(item.get("key_fields")),
                    field_mapping=mapping,
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "path", "vector", "geometry_type", "crs", "key_fields", "field_mapping", "required"}
                    },
                )
            )
            continue
        path = _resolve_path(item, config_dir)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        sources.append(
            IndustryVectorSource(
                id=f"industry_vector_{idx:03d}",
                path=path,
                geometry_type="polygon",
                field_mapping=default_mapping,
            )
        )
    return sources


def _parse_domain_knowledge(
    knowledge_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[DomainKnowledgeItem]:
    sources: list[DomainKnowledgeItem] = []
    raw_items = _as_list(_first_non_empty(knowledge_cfg.get("items"), knowledge_cfg.get("sources"), cfg.get("domain_knowledge")))
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = _resolve_path(item.get("path"), config_dir)
            if not path:
                continue
            sources.append(
                DomainKnowledgeItem(
                    id=str(item.get("id") or f"knowledge_{idx:03d}"),
                    type=_infer_knowledge_type(path, explicit_type=item.get("type")),
                    path=path,
                    title=item.get("title"),
                    sheet_name=item.get("sheet_name"),
                    tags=_as_string_list(item.get("tags")),
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "type", "path", "title", "sheet_name", "tags", "required"}
                    },
                )
            )
            continue
        path = _resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(
            DomainKnowledgeItem(
                id=f"knowledge_{idx:03d}",
                type=_infer_knowledge_type(path),
                path=path,
            )
        )
    return sources


def _normalize_dataset_sources(raw_items: Any, config_dir: Path | None) -> list[DatasetSource]:
    sources: list[DatasetSource] = []
    for idx, item in enumerate(_as_list(raw_items), 1):
        if isinstance(item, PublicDatasetSource):
            sources.append(item)
            continue
        if isinstance(item, str):
            path = _resolve_path(item, config_dir)
            if not path:
                continue
            dataset_format = _infer_dataset_format(path)
            if dataset_format == "coco":
                sources.append(
                    DatasetSource(
                        id=f"dataset_{idx:03d}",
                        format=dataset_format,
                        annotation_path=path,
                        root=str(Path(path).parent),
                    )
                )
            else:
                sources.append(
                    DatasetSource(
                        id=f"dataset_{idx:03d}",
                        format=dataset_format,
                        path=path,
                    )
                )
            continue
        if isinstance(item, dict):
            raw_path = _resolve_path(item.get("path"), config_dir)
            root = _resolve_path(item.get("root"), config_dir)
            image_root = _resolve_path(item.get("image_root"), config_dir)
            annotation_path = _resolve_path(item.get("annotation_path"), config_dir)
            dataset_format = _infer_dataset_format(annotation_path or raw_path, explicit_format=item.get("format"))
            sources.append(
                DatasetSource(
                    id=str(item.get("id") or item.get("name") or f"dataset_{idx:03d}"),
                    format=dataset_format,
                    path=raw_path,
                    root=root,
                    image_root=image_root,
                    annotation_path=annotation_path,
                    schema_mapping={str(k): str(v) for k, v in (item.get("schema_mapping") or {}).items() if v},
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "name", "format", "path", "root", "image_root", "annotation_path", "schema_mapping", "required"}
                    },
                )
            )
    return sources


def _parse_public_datasets(
    public_cfg: Any,
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[DatasetSource]:
    raw_items = public_cfg
    if isinstance(public_cfg, dict):
        raw_items = public_cfg.get("datasets")
    if raw_items is None:
        raw_items = cfg.get("public_datasets")
    return _normalize_dataset_sources(raw_items, config_dir)


def build_input_manifest(
    cfg: dict[str, Any],
    config_path: str | None = None,
) -> InputManifest:
    inputs = cfg.get("inputs") or {}
    config_dir = _config_dir(config_path)

    remote_sensing_cfg = inputs.get("remote_sensing") or {}
    terrain_cfg = inputs.get("terrain") or {}
    survey_cfg = inputs.get("survey_data") or {}
    inventory_cfg = inputs.get("inventory") or {}
    vector_cfg = inputs.get("industry_vectors") or {}
    knowledge_cfg = inputs.get("domain_knowledge") or inputs.get("knowledge") or {}
    public_cfg = inputs.get("public_datasets") or {}

    manifest = InputManifest(
        config_path=config_path,
        remote_sensing=_parse_remote_sensing_sources(remote_sensing_cfg, cfg, config_dir),
        terrain_dem=_parse_dem_sources(terrain_cfg, cfg, config_dir),
        survey_tables=_parse_survey_tables(survey_cfg, inventory_cfg, cfg, config_dir),
        industry_vectors=_parse_industry_vectors(vector_cfg, survey_cfg, inventory_cfg, cfg, config_dir),
        domain_knowledge_items=_parse_domain_knowledge(knowledge_cfg, cfg, config_dir),
        public_datasets=_parse_public_datasets(public_cfg, cfg, config_dir),
        metadata={
            "schema_version": "itd_input_v1",
            "config_dir": str(config_dir) if config_dir else None,
        },
    )
    manifest.validation = validate_input_manifest(manifest)
    manifest.preparation = build_prepared_input_index(manifest, cfg, config_path=config_path)
    return manifest


def normalize_agent_runtime_config(
    cfg: dict[str, Any],
    config_path: str | None = None,
) -> tuple[dict[str, Any], InputManifest]:
    runtime_cfg = deepcopy(cfg)
    config_dir = _config_dir(config_path)
    manifest = build_input_manifest(runtime_cfg, config_path=config_path)

    if "inputs" not in runtime_cfg:
        runtime_cfg["_input_manifest"] = manifest.to_dict()
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
    survey_data = inputs.get("survey_data") or {}
    inventory = inputs.get("inventory") or {}
    industry_vectors = inputs.get("industry_vectors") or {}
    planning = core_cfg.get("planning") or {}
    llm_gateway = core_cfg.get("llm_gateway") or {}
    data_processing = core_cfg.get("data_processing") or {}
    segmentation_models = core_cfg.get("segmentation_models") or core_cfg.get("segmentation_model") or {}
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

    first_image = _first_non_empty(
        remote_sensing.get("image"),
        remote_sensing.get("rgb_image"),
        manifest.remote_sensing_images[0] if manifest.remote_sensing_images else None,
        runtime_cfg.get("input_image"),
    )
    if first_image:
        runtime_cfg["input_image"] = str(first_image)

    first_dem = _first_non_empty(
        terrain.get("dem_tif"),
        manifest.dem_paths[0] if manifest.dem_paths else None,
        runtime_cfg.get("dem_tif"),
    )
    if first_dem:
        runtime_cfg["dem_tif"] = str(first_dem)

    first_vector = _first_non_empty(
        survey_data.get("survey_vector"),
        inventory.get("survey_vector"),
        industry_vectors.get("default_vector"),
        manifest.survey_vector,
        runtime_cfg.get("xiaoban_shp"),
    )
    if first_vector:
        runtime_cfg["xiaoban_shp"] = str(first_vector)

    runtime_cfg["grouped_inference_enabled"] = bool(
        _first_non_empty(
            planning.get("grouped_inference_enabled"),
            (planning.get("grouped_inference") or {}).get("enabled"),
            runtime_cfg.get("grouped_inference_enabled", False),
        )
    )
    runtime_cfg["grouped_inference_use_llm"] = bool(
        _first_non_empty(
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

    semantic_prior_script = _first_non_empty(
        semantic_prior_cfg.get("script"),
        semantic_prior_cfg.get("runner"),
        model_cfg.get("semantic_prior_script"),
        runtime_cfg.get("semantic_prior_script"),
    )
    if semantic_prior_script:
        runtime_cfg["semantic_prior_script"] = str(semantic_prior_script)

    segmentation_script = _first_non_empty(
        main_model_cfg.get("script"),
        main_model_cfg.get("runner"),
        model_cfg.get("segmentation_script"),
        runtime_cfg.get("segmentation_script"),
    )
    if segmentation_script:
        runtime_cfg["segmentation_script"] = str(segmentation_script)

    for key in ["segmentation_algorithm", "segmentation_algorithm_module", "semantic_prior_ckpt"]:
        value = _first_non_empty(
            main_model_cfg.get(key),
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
        value = _first_non_empty(segmentation.get(key), runtime_cfg.get(key))
        if value is not None:
            runtime_cfg[key] = value

    run_name = str(_first_non_empty(runtime_cfg.get("run_name"), runtime.get("run_name"), "agent_system_run"))
    persistent_root = _first_non_empty(
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
    use_temp_runtime = bool(_first_non_empty(temp_runtime_cfg.get("enabled"), False))
    if use_temp_runtime:
        temp_root = _first_non_empty(
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
    runtime_cfg["cleanup_policy"] = str(_first_non_empty(outputs.get("cleanup_policy"), runtime_cfg.get("cleanup_policy"), "standard"))
    runtime_cfg["cleanup_temp_runtime"] = bool(_first_non_empty(temp_runtime_cfg.get("cleanup_after_run"), True))
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
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "density_field",
        "area_ha_field",
    ]:
        value = _first_non_empty(runtime.get(key), runtime_cfg.get(key))
        if value is not None:
            runtime_cfg[key] = value

    runtime_cfg["_input_manifest"] = manifest.to_dict()
    runtime_cfg["_input_validation"] = manifest.validation.to_dict() if manifest.validation else None
    runtime_cfg["_prepared_input_index"] = manifest.preparation.to_dict() if manifest.preparation else None
    return runtime_cfg, manifest
