from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.contracts import (
    InputManifest,
    PreparedAsset,
    PreparedInputIndex,
)


def _sanitize_name(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "item"


def _workspace_root(cfg: dict[str, Any], config_path: str | None = None) -> Path:
    output_dir = cfg.get("output_dir")
    if output_dir:
        return Path(str(output_dir)).expanduser().resolve()

    outputs = cfg.get("outputs") or {}
    root_dir = outputs.get("root_dir")
    if root_dir:
        return Path(str(root_dir)).expanduser().resolve()

    runtime = cfg.get("runtime") or {}
    run_name = runtime.get("run_name") or cfg.get("run_name") or "itd_agent_run"
    project_root = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
    return (project_root / "outputs" / str(run_name)).resolve()


def derive_input_workspace(cfg: dict[str, Any], config_path: str | None = None) -> dict[str, str]:
    root = _workspace_root(cfg, config_path=config_path)
    registry_root = root / "input_registry"
    prepared_root = root / "prepared_inputs"
    return {
        "workspace_root": str(root),
        "registry_root": str(registry_root),
        "prepared_root": str(prepared_root),
    }


def _prepared_path(prepared_root: Path, category: str, source_id: str, raw_path: str | None) -> str:
    suffix = Path(raw_path or "").suffix
    return str(prepared_root / category / f"{_sanitize_name(source_id)}{suffix}")


def build_prepared_input_index(
    manifest: InputManifest,
    cfg: dict[str, Any],
    config_path: str | None = None,
) -> PreparedInputIndex:
    workspace = derive_input_workspace(cfg, config_path=config_path)
    registry_root = Path(workspace["registry_root"])
    prepared_root = Path(workspace["prepared_root"])
    assets: list[PreparedAsset] = []

    for item in manifest.remote_sensing:
        assets.append(
            PreparedAsset(
                source_type="remote_sensing",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "remote_sensing", item.id, item.path),
                registry_key=f"remote_sensing/{item.id}",
                preparation_actions=[
                    "validate_readability",
                    "reproject_if_needed",
                    "align_to_reference_grid",
                    "extract_image_quality_features",
                ],
                notes=["作为高分辨率遥感影像主输入。"],
            )
        )

    for item in manifest.terrain_dem:
        assets.append(
            PreparedAsset(
                source_type="terrain_dem",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "terrain_dem", item.id, item.path),
                registry_key=f"terrain_dem/{item.id}",
                preparation_actions=[
                    "validate_readability",
                    "reproject_if_needed",
                    "align_to_reference_grid",
                    "derive_slope_aspect_landform",
                ],
                notes=["为数据处理模块提供地形先验。"],
            )
        )

    for item in manifest.canopy_height:
        assets.append(
            PreparedAsset(
                source_type="canopy_height",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "canopy_height", item.id, item.path),
                registry_key=f"canopy_height/{item.id}",
                preparation_actions=[
                    "validate_readability",
                    "reproject_if_needed",
                    "align_to_reference_grid",
                    "extract_height_distribution",
                    "extract_local_peaks",
                ],
                notes=["作为冠层高度先验输入。"],
            )
        )

    for item in manifest.surface_models:
        assets.append(
            PreparedAsset(
                source_type="surface_model",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "surface_models", item.id, item.path),
                registry_key=f"surface_models/{item.id}",
                preparation_actions=[
                    "validate_readability",
                    "reproject_if_needed",
                    "align_to_reference_grid",
                    "derive_surface_statistics",
                ],
                notes=["作为 DSM 或表面高程辅助输入。"],
            )
        )

    for item in manifest.survey_tables:
        assets.append(
            PreparedAsset(
                source_type="survey_table",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "survey_tables", item.id, item.path),
                registry_key=f"survey_tables/{item.id}",
                preparation_actions=[
                    "normalize_encoding",
                    "standardize_field_mapping",
                    "export_tabular_index",
                ],
                notes=["用于样地调查属性接入和结果比对。"],
            )
        )

    for item in manifest.industry_vectors:
        assets.append(
            PreparedAsset(
                source_type="industry_vector",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "industry_vectors", item.id, item.path),
                registry_key=f"industry_vectors/{item.id}",
                preparation_actions=[
                    "reproject_if_needed",
                    "standardize_field_mapping",
                    "build_spatial_index",
                ],
                notes=["用于小班边界、行业属性和 ROI 约束。"],
            )
        )

    for item in manifest.domain_knowledge_items:
        assets.append(
            PreparedAsset(
                source_type="domain_knowledge",
                source_id=item.id,
                raw_path=item.path,
                prepared_path=_prepared_path(prepared_root, "domain_knowledge", item.id, item.path),
                registry_key=f"domain_knowledge/{item.id}",
                preparation_actions=[
                    "classify_knowledge_type",
                    "extract_structured_rules",
                    "build_retrieval_index",
                ],
                notes=["用于先验知识嵌入和规划调度提示上下文。"],
            )
        )

    for item in manifest.public_datasets:
        raw_path = item.path or item.annotation_path or item.root or item.image_root
        prepared_path = prepared_root / "public_datasets" / _sanitize_name(item.id)
        if item.format == "parquet" and raw_path:
            prepared_path = Path(_prepared_path(prepared_root, "public_datasets", item.id, raw_path))
        assets.append(
            PreparedAsset(
                source_type="public_dataset",
                source_id=item.id,
                raw_path=raw_path,
                prepared_path=str(prepared_path),
                registry_key=f"public_datasets/{item.id}",
                preparation_actions=[
                    "validate_dataset_schema",
                    "normalize_dataset_layout",
                    "prepare_finetune_candidates",
                ],
                notes=[f"公开数据集格式: {item.format}"],
            )
        )

    return PreparedInputIndex(
        registry_root=str(registry_root),
        prepared_root=str(prepared_root),
        assets=assets,
        metadata={
            "workspace": workspace,
            "asset_counts": {
                "remote_sensing": len(manifest.remote_sensing),
                "terrain_dem": len(manifest.terrain_dem),
                "canopy_height": len(manifest.canopy_height),
                "surface_models": len(manifest.surface_models),
                "survey_tables": len(manifest.survey_tables),
                "industry_vectors": len(manifest.industry_vectors),
                "domain_knowledge": len(manifest.domain_knowledge_items),
                "public_datasets": len(manifest.public_datasets),
            },
        },
    )
