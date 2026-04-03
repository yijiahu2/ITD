from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from input_layer.contracts import (
    InputManifest,
    ValidationIssue,
    ValidationReport,
)


RASTER_SUFFIXES = {".tif", ".tiff", ".img", ".vrt"}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}
TEXT_SUFFIXES = {".txt", ".md", ".rst"}
RULE_SUFFIXES = {".json", ".yaml", ".yml"}


def _path_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _suffix(path: str | None) -> str:
    return Path(path or "").suffix.lower()


def _add_issue(
    issues: list[ValidationIssue],
    *,
    level: str,
    code: str,
    source_type: str,
    source_id: str,
    message: str,
    path: str | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            level=level,
            code=code,
            source_type=source_type,
            source_id=source_id,
            message=message,
            path=path,
        )
    )


def _load_table_columns(path: str, sheet_name: str | None = None) -> list[str] | None:
    try:
        import pandas as pd
    except Exception:
        return None

    suffix = _suffix(path)
    try:
        if suffix == ".csv":
            return [str(col) for col in pd.read_csv(path, nrows=0).columns]
        if suffix == ".tsv":
            return [str(col) for col in pd.read_csv(path, sep="\t", nrows=0).columns]
        if suffix in {".xlsx", ".xls"}:
            return [str(col) for col in pd.read_excel(path, sheet_name=sheet_name, nrows=0).columns]
        if suffix == ".parquet":
            return [str(col) for col in pd.read_parquet(path).columns]
    except Exception:
        return None
    return None


def _load_vector_columns(path: str) -> list[str] | None:
    try:
        import geopandas as gpd
    except Exception:
        return None
    try:
        gdf = gpd.read_file(path, rows=1)
        return [str(col) for col in gdf.columns]
    except Exception:
        return None


def _validate_required_fields(
    *,
    columns: list[str] | None,
    required_fields: list[str],
    issues: list[ValidationIssue],
    source_type: str,
    source_id: str,
    path: str,
) -> None:
    if columns is None or not required_fields:
        return
    missing = [field for field in required_fields if field not in columns]
    if missing:
        _add_issue(
            issues,
            level="warning",
            code="missing_columns",
            source_type=source_type,
            source_id=source_id,
            message=f"缺少字段: {', '.join(missing)}",
            path=path,
        )


def _validate_mapping_fields(
    *,
    columns: list[str] | None,
    mapping: dict[str, str],
    issues: list[ValidationIssue],
    source_type: str,
    source_id: str,
    path: str,
) -> None:
    if columns is None or not mapping:
        return
    raw_fields = [value for value in mapping.values() if value]
    missing = [field for field in raw_fields if field not in columns]
    if missing:
        _add_issue(
            issues,
            level="warning",
            code="missing_mapping_fields",
            source_type=source_type,
            source_id=source_id,
            message=f"字段映射中缺少原始字段: {', '.join(missing)}",
            path=path,
        )


def _validate_coco_annotation(path: str, issues: list[ValidationIssue], source_id: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        _add_issue(
            issues,
            level="warning",
            code="coco_json_unreadable",
            source_type="public_dataset",
            source_id=source_id,
            message=f"COCO 标注文件无法解析: {exc}",
            path=path,
        )
        return

    for key in ["images", "annotations", "categories"]:
        if key not in payload:
            _add_issue(
                issues,
                level="warning",
                code="coco_missing_key",
                source_type="public_dataset",
                source_id=source_id,
                message=f"COCO 标注缺少顶层键: {key}",
                path=path,
            )


def validate_input_manifest(manifest: InputManifest) -> ValidationReport:
    issues: list[ValidationIssue] = []

    for item in manifest.remote_sensing:
        if not _path_exists(item.path):
            level = "error" if item.required else "warning"
            _add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="remote_sensing",
                source_id=item.id,
                message="遥感影像路径不存在。",
                path=item.path,
            )
        if _suffix(item.path) not in RASTER_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="remote_sensing",
                source_id=item.id,
                message="遥感影像后缀不在常用栅格格式列表中。",
                path=item.path,
            )

    for item in manifest.terrain_dem:
        if not _path_exists(item.path):
            level = "error" if item.required else "warning"
            _add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="terrain_dem",
                source_id=item.id,
                message="DEM 路径不存在。",
                path=item.path,
            )
        if _suffix(item.path) not in RASTER_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="terrain_dem",
                source_id=item.id,
                message="DEM 后缀不在常用栅格格式列表中。",
                path=item.path,
            )

    for item in manifest.survey_tables:
        if not _path_exists(item.path):
            level = "error" if item.required else "warning"
            _add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="survey_table",
                source_id=item.id,
                message="样地调查表路径不存在。",
                path=item.path,
            )
            continue
        if _suffix(item.path) not in TABLE_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="survey_table",
                source_id=item.id,
                message="样地调查表后缀不在常用表格格式列表中。",
                path=item.path,
            )
        columns = _load_table_columns(item.path, sheet_name=item.sheet_name)
        _validate_required_fields(
            columns=columns,
            required_fields=item.key_fields,
            issues=issues,
            source_type="survey_table",
            source_id=item.id,
            path=item.path,
        )
        _validate_mapping_fields(
            columns=columns,
            mapping=item.field_mapping,
            issues=issues,
            source_type="survey_table",
            source_id=item.id,
            path=item.path,
        )

    for item in manifest.industry_vectors:
        if not _path_exists(item.path):
            level = "error" if item.required else "warning"
            _add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="industry_vector",
                source_id=item.id,
                message="行业矢量数据路径不存在。",
                path=item.path,
            )
            continue
        if _suffix(item.path) not in VECTOR_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="industry_vector",
                source_id=item.id,
                message="行业矢量后缀不在常用矢量格式列表中。",
                path=item.path,
            )
        columns = _load_vector_columns(item.path)
        _validate_required_fields(
            columns=columns,
            required_fields=item.key_fields,
            issues=issues,
            source_type="industry_vector",
            source_id=item.id,
            path=item.path,
        )
        _validate_mapping_fields(
            columns=columns,
            mapping=item.field_mapping,
            issues=issues,
            source_type="industry_vector",
            source_id=item.id,
            path=item.path,
        )

    for item in manifest.domain_knowledge_items:
        if not _path_exists(item.path):
            level = "error" if item.required else "warning"
            _add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="domain_knowledge",
                source_id=item.id,
                message="领域知识文件路径不存在。",
                path=item.path,
            )
            continue
        suffix = _suffix(item.path)
        if item.type == "table" and suffix not in TABLE_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="domain_knowledge",
                source_id=item.id,
                message="知识表格后缀不在常用表格格式列表中。",
                path=item.path,
            )
        if item.type == "text" and suffix not in TEXT_SUFFIXES and suffix not in TABLE_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="domain_knowledge",
                source_id=item.id,
                message="知识文本后缀不在常用文本格式列表中。",
                path=item.path,
            )
        if item.type == "rule" and suffix not in RULE_SUFFIXES:
            _add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="domain_knowledge",
                source_id=item.id,
                message="规则知识后缀建议使用 JSON/YAML。",
                path=item.path,
            )

    for item in manifest.public_datasets:
        if item.format == "coco":
            annotation_path = item.annotation_path or item.path
            image_root = item.image_root or item.root
            if not _path_exists(annotation_path):
                level = "error" if item.required else "warning"
                _add_issue(
                    issues,
                    level=level,
                    code="missing_annotation",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="COCO 标注文件不存在。",
                    path=annotation_path,
                )
            else:
                _validate_coco_annotation(annotation_path, issues, item.id)
            if image_root and not Path(image_root).exists():
                level = "error" if item.required else "warning"
                _add_issue(
                    issues,
                    level=level,
                    code="missing_image_root",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="COCO 图像目录不存在。",
                    path=image_root,
                )
        elif item.format == "parquet":
            parquet_path = item.path or item.root
            if not _path_exists(parquet_path):
                level = "error" if item.required else "warning"
                _add_issue(
                    issues,
                    level=level,
                    code="missing_path",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="Parquet 数据集路径不存在。",
                    path=parquet_path,
                )
            elif _suffix(parquet_path) != ".parquet":
                _add_issue(
                    issues,
                    level="warning",
                    code="unexpected_suffix",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="Parquet 数据集建议使用 .parquet 后缀。",
                    path=parquet_path,
                )
        else:
            generic_path = item.path or item.annotation_path or item.root
            if item.required and not _path_exists(generic_path):
                _add_issue(
                    issues,
                    level="error",
                    code="missing_path",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="公开数据集路径不存在。",
                    path=generic_path,
                )

    status = "ok"
    if any(item.level == "error" for item in issues):
        status = "invalid"
    elif issues:
        status = "warning"
    return ValidationReport(status=status, issues=issues)
