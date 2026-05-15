from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from input_layer.contracts import ValidationIssue


RASTER_SUFFIXES = {".tif", ".tiff", ".img", ".vrt"}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}
TEXT_SUFFIXES = {".txt", ".md", ".rst"}
RULE_SUFFIXES = {".json", ".yaml", ".yml"}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_string_list(value: Any) -> list[str]:
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


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def infer_table_format(path: str | None) -> str | None:
    if not path:
        return None
    path_suffix = Path(path).suffix.lower()
    if path_suffix in {".xlsx", ".xls"}:
        return "excel"
    if path_suffix == ".csv":
        return "csv"
    if path_suffix == ".tsv":
        return "tsv"
    if path_suffix == ".parquet":
        return "parquet"
    return path_suffix.lstrip(".") or None


def infer_knowledge_type(path: str | None, explicit_type: str | None = None) -> str:
    if explicit_type:
        return str(explicit_type)
    path_suffix = Path(path or "").suffix.lower()
    if path_suffix in {".csv", ".xlsx", ".xls", ".parquet"}:
        return "table"
    if path_suffix in {".txt", ".md", ".rst"}:
        return "text"
    if path_suffix in {".json", ".yaml", ".yml"}:
        return "rule"
    return "text"


def infer_dataset_format(path: str | None, explicit_format: str | None = None) -> str:
    if explicit_format:
        return str(explicit_format)
    path_suffix = Path(path or "").suffix.lower()
    if path_suffix == ".parquet":
        return "parquet"
    if path_suffix == ".json":
        return "coco"
    return "unknown"


def resolve_path(path: Any, config_dir: Path | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    resolved = Path(text).expanduser()
    if resolved.is_absolute() or config_dir is None:
        return str(resolved)
    if resolved.parts and resolved.parts[0] in {"configs", "data", "input_layer", "output_layer", "outputs", "runtime_entrypoints", "scripts", "tests", "tools", "ITD_agent", "models"}:
        return str((Path.cwd() / resolved).resolve())
    return str((config_dir / resolved).resolve())


def path_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def suffix(path: str | None) -> str:
    return Path(path or "").suffix.lower()


def load_table_columns(path: str, sheet_name: str | None = None) -> list[str] | None:
    try:
        import pandas as pd
    except Exception:
        return None

    path_suffix = suffix(path)
    try:
        if path_suffix == ".csv":
            return [str(col) for col in pd.read_csv(path, nrows=0).columns]
        if path_suffix == ".tsv":
            return [str(col) for col in pd.read_csv(path, sep="\t", nrows=0).columns]
        if path_suffix in {".xlsx", ".xls"}:
            return [str(col) for col in pd.read_excel(path, sheet_name=sheet_name, nrows=0).columns]
        if path_suffix == ".parquet":
            return [str(col) for col in pd.read_parquet(path).columns]
    except Exception:
        return None
    return None


def load_vector_columns(path: str) -> list[str] | None:
    try:
        import geopandas as gpd
    except Exception:
        return None
    try:
        gdf = gpd.read_file(path, rows=1)
        return [str(col) for col in gdf.columns]
    except Exception:
        return None


def load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def add_issue(
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


def validate_required_fields(
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
        add_issue(
            issues,
            level="warning",
            code="missing_columns",
            source_type=source_type,
            source_id=source_id,
            message=f"缺少字段: {', '.join(missing)}",
            path=path,
        )


def validate_mapping_fields(
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
        add_issue(
            issues,
            level="warning",
            code="missing_mapping_fields",
            source_type=source_type,
            source_id=source_id,
            message=f"字段映射中缺少原始字段: {', '.join(missing)}",
            path=path,
        )


def validate_coco_annotation(path: str, issues: list[ValidationIssue], source_id: str) -> None:
    payload = load_json(path)
    if payload is None:
        add_issue(
            issues,
            level="warning",
            code="coco_json_unreadable",
            source_type="public_dataset",
            source_id=source_id,
            message="COCO 标注文件无法解析。",
            path=path,
        )
        return
    for key in ["images", "annotations", "categories"]:
        if key not in payload:
            add_issue(
                issues,
                level="warning",
                code="coco_missing_key",
                source_type="public_dataset",
                source_id=source_id,
                message=f"COCO 标注缺少顶层键: {key}",
                path=path,
            )
