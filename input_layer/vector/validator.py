from __future__ import annotations

from input_layer.common import VECTOR_SUFFIXES, add_issue, load_vector_columns, path_exists, suffix, validate_mapping_fields, validate_required_fields
from input_layer.contracts import IndustryVectorSource, ValidationIssue


def validate_vector_sources(
    items: list[IndustryVectorSource],
    issues: list[ValidationIssue],
) -> None:
    for item in items:
        if not path_exists(item.path):
            level = "error" if item.required else "warning"
            add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="industry_vector",
                source_id=item.id,
                message="矢量数据路径不存在。",
                path=item.path,
            )
            continue
        if suffix(item.path) not in VECTOR_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="industry_vector",
                source_id=item.id,
                message="矢量数据后缀不在常用矢量格式列表中。",
                path=item.path,
            )
        columns = load_vector_columns(item.path)
        validate_required_fields(
            columns=columns,
            required_fields=item.key_fields,
            issues=issues,
            source_type="industry_vector",
            source_id=item.id,
            path=item.path,
        )
        validate_mapping_fields(
            columns=columns,
            mapping=item.field_mapping,
            issues=issues,
            source_type="industry_vector",
            source_id=item.id,
            path=item.path,
        )
