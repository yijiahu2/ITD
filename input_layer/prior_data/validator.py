from __future__ import annotations

from input_layer.common import RULE_SUFFIXES, TABLE_SUFFIXES, TEXT_SUFFIXES, add_issue, load_table_columns, path_exists, suffix, validate_mapping_fields, validate_required_fields
from input_layer.contracts import DomainKnowledgeItem, SurveyTableSource, ValidationIssue


def validate_prior_data_tables(items: list[SurveyTableSource], issues: list[ValidationIssue]) -> None:
    for item in items:
        if not path_exists(item.path):
            level = "error" if item.required else "warning"
            add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="survey_table",
                source_id=item.id,
                message="表格先验数据路径不存在。",
                path=item.path,
            )
            continue
        if suffix(item.path) not in TABLE_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="survey_table",
                source_id=item.id,
                message="表格先验数据后缀不在常用表格格式列表中。",
                path=item.path,
            )
        columns = load_table_columns(item.path, sheet_name=item.sheet_name)
        validate_required_fields(
            columns=columns,
            required_fields=item.key_fields,
            issues=issues,
            source_type="survey_table",
            source_id=item.id,
            path=item.path,
        )
        validate_mapping_fields(
            columns=columns,
            mapping=item.field_mapping,
            issues=issues,
            source_type="survey_table",
            source_id=item.id,
            path=item.path,
        )


def validate_prior_data_knowledge_items(
    items: list[DomainKnowledgeItem],
    issues: list[ValidationIssue],
) -> None:
    for item in items:
        if not path_exists(item.path):
            level = "error" if item.required else "warning"
            add_issue(
                issues,
                level=level,
                code="missing_path",
                source_type="domain_knowledge",
                source_id=item.id,
                message="先验知识数据路径不存在。",
                path=item.path,
            )
            continue
        path_suffix = suffix(item.path)
        if item.type == "table" and path_suffix not in TABLE_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="domain_knowledge",
                source_id=item.id,
                message="先验知识表格后缀不在常用表格格式列表中。",
                path=item.path,
            )
        if item.type == "text" and path_suffix not in TEXT_SUFFIXES and path_suffix not in TABLE_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="domain_knowledge",
                source_id=item.id,
                message="先验知识文本后缀不在常用文本格式列表中。",
                path=item.path,
            )
        if item.type == "rule" and path_suffix not in RULE_SUFFIXES:
            add_issue(
                issues,
                level="warning",
                code="unexpected_suffix",
                source_type="domain_knowledge",
                source_id=item.id,
                message="先验规则数据后缀建议使用 JSON/YAML。",
                path=item.path,
            )
