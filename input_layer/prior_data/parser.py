from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.common import as_list, as_string_list, first_non_empty, infer_knowledge_type, infer_table_format, resolve_path
from input_layer.contracts import DomainKnowledgeItem, SurveyTableSource
from input_layer.prior_data.shared import default_prior_table_field_mapping


def parse_prior_data_tables(
    survey_cfg: dict[str, Any],
    inventory_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[SurveyTableSource]:
    sources: list[SurveyTableSource] = []
    default_mapping = default_prior_table_field_mapping(cfg)
    raw_items = as_list(first_non_empty(survey_cfg.get("tables"), inventory_cfg.get("tables")))
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = resolve_path(item.get("path"), config_dir)
            if not path:
                continue
            mapping = dict(default_mapping)
            mapping.update({str(k): str(v) for k, v in (item.get("field_mapping") or {}).items() if v})
            sources.append(
                SurveyTableSource(
                    id=str(item.get("id") or f"prior_table_{idx:03d}"),
                    path=path,
                    format=item.get("format") or infer_table_format(path),
                    sheet_name=item.get("sheet_name"),
                    key_fields=as_string_list(item.get("key_fields")),
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
        path = resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(
            SurveyTableSource(
                id=f"prior_table_{idx:03d}",
                path=path,
                format=infer_table_format(path),
                field_mapping=default_mapping,
            )
        )
    return sources


def parse_prior_data_knowledge_items(
    knowledge_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[DomainKnowledgeItem]:
    sources: list[DomainKnowledgeItem] = []
    raw_items = as_list(first_non_empty(knowledge_cfg.get("items"), knowledge_cfg.get("sources"), cfg.get("domain_knowledge")))
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = resolve_path(item.get("path"), config_dir)
            if not path:
                continue
            sources.append(
                DomainKnowledgeItem(
                    id=str(item.get("id") or f"prior_knowledge_{idx:03d}"),
                    type=infer_knowledge_type(path, explicit_type=item.get("type")),
                    path=path,
                    title=item.get("title"),
                    sheet_name=item.get("sheet_name"),
                    tags=as_string_list(item.get("tags")),
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "type", "path", "title", "sheet_name", "tags", "required"}
                    },
                )
            )
            continue
        path = resolve_path(item, config_dir)
        if not path:
            continue
        sources.append(
            DomainKnowledgeItem(
                id=f"prior_knowledge_{idx:03d}",
                type=infer_knowledge_type(path),
                path=path,
            )
        )
    return sources
