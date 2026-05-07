from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.common import as_list, as_string_list, first_non_empty, resolve_path
from input_layer.contracts import IndustryVectorSource


def default_vector_field_mapping(cfg: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "reference_unit_id": cfg.get("reference_id_field") or cfg.get("inventory_id_field") or cfg.get("xiaoban_id_field"),
        "xiaoban_id": cfg.get("xiaoban_id_field"),
        "tree_count": cfg.get("tree_count_field"),
        "crown_width": cfg.get("crown_field"),
        "closure": cfg.get("closure_field"),
        "density": cfg.get("density_field"),
        "area_ha": cfg.get("area_ha_field"),
    }
    return {key: str(value) for key, value in mapping.items() if value}


def parse_vector_sources(
    vector_cfg: dict[str, Any],
    survey_cfg: dict[str, Any],
    inventory_cfg: dict[str, Any],
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[IndustryVectorSource]:
    sources: list[IndustryVectorSource] = []
    default_mapping = default_vector_field_mapping(cfg)
    raw_items = as_list(vector_cfg.get("vectors"))
    fallback_vector = first_non_empty(
        survey_cfg.get("survey_vector"),
        survey_cfg.get("sample_plot_vector"),
        inventory_cfg.get("survey_vector"),
        inventory_cfg.get("sample_plot_vector"),
        cfg.get("reference_vector_path"),
        cfg.get("inventory_vector_path"),
        cfg.get("xiaoban_shp"),
    )
    if fallback_vector and not raw_items:
        raw_items = [fallback_vector]
    elif fallback_vector:
        raw_items = [fallback_vector] + raw_items

    seen_paths: set[str] = set()
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, dict):
            path = resolve_path(item.get("path") or item.get("vector"), config_dir)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            mapping = dict(default_mapping)
            mapping.update({str(k): str(v) for k, v in (item.get("field_mapping") or {}).items() if v})
            sources.append(
                IndustryVectorSource(
                    id=str(item.get("id") or f"vector_{idx:03d}"),
                    path=path,
                    geometry_type=item.get("geometry_type"),
                    crs=item.get("crs"),
                    key_fields=as_string_list(item.get("key_fields")),
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
        path = resolve_path(item, config_dir)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        sources.append(
            IndustryVectorSource(
                id=f"vector_{idx:03d}",
                path=path,
                geometry_type="polygon",
                field_mapping=default_mapping,
            )
        )
    return sources
