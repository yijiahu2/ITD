from __future__ import annotations

from pathlib import Path

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import KnowledgeProfile


def _normalize_type(path: str, raw_type: str, tags: list[str]) -> tuple[str, str, str | None]:
    suffix = Path(path).suffix.lower()
    lower_tags = {str(tag).lower() for tag in tags}
    if suffix in {".tif", ".tiff", ".vrt", ".img"}:
        return "raster_prior", "roi_or_tile_summary", "coarse_raster"
    if raw_type == "table":
        return "tabular_prior", "scene_lookup", "table"
    if raw_type == "rule":
        return "rule_knowledge", "planning_constraint", "rule"
    if "strategy" in lower_tags or "failure" in lower_tags or "经验" in "".join(tags):
        return "strategy_knowledge", "planning_hint", "text"
    return "text_knowledge", "retrieval_context", "text"


def build_knowledge_profiles(manifest: InputManifest) -> list[KnowledgeProfile]:
    profiles: list[KnowledgeProfile] = []
    for item in manifest.domain_knowledge_items:
        normalized_type, use_scope, spatial_scale = _normalize_type(item.path, item.type, item.tags)
        profiles.append(
            KnowledgeProfile(
                source_id=item.id,
                path=item.path,
                raw_type=item.type,
                normalized_type=normalized_type,
                use_scope=use_scope,
                spatial_scale=spatial_scale,
                tags=item.tags,
                extraction_summary={
                    "title": item.title,
                    "sheet_name": item.sheet_name,
                    "path_exists": Path(item.path).exists(),
                },
                metadata=item.metadata,
            )
        )
    return profiles
