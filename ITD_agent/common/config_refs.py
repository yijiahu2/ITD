from __future__ import annotations

from typing import Any


def reference_vector_path(cfg: dict[str, Any]) -> str | None:
    return (
        cfg.get("reference_vector_path")
        or cfg.get("inventory_vector_path")
        or cfg.get("xiaoban_shp")
    )


def reference_id_field(cfg: dict[str, Any]) -> str | None:
    return (
        cfg.get("reference_id_field")
        or cfg.get("inventory_id_field")
        or cfg.get("xiaoban_id_field")
    )
