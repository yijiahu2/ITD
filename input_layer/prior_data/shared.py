from __future__ import annotations


def default_prior_table_field_mapping(cfg: dict[str, object]) -> dict[str, str]:
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
