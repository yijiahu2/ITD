from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_data_processing_dirs(runtime_cfg: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(runtime_cfg["output_dir"]).resolve()
    root = output_dir / "data_processing"

    def _build_layout(base_root: Path) -> dict[str, str]:
        return {
            "root": str(base_root),
            "input_profiles": str(base_root / "input_profiles"),
            "raster_cache": str(base_root / "raster_cache"),
            "terrain_cache": str(base_root / "terrain_cache"),
            "vector_cache": str(base_root / "vector_cache"),
            "knowledge_cache": str(base_root / "knowledge_cache"),
            "public_dataset_index": str(base_root / "public_dataset_index"),
            "roi_cache": str(base_root / "roi_cache"),
            "fusion_cache": str(base_root / "fusion_cache"),
            "requests": str(base_root / "requests"),
            "summaries": str(base_root / "summaries"),
        }

    layout = _build_layout(root)
    try:
        for path in layout.values():
            Path(path).mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback_root = Path("/tmp") / "itd_agent_data_processing" / str(runtime_cfg.get("run_name") or "default_run")
        layout = _build_layout(fallback_root)
        for path in layout.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        layout["requested_root"] = str(root)
        layout["fallback_root"] = str(fallback_root)
    return layout


def write_json(payload: dict[str, Any], path: str | Path) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(out_path)
