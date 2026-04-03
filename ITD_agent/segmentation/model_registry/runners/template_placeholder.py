from __future__ import annotations

from pathlib import Path
from typing import Any


def run(cfg: dict[str, Any], m_sem_tif: str) -> dict[str, Any]:
    algorithm_name = str(cfg.get("segmentation_algorithm", "template_placeholder"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_outputs = {
        "y_inst_tif": str(output_dir / "Y_inst.tif"),
        "y_inst_shp": str(output_dir / "Y_inst.shp"),
        "y_inst_color_png": str(output_dir / "Y_inst_color.png"),
    }

    extra_cfg = cfg.get("segmentation_algorithm_cfg", {})
    raise NotImplementedError(
        "Segmentation algorithm template placeholder is selected but no real algorithm has been deployed. "
        f"algorithm={algorithm_name}, m_sem_tif={m_sem_tif}, "
        f"expected_outputs={expected_outputs}, segmentation_algorithm_cfg={extra_cfg}. "
        "Implement a module with run(cfg, m_sem_tif) -> dict and point segmentation_algorithm_module to it."
    )
