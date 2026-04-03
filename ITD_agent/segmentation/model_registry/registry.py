from __future__ import annotations

import importlib
from typing import Any

from ITD_agent.segmentation.model_registry.base import SegmentationAlgorithmSpec
from ITD_agent.segmentation.model_registry.mmdet_specs import MMDET_ALGORITHM_SPECS


BUILTIN_ALGORITHMS: dict[str, SegmentationAlgorithmSpec] = {
    "template_placeholder": SegmentationAlgorithmSpec(
        name="template_placeholder",
        module="ITD_agent.segmentation.model_registry.runners.template_placeholder",
        description="Template SOTA runner placeholder. Replace with a real algorithm module after deployment.",
    ),
    "maskdino_official": SegmentationAlgorithmSpec(
        name="maskdino_official",
        module="ITD_agent.segmentation.model_registry.runners.maskdino_official",
        description="MaskDINO official implementation external runner.",
    ),
}
for _spec in MMDET_ALGORITHM_SPECS.values():
    BUILTIN_ALGORITHMS[_spec.name] = SegmentationAlgorithmSpec(
        name=_spec.name,
        module=_spec.runner_module,
        description=_spec.description,
    )


def list_segmentation_models() -> list[dict[str, str]]:
    return [
        {
            "name": spec.name,
            "module": spec.module,
            "description": spec.description,
        }
        for spec in BUILTIN_ALGORITHMS.values()
    ]


def _resolve_algorithm_module(cfg: dict[str, Any]):
    algorithm_name = str(cfg.get("segmentation_algorithm", "")).strip().lower()
    if not algorithm_name:
        raise ValueError("segmentation_algorithm is empty")

    custom_module = cfg.get("segmentation_algorithm_module")
    if custom_module:
        return importlib.import_module(str(custom_module))

    spec = BUILTIN_ALGORITHMS.get(algorithm_name)
    if spec is None:
        known = ", ".join(sorted(BUILTIN_ALGORITHMS.keys()))
        raise KeyError(
            f"Unknown segmentation_algorithm: {algorithm_name}. "
            f"Known builtins: [{known}]. "
            "Or set segmentation_algorithm_module to a custom module path."
        )
    return importlib.import_module(spec.module)


def run_segmentation_algorithm(cfg: dict[str, Any], m_sem_tif: str) -> dict[str, Any]:
    mod = _resolve_algorithm_module(cfg)
    if not hasattr(mod, "run"):
        raise AttributeError(f"Segmentation algorithm module has no run(cfg, m_sem_tif): {mod.__name__}")
    return mod.run(cfg, m_sem_tif)
