from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


RunSegmentationFn = Callable[[dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class SegmentationAlgorithmSpec:
    name: str
    module: str
    description: str
