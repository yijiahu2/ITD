from __future__ import annotations

from dataclasses import asdict
from typing import Any


class DataclassDictMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
