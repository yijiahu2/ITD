from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.segmentation.model_registry import list_segmentation_models


def main() -> None:
    models = list_segmentation_models()
    print(json.dumps({"segmentation_models": models, "algorithms": models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
