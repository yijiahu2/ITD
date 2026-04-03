from __future__ import annotations

import argparse
import json
from pathlib import Path

from ITD_agent.segmentation.model_registry.registry import run_segmentation_algorithm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--msem_tif", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    result = run_segmentation_algorithm(cfg, args.msem_tif)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
