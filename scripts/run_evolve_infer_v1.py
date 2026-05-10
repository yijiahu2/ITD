from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evolution.evolve_infer_runner import run_evolve_infer_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised COCO main-expert evolution loop V1.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML V1 config.")
    args = parser.parse_args()
    summary = run_evolve_infer_v1(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
