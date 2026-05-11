from __future__ import annotations

import argparse
from typing import Any

from ITD_agent.orchestration.workflow import run


def run_ITD_agent_experiment(config_path: str) -> dict[str, Any]:
    return run(config_path)["result"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_ITD_agent_experiment(args.config)


if __name__ == "__main__":
    main()
