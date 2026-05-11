from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from ITD_agent.orchestration import workflow


def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("run", help="Run the full ITD_agent workflow.")
    parser.add_argument("--config", required=True)
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    return workflow.run(args.config)
