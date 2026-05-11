from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from ITD_agent.orchestration import workflow


def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("train", help="Run controlled training from reviewed sample assets.")
    parser.add_argument("--config", required=True)
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    return workflow.train(args.config)
