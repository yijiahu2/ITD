from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("train", help="Run controlled training from reviewed sample assets.")
    parser.add_argument("--config", required=True)
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    from ITD_agent.orchestration.workflow import train

    return train(args.config)
