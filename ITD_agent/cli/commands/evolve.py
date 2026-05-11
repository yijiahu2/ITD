from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from ITD_agent.orchestration.evolution_workflow import run_controlled_evolution


def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("evolve", help="Run controlled self-evolution workflow.")
    parser.add_argument("--config", required=True)
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    return run_controlled_evolution(args.config)
