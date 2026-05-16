from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("evolve", help="Run controlled self-evolution workflow.")
    parser.add_argument("--config", required=True)
    parser.set_defaults(handler=handle)
    infer_parser: ArgumentParser = subparsers.add_parser("evolve-infer", help="Run DOM-only COCO evolve-infer workflow.")
    infer_parser.add_argument("--config", required=True)
    infer_parser.set_defaults(handler=handle_infer)


def handle(args: Namespace) -> dict[str, Any]:
    from ITD_agent.orchestration.evolution_workflow import run_controlled_evolution

    return run_controlled_evolution(args.config)


def handle_infer(args: Namespace) -> dict[str, Any]:
    from ITD_agent.orchestration.workflow import evolve_infer

    return evolve_infer(args.config)
