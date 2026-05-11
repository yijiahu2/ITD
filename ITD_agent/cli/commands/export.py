from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from ITD_agent.orchestration import workflow


def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("export", help="Export publishable artifacts from a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    return workflow.export(args.run_dir, args.out)
