from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from ITD_agent.orchestration import workflow


def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser("state", help="Query workflow SQLite state.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--detail", choices=["summary", "pending", "review-pending", "review-assets"], default="summary")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--review-run-id")
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    return workflow.state(args.db, detail=args.detail, limit=args.limit, review_run_id=args.review_run_id)
