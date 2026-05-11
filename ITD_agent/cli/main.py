from __future__ import annotations

import argparse
import json
from typing import Any

from ITD_agent.cli.commands import export, review, run, state, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="itd-agent", description="ITD_agent command surface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run.register(subparsers)
    review.register(subparsers)
    train.register(subparsers)
    state.register(subparsers)
    export.register(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command handler configured.")
    payload: dict[str, Any] = handler(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
