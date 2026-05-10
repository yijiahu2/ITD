from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evolution.state.queries import list_pending_reviews, summarize_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Query V1 evolve-infer SQLite state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Show run and table counts.")
    summary_parser.add_argument("--db", required=True, help="Path to state.sqlite.")

    pending_parser = subparsers.add_parser("pending", help="Show pending review candidates.")
    pending_parser.add_argument("--db", required=True, help="Path to state.sqlite.")
    pending_parser.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()
    if args.command == "summary":
        payload = summarize_state(args.db)
    else:
        payload = list_pending_reviews(args.db, limit=args.limit)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
