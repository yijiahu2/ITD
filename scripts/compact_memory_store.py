from __future__ import annotations

import argparse
import json

from ITD_agent.memory_store import compact_memory_store_records, rebuild_memory_indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact oversized ITD_agent memory_store records.")
    parser.add_argument("--memory-root", default=None, help="Optional custom memory_store root.")
    parser.add_argument("--keep-legacy-duplicates", action="store_true", help="Keep legacy duplicate logs such as execution_log.jsonl.")
    args = parser.parse_args()

    kwargs = {"remove_legacy_duplicates": not args.keep_legacy_duplicates}
    if args.memory_root:
        kwargs["memory_root"] = args.memory_root
    compact_result = compact_memory_store_records(**kwargs)
    rebuild_kwargs = {}
    if args.memory_root:
        rebuild_kwargs["memory_root"] = args.memory_root
    rebuild_result = rebuild_memory_indexes(**rebuild_kwargs)
    print(
        json.dumps(
            {
                "compact_result": compact_result,
                "rebuild_result": rebuild_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
