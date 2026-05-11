from __future__ import annotations

import argparse
import json

from ITD_agent.evolution.evolve_infer_runner import run_evolve_infer_v1
from ITD_agent.evolution.config_preflight import preflight_evolve_config_v1
from ITD_agent.evolution.review.finetune_bundle_exporter import export_finetune_bundle
from ITD_agent.evolution.review.review_runner import run_review_v2
from ITD_agent.evolution.state.queries import list_pending_reviews, list_review_pending, summarize_review_assets, summarize_state
from ITD_agent.training_loop.training_runner import run_training_loop_v3


def main() -> None:
    parser = argparse.ArgumentParser(prog="itd-agent", description="ITD_agent command surface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evolve_parser = subparsers.add_parser("evolve-infer", help="Run supervised COCO main-expert evolution loop V1.")
    evolve_parser.add_argument("--config", required=True)
    preflight_parser = subparsers.add_parser("evolve-preflight", help="Validate a V1 evolve-infer config without loading models.")
    preflight_parser.add_argument("--config", required=True)

    state_parser = subparsers.add_parser("state", help="Query V1 SQLite state.")
    state_subparsers = state_parser.add_subparsers(dest="state_command", required=True)
    summary_parser = state_subparsers.add_parser("summary", help="Show run and table counts.")
    summary_parser.add_argument("--db", required=True)
    pending_parser = state_subparsers.add_parser("pending", help="Show pending review candidates.")
    pending_parser.add_argument("--db", required=True)
    pending_parser.add_argument("--limit", type=int, default=50)

    review_parser = subparsers.add_parser("review", help="Run or query V2 trajectory review.")
    review_subparsers = review_parser.add_subparsers(dest="review_command", required=True)
    review_run_parser = review_subparsers.add_parser("run", help="Review a completed V1 run and write V2 assets.")
    review_run_parser.add_argument("--config", required=True)
    review_pending_parser = review_subparsers.add_parser("pending", help="List V2 deferred, rejected, and human-review events.")
    review_pending_parser.add_argument("--db", required=True)
    review_pending_parser.add_argument("--limit", type=int, default=50)
    review_assets_parser = review_subparsers.add_parser("assets", help="Summarize V2 review assets.")
    review_assets_parser.add_argument("--db", required=True)
    review_assets_parser.add_argument("--review-run-id")

    finetune_parser = subparsers.add_parser("finetune-pool", help="V2 finetune-pool utilities.")
    finetune_subparsers = finetune_parser.add_subparsers(dest="finetune_command", required=True)
    finetune_export_parser = finetune_subparsers.add_parser("export", help="Export V2 finetune pool bundle without training.")
    finetune_export_parser.add_argument("--review-output-dir", required=True)
    finetune_export_parser.add_argument("--out", required=True)

    train_parser = subparsers.add_parser("train", help="Run V3 controlled training loop.")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_run_parser = train_subparsers.add_parser("run", help="Run V3 controlled training orchestration.")
    train_run_parser.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "evolve-infer":
        payload = run_evolve_infer_v1(args.config)
    elif args.command == "evolve-preflight":
        payload = preflight_evolve_config_v1(args.config)
    elif args.command == "review" and args.review_command == "run":
        payload = run_review_v2(args.config)
    elif args.command == "review" and args.review_command == "pending":
        payload = list_review_pending(args.db, limit=args.limit)
    elif args.command == "review" and args.review_command == "assets":
        payload = summarize_review_assets(args.db, review_run_id=args.review_run_id)
    elif args.command == "finetune-pool" and args.finetune_command == "export":
        payload = export_finetune_bundle(review_output_dir=args.review_output_dir, out_dir=args.out)
    elif args.command == "train" and args.train_command == "run":
        payload = run_training_loop_v3(args.config)
    elif args.state_command == "summary":
        payload = summarize_state(args.db)
    else:
        payload = list_pending_reviews(args.db, limit=args.limit)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
