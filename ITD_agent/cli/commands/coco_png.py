from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

def register(subparsers: Any) -> None:
    parser: ArgumentParser = subparsers.add_parser(
        "coco-png-infer",
        help="Run COCO/PNG dataset inference through the formal ITD_agent workflow.",
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-id", action="append", default=None)
    parser.add_argument("--image-name", action="append", default=None)
    parser.add_argument("--max-expert-rounds", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.set_defaults(handler=handle)


def handle(args: Namespace) -> dict[str, Any]:
    from ITD_agent.orchestration.workflow import coco_png_infer

    return coco_png_infer(
        template=args.template,
        dataset_root=args.dataset_root,
        image_root=args.image_root,
        annotation=args.annotation,
        output_dir=args.output_dir,
        run_name=args.run_name,
        split=args.split,
        max_images=args.max_images,
        image_ids=args.image_id,
        image_names=args.image_name,
        max_expert_rounds=args.max_expert_rounds,
        device=args.device,
    )
