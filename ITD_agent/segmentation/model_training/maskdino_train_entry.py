from __future__ import annotations

import os
import random
import sys
from pathlib import Path


def _split_classes(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _register_dataset(name: str, json_file: str, image_root: str, thing_classes: list[str]) -> None:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    if name in DatasetCatalog.list():
        return
    register_coco_instances(
        name,
        {"thing_classes": thing_classes},
        json_file,
        image_root,
    )


def main() -> None:
    repo_root = Path(os.environ.get("MASKDINO_REPO_ROOT", "/home/xth/MaskDINO")).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from detectron2.engine import default_argument_parser, launch

    parser = default_argument_parser()
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--test-json")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--train-dataset-name", required=True)
    parser.add_argument("--val-dataset-name", required=True)
    parser.add_argument("--test-dataset-name")
    parser.add_argument("--thing-classes", required=True)
    parser.add_argument("--eval-only", dest="eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    args = parser.parse_args()

    thing_classes = _split_classes(args.thing_classes)

    def _worker(worker_args) -> None:
        _register_dataset(
            worker_args.train_dataset_name,
            worker_args.train_json,
            worker_args.image_root,
            thing_classes,
        )
        _register_dataset(
            worker_args.val_dataset_name,
            worker_args.val_json,
            worker_args.image_root,
            thing_classes,
        )
        if worker_args.test_json and worker_args.test_dataset_name:
            _register_dataset(
                worker_args.test_dataset_name,
                worker_args.test_json,
                worker_args.image_root,
                thing_classes,
            )
        import train_net as maskdino_train_net

        maskdino_train_net.main(worker_args)

    port = random.randint(1000, 20000)
    args.dist_url = f"tcp://127.0.0.1:{port}"
    launch(
        _worker,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
