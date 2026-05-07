from __future__ import annotations

import os
import random
import sys
from pathlib import Path


class _BrokenPipeTolerantStream:
    def __init__(self, stream):
        self._stream = stream
        self._broken = False

    def write(self, data):
        if self._broken:
            return len(data)
        try:
            return self._stream.write(data)
        except BrokenPipeError:
            self._broken = True
            return len(data)

    def flush(self):
        if self._broken:
            return
        try:
            self._stream.flush()
        except BrokenPipeError:
            self._broken = True

    def __getattr__(self, name):
        return getattr(self._stream, name)


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


def _patch_gradient_accumulation(maskdino_train_net, grad_accum_steps: int) -> None:
    if grad_accum_steps <= 1:
        return

    import torch

    class GradientAccumulationOptimizer(torch.optim.Optimizer):
        def __init__(self, optimizer, accum_steps: int):
            super().__init__(optimizer.param_groups, optimizer.defaults)
            self._optimizer = optimizer
            self._accum_steps = max(1, int(accum_steps))
            self._micro_step = 0
            self._step_supports_amp_scaling = getattr(optimizer, "_step_supports_amp_scaling", False)
            self.state = optimizer.state

        def zero_grad(self, *args, **kwargs):
            if self._micro_step % self._accum_steps == 0:
                return self._optimizer.zero_grad(*args, **kwargs)
            return None

        def step(self, *args, **kwargs):
            self._micro_step += 1
            if self._micro_step % self._accum_steps != 0:
                return None
            for group in self._optimizer.param_groups:
                for param in group.get("params", []):
                    grad = getattr(param, "grad", None)
                    if grad is not None:
                        grad.div_(self._accum_steps)
            return self._optimizer.step(*args, **kwargs)

        def state_dict(self):
            state = self._optimizer.state_dict()
            state["_forest_agent_grad_accum_steps"] = self._accum_steps
            state["_forest_agent_micro_step"] = self._micro_step
            return state

        def load_state_dict(self, state_dict):
            self._accum_steps = int(state_dict.pop("_forest_agent_grad_accum_steps", self._accum_steps))
            self._micro_step = int(state_dict.pop("_forest_agent_micro_step", 0))
            return self._optimizer.load_state_dict(state_dict)

        def __getattr__(self, name):
            return getattr(self._optimizer, name)

    original_build_optimizer = maskdino_train_net.Trainer.build_optimizer

    @classmethod
    def patched_build_optimizer(cls, cfg, model):
        optimizer = original_build_optimizer(cfg, model)
        return GradientAccumulationOptimizer(optimizer, grad_accum_steps)

    maskdino_train_net.Trainer.build_optimizer = patched_build_optimizer


def main() -> None:
    sys.stdout = _BrokenPipeTolerantStream(sys.stdout)
    sys.stderr = _BrokenPipeTolerantStream(sys.stderr)

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
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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
        _patch_gradient_accumulation(maskdino_train_net, max(1, int(worker_args.grad_accum_steps)))

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
