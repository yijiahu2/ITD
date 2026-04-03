from __future__ import annotations

import argparse
from pathlib import Path

from ITD_agent.segmentation.finetuning.io_utils import dump_json, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = Path(cfg["output_dir"]) / "segmentation_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "template_only",
        "message": (
            "This is a template entrypoint for future segmentation SOTA algorithms. "
            "Replace it with a real trainer after the target model is installed."
        ),
        "segmentation_algorithm": cfg.get("segmentation_algorithm"),
        "segmentation_algorithm_module": cfg.get("segmentation_algorithm_module"),
        "segmentation_algorithm_cfg": cfg.get("segmentation_algorithm_cfg", {}),
        "expected_training_artifacts": [
            str(output_dir / "best_model.ckpt"),
            str(output_dir / "train_summary.json"),
        ],
    }
    dump_json(summary, output_dir / "train_summary.json")
    print(f"[TEMPLATE] segmentation training template summary written: {output_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
