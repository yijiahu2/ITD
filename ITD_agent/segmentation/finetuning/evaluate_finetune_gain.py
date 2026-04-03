from __future__ import annotations

import argparse

from ITD_agent.evaluation_analysis.finetune_effect_assessment import compare_finetune_effect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_csv", required=True)
    parser.add_argument("--after_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", required=False)
    parser.add_argument("--before_pred_shp")
    parser.add_argument("--after_pred_shp")
    parser.add_argument("--gt_tree_crowns_shp")
    parser.add_argument("--score_field")
    args = parser.parse_args()

    compare_finetune_effect(
        before_csv=args.before_csv,
        after_csv=args.after_csv,
        out_dir=args.out_dir,
        before_pred_shp=args.before_pred_shp,
        after_pred_shp=args.after_pred_shp,
        gt_shp=args.gt_tree_crowns_shp,
        score_field=args.score_field,
        config_path=args.config,
    )
    print(f"[OK] finetune compare done: {args.out_dir}")


if __name__ == "__main__":
    main()
