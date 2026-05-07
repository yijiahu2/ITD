from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

RUN_ID = "dataset4_model_sweep_fair_20260427_02"
CFG_DIR = Path("/home/xth/forest_agent_project/configs/generated/dataset4_model_sweep_fair_20260427_02")
OUT_ROOT = Path("/home/xth/forest_agent_project/outputs/dataset4_model_sweep_fair_20260427_02")
manifest = yaml.safe_load((CFG_DIR / "manifest.yaml").read_text(encoding="utf-8"))["models"]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_metrics_from_log(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    patterns = {
        "segm_ap50": r"coco/segm_mAP_50:\s*([0-9.]+)",
        "segm_ap75": r"coco/segm_mAP_75:\s*([0-9.]+)",
        "bbox_ap50": r"coco/bbox_mAP_50:\s*([0-9.]+)",
        "bbox_ap75": r"coco/bbox_mAP_75:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


rows = []
for model_key, meta in manifest.items():
    out_dir = Path(meta["output_dir"])
    log_path = OUT_ROOT / "logs" / f"{model_key}.log"
    train_summary = load_json(out_dir / "segmentation_training" / "train_summary.json") or {}
    test_summary = load_json(out_dir / "segmentation_training" / "evaluation" / "test_summary.json") or {}
    log_metrics = parse_metrics_from_log(log_path)
    rows.append(
        {
            "model_key": model_key,
            "algorithm": meta["algorithm"],
            "status": "done" if (OUT_ROOT / "status" / f"{model_key}.done").exists() else "failed" if (OUT_ROOT / "status" / f"{model_key}.failed").exists() else "unknown",
            "segm_ap50": log_metrics.get("segm_ap50"),
            "segm_ap75": log_metrics.get("segm_ap75"),
            "bbox_ap50": log_metrics.get("bbox_ap50"),
            "bbox_ap75": log_metrics.get("bbox_ap75"),
            "best_ckpt": train_summary.get("best_ckpt"),
            "test_summary_json": str(out_dir / "segmentation_training" / "evaluation" / "test_summary.json") if test_summary else "",
        }
    )

csv_path = OUT_ROOT / "model_sweep_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model_key", "algorithm", "status", "segm_ap50", "segm_ap75", "bbox_ap50", "bbox_ap75", "best_ckpt", "test_summary_json"],
    )
    writer.writeheader()
    writer.writerows(rows)

ranked = sorted(rows, key=lambda row: ((row["segm_ap75"] or -1), (row["segm_ap50"] or -1)), reverse=True)
report_lines = [
    f"# {RUN_ID}",
    "",
    "| Model | Status | segm AP50 | segm AP75 |",
    "|---|---:|---:|---:|",
]
for row in ranked:
    report_lines.append(
        f"| {row['model_key']} | {row['status']} | {row['segm_ap50'] if row['segm_ap50'] is not None else ''} | {row['segm_ap75'] if row['segm_ap75'] is not None else ''} |"
    )

(OUT_ROOT / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print(f"[OK] summary written: {csv_path}")
