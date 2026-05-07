from __future__ import annotations

import csv
import json
from pathlib import Path

from ITD_agent.data_processing.contracts import ProcessingBlockProfile, RemoteSensingPreflightSummary, TileRunContext


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_block_profiles_jsonl(block_profiles: list[ProcessingBlockProfile], path: str | Path) -> str:
    out = _ensure_parent(path)
    with open(out, "w", encoding="utf-8") as f:
        for item in block_profiles:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return str(out)


def write_tile_plan_csv(tile_contexts: list[TileRunContext], path: str | Path) -> str:
    out = _ensure_parent(path)
    fieldnames = [
        "tile_id",
        "block_id",
        "tile_index",
        "read_window_x",
        "read_window_y",
        "read_window_w",
        "read_window_h",
        "valid_write_x",
        "valid_write_y",
        "valid_write_w",
        "valid_write_h",
        "padding_ratio",
        "edge_tile_flag",
        "skip",
        "final_diam_list",
        "final_augment",
        "final_iou_merge_thr",
        "final_fusion_priority",
        "status",
    ]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in tile_contexts:
            writer.writerow(
                {
                    "tile_id": item.tile_id,
                    "block_id": item.block_id,
                    "tile_index": item.tile_index,
                    "read_window_x": item.read_window[0],
                    "read_window_y": item.read_window[1],
                    "read_window_w": item.read_window[2],
                    "read_window_h": item.read_window[3],
                    "valid_write_x": item.valid_write_window[0],
                    "valid_write_y": item.valid_write_window[1],
                    "valid_write_w": item.valid_write_window[2],
                    "valid_write_h": item.valid_write_window[3],
                    "padding_ratio": item.padding_ratio,
                    "edge_tile_flag": item.edge_tile_flag,
                    "skip": item.skip,
                    "final_diam_list": item.final_diam_list,
                    "final_augment": item.final_augment,
                    "final_iou_merge_thr": item.final_iou_merge_thr,
                    "final_fusion_priority": item.final_fusion_priority,
                    "status": item.status,
                }
            )
    return str(out)


def write_tile_context_exceptions_jsonl(tile_contexts: list[TileRunContext], path: str | Path) -> str:
    out = _ensure_parent(path)
    with open(out, "w", encoding="utf-8") as f:
        for item in tile_contexts:
            if not (item.skip or item.edge_tile_flag or item.tile_delta_detected):
                continue
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return str(out)


def write_preflight_report(summary: RemoteSensingPreflightSummary, path: str | Path) -> str:
    out = _ensure_parent(path)
    template_counts: dict[str, int] = {}
    for item in summary.block_profiles:
        name = str(item.policy_template_name or "unknown")
        template_counts[name] = template_counts.get(name, 0) + 1
    payload = {
        "dom_id": summary.dom_id,
        "working_dom_path": summary.working_dom_path,
        "valid_mask_path": summary.valid_mask_path,
        "block_count": len(summary.block_plan),
        "skip_block_count": sum(1 for item in summary.block_profiles if item.skip_block_candidate),
        "high_heterogeneity_block_count": sum(1 for item in summary.block_profiles if item.block_heterogeneity_level == "high"),
        "policy_template_counts": template_counts,
        "tile_context_count": summary.tile_context_count,
        "artifacts": summary.artifacts,
        "metadata": summary.metadata,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(out)
