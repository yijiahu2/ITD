from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import ImagePriorProfile, RemoteSensingPreflightSummary
from ITD_agent.data_processing.remote_sensing.artifact_io import (
    write_block_profiles_jsonl,
    write_preflight_report,
    write_tile_context_exceptions_jsonl,
    write_tile_plan_csv,
)
from ITD_agent.data_processing.remote_sensing.block_plan import generate_logical_block_plan
from ITD_agent.data_processing.remote_sensing.block_profile import build_processing_block_profiles
from ITD_agent.data_processing.remote_sensing.quality import estimate_quality, prepare_rgb_preview
from ITD_agent.data_processing.remote_sensing.tile_context import build_tile_contexts_for_block
from ITD_agent.data_processing.remote_sensing.texture import estimate_texture
from ITD_agent.data_processing.remote_sensing.tile_plan import build_tile_plan


def build_image_profiles(
    manifest: InputManifest,
    runtime_cfg: dict[str, object],
) -> list[ImagePriorProfile]:
    profiles: list[ImagePriorProfile] = []
    for item in manifest.remote_sensing:
        path = Path(item.path)
        if not path.exists():
            profiles.append(ImagePriorProfile(source_id=item.id, path=item.path, metadata={"status": "missing"}))
            continue
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            bounds = src.bounds
            crs = str(src.crs) if src.crs else item.crs
            width = int(src.width)
            height = int(src.height)
            res_x = abs(float(src.transform.a))
            res_y = abs(float(src.transform.e))
            area_ha = float(width * height * res_x * res_y / 10000.0)
            nodata = src.nodata
            gray_preview = src.read(1, out_shape=(min(src.height, 512), min(src.width, 512))).astype(np.float32)
            gray_valid_mask = np.isfinite(gray_preview)
            if nodata is not None:
                gray_valid_mask &= ~np.isclose(gray_preview, nodata)
            rgb_preview, rgb_valid_mask = prepare_rgb_preview(src, nodata=nodata, max_dim=512)
            quality_summary = {
                "nodata": nodata,
                "bounds": {
                    "left": float(bounds.left),
                    "bottom": float(bounds.bottom),
                    "right": float(bounds.right),
                    "top": float(bounds.top),
                },
                "valid_pixel_ratio_estimate": float(np.mean(gray_valid_mask)) if gray_preview.size else 0.0,
                "quality_metrics": estimate_quality(rgb_preview, rgb_valid_mask),
            }
            texture_summary = estimate_texture(gray_preview, nodata=nodata)
            profiles.append(
                ImagePriorProfile(
                    source_id=item.id,
                    path=item.path,
                    width=width,
                    height=height,
                    crs=crs,
                    resolution_x_m=res_x,
                    resolution_y_m=res_y,
                    area_ha=area_ha,
                    band_count=int(src.count),
                    dtype=str(profile.get("dtype")),
                    quality_summary=quality_summary,
                    texture_summary=texture_summary,
                    tile_plan=build_tile_plan(width=width, height=height, res_x=res_x, res_y=res_y, runtime_cfg=runtime_cfg),
                    metadata={
                        "sensor": item.sensor,
                        "bands": item.bands,
                        "required": item.required,
                    },
                )
            )
    return profiles


def build_remote_sensing_preflight(
    manifest: InputManifest,
    runtime_cfg: dict[str, object],
    storage_layout: dict[str, str],
    image_profiles: list[ImagePriorProfile],
) -> RemoteSensingPreflightSummary | None:
    dom_contract = manifest.dom_input_contract.to_dict() if manifest.dom_input_contract else (runtime_cfg.get("_dom_input_contract") or None)
    if not dom_contract or not image_profiles:
        return None
    working_dom_path = Path(str(dom_contract.get("working_dom_path") or ""))
    if not working_dom_path.exists():
        dom_contract = dict(dom_contract)
        dom_contract["working_dom_path"] = image_profiles[0].path
        valid_mask_path = Path(str(dom_contract.get("valid_mask_path") or ""))
        if not valid_mask_path.exists():
            dom_contract["valid_mask_path"] = None

    block_plan = generate_logical_block_plan(dom_contract)
    block_profiles = build_processing_block_profiles(dom_contract, block_plan, runtime_cfg)

    tile_contexts = []
    for block_profile in block_profiles:
        tile_contexts.extend(build_tile_contexts_for_block(dom_contract, block_profile))

    remote_root = Path(storage_layout["root"]) / "remote_sensing"
    remote_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "processing_block_profile_jsonl": write_block_profiles_jsonl(block_profiles, remote_root / "processing_block_profile.jsonl"),
        "inference_tile_plan_csv": write_tile_plan_csv(tile_contexts, remote_root / "inference_tile_plan.csv"),
        "tile_context_exceptions_jsonl": write_tile_context_exceptions_jsonl(tile_contexts, remote_root / "tile_context_exceptions.jsonl"),
    }
    summary = RemoteSensingPreflightSummary(
        dom_id=str(dom_contract["dom_id"]),
        working_dom_path=str(dom_contract.get("working_dom_path") or ""),
        valid_mask_path=dom_contract.get("valid_mask_path"),
        block_plan=block_plan,
        block_profiles=block_profiles,
        tile_context_count=len(tile_contexts),
        artifacts=artifacts,
        metadata={
            "image_profile_source_id": image_profiles[0].source_id,
            "estimated_block_count": len(block_plan),
            "estimated_tile_count": len(tile_contexts),
        },
    )
    artifacts["preflight_report_json"] = write_preflight_report(summary, remote_root / "preflight_report.json")
    summary.artifacts = artifacts
    return summary


__all__ = ["build_image_profiles", "build_remote_sensing_preflight"]
