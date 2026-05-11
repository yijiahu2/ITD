from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.common.json_store import append_jsonl, load_jsonl_many, write_json
from ITD_agent.common.scene_profile import scene_profile_from_runtime, scene_profile_from_summary
from ITD_agent.finetune_pool.query import _pool_roots
from ITD_agent.finetune_pool.store import DEFAULT_FINETUNE_POOL_ROOT, SOURCE_LEGACY_FINETUNE_POOL_ROOT
from ITD_agent.memory_store.query import DEFAULT_MEMORY_ROOT, _memory_roots
from ITD_agent.segmentation.coco_utils import build_image_index, load_merged_coco, normalize_split_mapping, resolve_image_path
from ITD_agent.segmentation.model_training.training_utils import grad_accum_steps


def test_runtime_state_defaults_write_outside_source_tree_and_read_legacy() -> None:
    assert PROJECT_ROOT / "outputs" in DEFAULT_MEMORY_ROOT.parents
    assert PROJECT_ROOT / "outputs" in DEFAULT_FINETUNE_POOL_ROOT.parents
    assert PROJECT_ROOT / "ITD_agent" / "memory_store" in _memory_roots()
    assert SOURCE_LEGACY_FINETUNE_POOL_ROOT in _pool_roots()


def test_jsonl_store_dedupes_across_current_and_legacy_roots(tmp_path: Path) -> None:
    current = tmp_path / "current.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    append_jsonl(current, {"sample_id": "same", "value": 1})
    append_jsonl(legacy, {"sample_id": "same", "value": 2})
    append_jsonl(legacy, {"sample_id": "other", "value": 3})

    rows = load_jsonl_many([current, legacy], dedupe_key=lambda item: str(item.get("sample_id") or ""))

    assert [row["sample_id"] for row in rows] == ["same", "other"]
    assert rows[0]["value"] == 1


def test_coco_helpers_merge_ids_and_resolve_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotation"
    image_dir.mkdir()
    ann_dir.mkdir()
    (image_dir / "tile_a.png").write_bytes(b"fake")
    (image_dir / "tile_b.png").write_bytes(b"fake")
    write_json(
        ann_dir / "part1.json",
        {
            "images": [{"id": 7, "file_name": "tile_a.png", "width": 4, "height": 4}],
            "annotations": [{"id": 3, "image_id": 7, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "segmentation": [[0, 0, 1, 0, 1, 1]]}],
            "categories": [{"id": 1, "name": "crown"}],
        },
    )
    write_json(
        ann_dir / "part2.json",
        {
            "images": [{"id": 7, "file_name": "nested/tile_b.png", "width": 4, "height": 4}],
            "annotations": [{"id": 3, "image_id": 7, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "segmentation": [[0, 0, 1, 0, 1, 1]]}],
            "categories": [{"id": 1, "name": "crown"}],
        },
    )

    coco = load_merged_coco(ann_dir)
    by_name, by_stem = build_image_index(image_dir)

    assert [image["id"] for image in coco["images"]] == [1, 2]
    assert [ann["image_id"] for ann in coco["annotations"]] == [1, 2]
    assert resolve_image_path("nested/tile_b.png", by_name, by_stem, image_dir) == image_dir / "tile_b.png"
    assert normalize_split_mapping({"validation": "Val"})["validation"] == "Val"


def test_scene_profile_is_consistent_between_runtime_and_summary() -> None:
    scene_analysis = {
        "forest_type": "mixed",
        "stand_condition": {"labels": ["dense"]},
        "image_texture_analysis": {"labels": ["high_texture"], "levels": {"entropy": "high"}},
        "image_quality_analysis": {"labels": ["shadow"], "levels": {"shadow": "medium"}},
        "terrain_analysis": {"labels": ["steep"], "dom_context": {"landform_type": "mountain"}},
    }
    runtime_cfg = {
        "mainline_profile": "B_DOM_DEM_CHM_KNOWLEDGE",
        "_mainline_capabilities": {
            "allow_dem": True,
            "allow_external_knowledge": True,
            "allow_public_datasets": True,
        },
        "_input_manifest": {"metadata": {"input_modalities": {"image": True, "dem": True}}},
        "_input_assessment": {"scene_analysis": scene_analysis},
        "_data_processing_summary": {
            "image_profiles": [{"resolution_x_m": 0.2}],
            "metadata": {"input_manifest_summary": {"domain_knowledge_items": [{"normalized_type": "rule"}], "public_datasets": [{"usage_roles": ["benchmark"]}]}},
        },
    }
    summary = {
        "run_meta": {},
        "data_processing": {
            "input_assessment": {"scene_analysis": scene_analysis},
            "processing_summary": runtime_cfg["_data_processing_summary"],
        },
    }
    input_manifest = {
        "metadata": {
            "mainline_profile": runtime_cfg["mainline_profile"],
            "mainline_capabilities": runtime_cfg["_mainline_capabilities"],
            "input_modalities": {"image": True, "dem": True},
        }
    }

    from_runtime = scene_profile_from_runtime(runtime_cfg)
    from_summary = scene_profile_from_summary(summary, input_manifest)

    assert from_runtime["terrain_type"] == from_summary["terrain_type"] == "mountain"
    assert from_runtime["knowledge_profile_types"] == from_summary["knowledge_profile_types"] == ["rule"]
    assert "steep" in from_runtime["tags"]
    assert "steep" in from_summary["tags"]


def test_training_helpers_normalize_grad_accumulation() -> None:
    assert grad_accum_steps({"segmentation_train_grad_accum_steps": "4"}) == 4
    assert grad_accum_steps({"segmentation_train_grad_accum_steps": "bad"}) == 1
    assert grad_accum_steps({"segmentation_train_grad_accum_steps": 0}) == 1
