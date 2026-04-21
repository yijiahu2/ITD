from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.knowledge_base import build_public_dataset_knowledge_index


def test_public_dataset_knowledge_index_reads_isprs_metadata() -> None:
    index = build_public_dataset_knowledge_index("/home/xth/forest_agent_project/data/isprs_itd_dataset_metadata.yaml")

    assert index["dataset_count"] >= 11
    dataset4 = next(item for item in index["datasets"] if item["dataset_key"] == "Dataset_4")
    assert dataset4["forest_type_en"] == "subtropical_evergreen_broadleaf_forest"
    assert "dense_adhesion" in dataset4["recommended_expert_families"]
    assert dataset4["parameter_prior"]["tile_size"] in {1280, 1536}
