from __future__ import annotations

import unittest

from ITD_agent.planning.scheduler import parameter_search
from ITD_agent.planning.scheduler import adaptive_config_generator


class ParameterSearchBsizeTest(unittest.TestCase):
    def test_build_candidate_pool_forces_safe_bsize(self) -> None:
        candidates = parameter_search._build_candidate_pool(
            runtime_cfg={"bsize": 128},
            preliminary_updates={"bsize": 128, "tile": 2048, "overlap": 128, "tile_overlap": 0.35, "augment": True, "iou_merge_thr": 0.35, "diam_list": "128,192,320"},
            scheduler_context={},
            max_candidates=4,
        )

        self.assertGreater(len(candidates), 0)
        self.assertTrue(all(item["params"]["bsize"] == 256 for item in candidates))

    def test_enforce_runtime_caps_forces_safe_bsize(self) -> None:
        generated_cfg = {"bsize": 128}
        runtime_cfg = {"ITD_agent": {"planning": {}}}

        capped = adaptive_config_generator._enforce_runtime_caps(generated_cfg, runtime_cfg)

        self.assertEqual(capped["bsize"], 256)


if __name__ == "__main__":
    unittest.main()
