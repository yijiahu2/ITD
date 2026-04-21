from __future__ import annotations

import unittest
from unittest import mock

from ITD_agent.planning.agent import local_refine


class RunOneGroupRefinementSignalIdsTest(unittest.TestCase):
    def test_ignores_signal_roi_placeholder_ids_for_geometry_groups(self) -> None:
        captured: dict[str, object] = {}

        def fake_prepare_roi_refinement_inputs(**kwargs):
            captured["xiaoban_ids"] = kwargs.get("xiaoban_ids")
            return {
                "roi_image_tif": "/tmp/roi_image.tif",
                "roi_xiaoban_gpkg": None,
                "roi_dem_tif": None,
                "roi_slope_tif": None,
                "roi_aspect_tif": None,
                "roi_landform_tif": None,
                "roi_slope_position_tif": None,
            }

        group = {
            "strategy": "dense_balanced",
            "params": {"diam_list": "160,256,384"},
            "xiaoban_ids": ["signal_roi_00_02"],
            "prior_xiaoban_ids": [],
            "roi_geometry_wkt": "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
            "roi_geometry_crs": "EPSG:4547",
            "roi_candidate": {"candidate_id": "signal_roi_00_02"},
        }

        with mock.patch.object(local_refine, "prepare_roi_refinement_inputs", side_effect=fake_prepare_roi_refinement_inputs), \
             mock.patch.object(local_refine, "build_local_refine_config", return_value={"output_dir": "/tmp/local", "xiaoban_id_field": "XBH"}), \
             mock.patch.object(local_refine, "run_semantic_prior_task_via_worker", return_value={"m_sem_tif": "/tmp/m_sem.tif"}), \
             mock.patch.object(local_refine, "execute_segmentation_model", return_value={"y_inst_shp": "/tmp/local_y_inst.shp"}), \
             mock.patch.object(local_refine.Path, "exists", return_value=True), \
             mock.patch.object(local_refine, "ensure_dir"), \
             mock.patch.object(local_refine, "merge_global_and_local_instances", side_effect=RuntimeError("stop-after-prepare")):
            with self.assertRaisesRegex(RuntimeError, "stop-after-prepare"):
                local_refine.run_one_group_refinement(
                    base_config_path="/tmp/base.yaml",
                    base_cfg={"xiaoban_id_field": "XBH"},
                    current_global_shp="/tmp/global.shp",
                    group_idx=1,
                    group=group,
                    xiaoban_id_field="XBH",
                    buffer_m=5.0,
                    local_root=local_refine.Path("/tmp/local_refine"),
                    terrain_info={},
                )

        self.assertIsNone(captured["xiaoban_ids"])

    def test_keeps_real_xiaoban_ids_for_non_geometry_groups(self) -> None:
        captured: dict[str, object] = {}

        def fake_prepare_roi_refinement_inputs(**kwargs):
            captured["xiaoban_ids"] = kwargs.get("xiaoban_ids")
            return {
                "roi_image_tif": "/tmp/roi_image.tif",
                "roi_xiaoban_gpkg": None,
                "roi_dem_tif": None,
                "roi_slope_tif": None,
                "roi_aspect_tif": None,
                "roi_landform_tif": None,
                "roi_slope_position_tif": None,
            }

        group = {
            "strategy": "dense_balanced",
            "params": {"diam_list": "160,256,384"},
            "xiaoban_ids": ["16", "25"],
            "prior_xiaoban_ids": [],
            "roi_geometry_wkt": None,
            "roi_geometry_crs": None,
            "roi_candidate": None,
        }

        with mock.patch.object(local_refine, "prepare_roi_refinement_inputs", side_effect=fake_prepare_roi_refinement_inputs), \
             mock.patch.object(local_refine, "build_local_refine_config", return_value={"output_dir": "/tmp/local", "xiaoban_id_field": "XBH"}), \
             mock.patch.object(local_refine, "run_semantic_prior_task_via_worker", return_value={"m_sem_tif": "/tmp/m_sem.tif"}), \
             mock.patch.object(local_refine, "execute_segmentation_model", return_value={"y_inst_shp": "/tmp/local_y_inst.shp"}), \
             mock.patch.object(local_refine.Path, "exists", return_value=True), \
             mock.patch.object(local_refine, "ensure_dir"), \
             mock.patch.object(local_refine, "merge_global_and_local_instances", side_effect=RuntimeError("stop-after-prepare")):
            with self.assertRaisesRegex(RuntimeError, "stop-after-prepare"):
                local_refine.run_one_group_refinement(
                    base_config_path="/tmp/base.yaml",
                    base_cfg={"xiaoban_id_field": "XBH"},
                    current_global_shp="/tmp/global.shp",
                    group_idx=1,
                    group=group,
                    xiaoban_id_field="XBH",
                    buffer_m=5.0,
                    local_root=local_refine.Path("/tmp/local_refine"),
                    terrain_info={},
                )

        self.assertEqual(captured["xiaoban_ids"], ["16", "25"])


if __name__ == "__main__":
    unittest.main()
