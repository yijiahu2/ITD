from __future__ import annotations

from input_layer.common import add_issue, path_exists
from input_layer.contracts import DomInputContract, ValidationIssue


def validate_dom_input_contract(contract: DomInputContract | None, issues: list[ValidationIssue]) -> None:
    if contract is None:
        return

    if "dom_contract_unreadable_source" in contract.warnings:
        add_issue(
            issues,
            level="error",
            code="dom_source_unreadable",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 无法读取为受支持的栅格格式。",
            path=contract.source_path,
        )
    if not path_exists(contract.source_path):
        add_issue(
            issues,
            level="error",
            code="missing_dom_source",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 源文件不存在。",
            path=contract.source_path,
        )
    if contract.width <= 0 or contract.height <= 0:
        add_issue(
            issues,
            level="error",
            code="invalid_dom_shape",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 宽高必须大于 0。",
            path=contract.source_path,
        )
    if not contract.crs:
        add_issue(
            issues,
            level="error",
            code="missing_dom_crs",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 缺少 CRS。",
            path=contract.source_path,
        )
    if not contract.transform:
        add_issue(
            issues,
            level="error",
            code="missing_dom_transform",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 缺少 transform。",
            path=contract.source_path,
        )
    if len(contract.bounds) != 4 or contract.bounds[0] >= contract.bounds[2] or contract.bounds[1] >= contract.bounds[3]:
        add_issue(
            issues,
            level="error",
            code="invalid_dom_bounds",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM bounds 非法。",
            path=contract.source_path,
        )
    if contract.gsd_x_m is None or contract.gsd_y_m is None:
        add_issue(
            issues,
            level="error",
            code="missing_dom_gsd",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 无法计算 GSD。",
            path=contract.source_path,
        )
    if not {"red", "green", "blue"}.issubset(set(contract.band_mapping.keys())):
        add_issue(
            issues,
            level="error",
            code="missing_rgb_mapping",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 无法构造 RGB 波段映射。",
            path=contract.source_path,
        )
    if not contract.normalization_policy:
        add_issue(
            issues,
            level="error",
            code="missing_normalization_policy",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 缺少归一化策略。",
            path=contract.source_path,
        )
    if not contract.nodata_policy:
        add_issue(
            issues,
            level="error",
            code="missing_nodata_policy",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message="DOM 缺少 nodata 处理策略。",
            path=contract.source_path,
        )
    expected_stride = contract.tile_px - contract.tile_overlap_px
    if contract.tile_stride_px != expected_stride:
        add_issue(
            issues,
            level="warning",
            code="tile_stride_overlap_mismatch",
            source_type="dom_input_contract",
            source_id=contract.dom_id,
            message=f"tile_stride_px={contract.tile_stride_px} 与 tile_px - tile_overlap_px={expected_stride} 不一致。",
            path=contract.source_path,
        )
