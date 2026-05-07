from __future__ import annotations

from pathlib import Path

from input_layer.common import add_issue, path_exists, suffix, validate_coco_annotation
from input_layer.contracts import PublicDatasetSource, ValidationIssue


def validate_public_datasets(
    items: list[PublicDatasetSource],
    issues: list[ValidationIssue],
) -> None:
    for item in items:
        if item.format == "coco":
            annotation_path = item.annotation_path or item.path
            image_root = item.image_root or item.root
            if not path_exists(annotation_path):
                level = "error" if item.required else "warning"
                add_issue(
                    issues,
                    level=level,
                    code="missing_annotation",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="COCO 标注文件不存在。",
                    path=annotation_path,
                )
            else:
                validate_coco_annotation(annotation_path, issues, item.id)
            if image_root and not Path(image_root).exists():
                level = "error" if item.required else "warning"
                add_issue(
                    issues,
                    level=level,
                    code="missing_image_root",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="COCO 图像目录不存在。",
                    path=image_root,
                )
        elif item.format == "parquet":
            parquet_path = item.path or item.root
            if not path_exists(parquet_path):
                level = "error" if item.required else "warning"
                add_issue(
                    issues,
                    level=level,
                    code="missing_path",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="Parquet 数据集路径不存在。",
                    path=parquet_path,
                )
            elif suffix(parquet_path) != ".parquet":
                add_issue(
                    issues,
                    level="warning",
                    code="unexpected_suffix",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="Parquet 数据集建议使用 .parquet 后缀。",
                    path=parquet_path,
                )
        else:
            generic_path = item.path or item.annotation_path or item.root
            if item.required and not path_exists(generic_path):
                add_issue(
                    issues,
                    level="error",
                    code="missing_path",
                    source_type="public_dataset",
                    source_id=item.id,
                    message="公开数据集路径不存在。",
                    path=generic_path,
                )
