from __future__ import annotations

from pathlib import Path
from typing import Any

from input_layer.common import as_list, infer_dataset_format, resolve_path
from input_layer.contracts import DatasetSource, PublicDatasetSource


def normalize_dataset_sources(raw_items: Any, config_dir: Path | None) -> list[DatasetSource]:
    sources: list[DatasetSource] = []
    for idx, item in enumerate(as_list(raw_items), 1):
        if isinstance(item, PublicDatasetSource):
            sources.append(item)
            continue
        if isinstance(item, str):
            path = resolve_path(item, config_dir)
            if not path:
                continue
            dataset_format = infer_dataset_format(path)
            if dataset_format == "coco":
                sources.append(
                    DatasetSource(
                        id=f"dataset_{idx:03d}",
                        format=dataset_format,
                        annotation_path=path,
                        root=str(Path(path).parent),
                    )
                )
            else:
                sources.append(DatasetSource(id=f"dataset_{idx:03d}", format=dataset_format, path=path))
            continue
        if isinstance(item, dict):
            raw_path = resolve_path(item.get("path"), config_dir)
            root = resolve_path(item.get("root"), config_dir)
            image_root = resolve_path(item.get("image_root"), config_dir)
            annotation_path = resolve_path(item.get("annotation_path"), config_dir)
            dataset_format = infer_dataset_format(annotation_path or raw_path, explicit_format=item.get("format"))
            sources.append(
                DatasetSource(
                    id=str(item.get("id") or item.get("name") or f"dataset_{idx:03d}"),
                    format=dataset_format,
                    path=raw_path,
                    root=root,
                    image_root=image_root,
                    annotation_path=annotation_path,
                    schema_mapping={str(k): str(v) for k, v in (item.get("schema_mapping") or {}).items() if v},
                    required=bool(item.get("required", False)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"id", "name", "format", "path", "root", "image_root", "annotation_path", "schema_mapping", "required"}
                    },
                )
            )
    return sources


def parse_public_datasets(
    public_cfg: Any,
    cfg: dict[str, Any],
    config_dir: Path | None,
) -> list[DatasetSource]:
    raw_items = public_cfg
    if isinstance(public_cfg, dict):
        raw_items = public_cfg.get("datasets")
    if raw_items is None:
        raw_items = cfg.get("public_datasets")
    return normalize_dataset_sources(raw_items, config_dir)
