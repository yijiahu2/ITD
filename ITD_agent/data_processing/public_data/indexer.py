from __future__ import annotations

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import PublicDatasetProfile


def _infer_usage_roles(item) -> list[str]:
    roles = []
    metadata = item.metadata or {}
    target_models = metadata.get("target_models") or metadata.get("usage_roles") or []
    if isinstance(target_models, str):
        target_models = [part.strip() for part in target_models.split(",") if part.strip()]
    for role in target_models:
        role_str = str(role).strip().lower()
        if role_str in {"main", "main_model", "primary"}:
            roles.append("main_model")
        elif role_str in {"child", "child_model", "expert", "sub"}:
            roles.append("child_model")
    if not roles:
        roles = ["main_model", "child_model"]
    return roles


def _infer_annotation_type(item) -> str:
    fmt = str(item.format).lower()
    if fmt == "coco":
        return "instance_mask"
    if fmt == "parquet":
        return "dataset_table"
    return "unknown"


def build_public_dataset_profiles(manifest: InputManifest) -> list[PublicDatasetProfile]:
    profiles: list[PublicDatasetProfile] = []
    for item in manifest.public_datasets:
        profiles.append(
            PublicDatasetProfile(
                source_id=item.id,
                dataset_format=item.format,
                root_path=item.root or item.path or item.image_root,
                annotation_path=item.annotation_path,
                usage_roles=_infer_usage_roles(item),
                annotation_type=_infer_annotation_type(item),
                finetune_ready=bool((item.annotation_path or item.path) and item.format in {"coco", "parquet"}),
                metadata=item.metadata,
            )
        )
    return profiles
