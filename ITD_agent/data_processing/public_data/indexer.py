from __future__ import annotations

from input_layer.contracts import InputManifest

from ITD_agent.data_processing.contracts import PublicDatasetProfile
from ITD_agent.model_roles import EXPERT_MODEL_ROLE
from ITD_agent.planning.scheduler.expert_taxonomy import infer_domain_tags, infer_target_expert_families


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
            roles.append(EXPERT_MODEL_ROLE)
    if not roles:
        roles = ["main_model", EXPERT_MODEL_ROLE]
    return roles


def _infer_annotation_type(item) -> str:
    fmt = str(item.format).lower()
    if fmt == "coco":
        return "instance_mask"
    if fmt == "parquet":
        return "dataset_table"
    return "unknown"


def _infer_text_list(metadata: dict, *keys: str) -> list[str]:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(part).strip() for part in value if str(part).strip()]
    return []


def build_public_dataset_profiles(manifest: InputManifest) -> list[PublicDatasetProfile]:
    profiles: list[PublicDatasetProfile] = []
    for item in manifest.public_datasets:
        metadata = item.metadata or {}
        usage_roles = _infer_usage_roles(item)
        profiles.append(
            PublicDatasetProfile(
                source_id=item.id,
                dataset_format=item.format,
                dataset_name=str(metadata.get("dataset_name") or item.id or ""),
                root_path=item.root or item.path or item.image_root,
                annotation_path=item.annotation_path,
                usage_roles=usage_roles,
                target_expert_families=infer_target_expert_families(
                    metadata=metadata,
                    root_path=item.root or item.path or item.image_root,
                    source_id=item.id,
                    usage_roles=usage_roles,
                ),
                forest_types=_infer_text_list(metadata, "forest_types", "forest_type"),
                terrain_tags=_infer_text_list(metadata, "terrain_tags", "terrain_type"),
                domain_tags=infer_domain_tags(
                    metadata=metadata,
                    root_path=item.root or item.path or item.image_root,
                    source_id=item.id,
                ),
                sensor_type=metadata.get("sensor_type"),
                resolution_range=metadata.get("resolution_range"),
                label_quality=metadata.get("label_quality"),
                annotation_type=_infer_annotation_type(item),
                finetune_ready=bool((item.annotation_path or item.path) and item.format in {"coco", "parquet"}),
                metadata=metadata,
            )
        )
    return profiles
