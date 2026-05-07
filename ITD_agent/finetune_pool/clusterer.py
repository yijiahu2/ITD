from __future__ import annotations

from collections import defaultdict
from typing import Any

from ITD_agent.finetune_pool.contracts import FinetunePoolCluster
from ITD_agent.model_roles import normalize_model_role


def build_pool_clusters(samples: list[dict[str, Any]]) -> list[FinetunePoolCluster]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        category = str(sample.get("failure_category") or "uncategorized")
        role = normalize_model_role(sample.get("target_model_role"), default="main_model")
        family = str(sample.get("target_expert_family") or "cross_domain_generalist")
        grouped[(role, family, category)].append(sample)

    clusters: list[FinetunePoolCluster] = []
    for (role, family, category), rows in grouped.items():
        label_breakdown: dict[str, int] = {}
        source_types = sorted({str(item.get("source_type") or "unknown") for item in rows})
        tags = sorted({tag for item in rows for tag in (item.get("tags") or [])})
        scene_profiles = [item.get("scene_profile") or {} for item in rows[:5]]
        ready_count = 0
        for item in rows:
            label_status = str(item.get("label_status") or "unknown")
            label_breakdown[label_status] = label_breakdown.get(label_status, 0) + 1
            if item.get("ready_for_training"):
                ready_count += 1
        clusters.append(
            FinetunePoolCluster(
                cluster_id=f"{role}:{family}:{category}",
                target_model_role=role,
                target_expert_family=family,
                failure_category=category,
                source_types=source_types,
                sample_ids=[str(item.get("sample_id")) for item in rows if item.get("sample_id")],
                sample_count=len(rows),
                ready_sample_count=ready_count,
                label_status_breakdown=label_breakdown,
                tags=tags,
                scene_profiles=scene_profiles,
            )
        )
    return sorted(clusters, key=lambda item: (item.ready_sample_count, item.sample_count), reverse=True)
