from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Optional

from ITD_agent.orchestration.runtime_paths import get_stage_output_paths
from ITD_agent.orchestration.runtime_support import require_file, run_bash_in_conda_env


def _normalize_extra_args(extra_args: Any) -> list[str]:
    if extra_args is None:
        return []
    if isinstance(extra_args, list):
        return [str(x) for x in extra_args]
    if isinstance(extra_args, str) and extra_args.strip():
        return shlex.split(extra_args)
    return []


def _normalize_semantic_prior_extra_args(extra_args: Any) -> list[str]:
    args = _normalize_extra_args(extra_args)
    cleaned: list[str] = []
    skip_next = False
    for item in args:
        s = str(item).strip()
        if skip_next:
            skip_next = False
            continue
        if s in {"--ckpt", "--checkpoint"}:
            skip_next = True
            continue
        if s.startswith("--ckpt=") or s.startswith("--checkpoint="):
            continue
        cleaned.append(s)
    return cleaned


def run_semantic_prior_task(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = get_stage_output_paths(cfg)
    Path(paths["m_sem_tif"]).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        cfg["semantic_prior_script"],
        "--in_tif",
        cfg["input_image"],
        "--out_dir",
        cfg["output_dir"],
    ]
    semantic_prior_ckpt = cfg.get("semantic_prior_ckpt")
    if semantic_prior_ckpt:
        require_file(semantic_prior_ckpt, "semantic_prior_ckpt")
        cmd.extend(["--ckpt", str(semantic_prior_ckpt)])
    semantic_prior_extra_args = _normalize_semantic_prior_extra_args(cfg.get("semantic_prior_extra_args"))
    if semantic_prior_extra_args:
        cmd.extend(semantic_prior_extra_args)

    res = run_bash_in_conda_env(
        command=" ".join(shlex.quote(x) for x in cmd),
        conda_sh=cfg["conda_sh"],
        conda_env=cfg["conda_env"],
        cwd=cfg["work_dir"],
    )
    if res.returncode != 0:
        raise RuntimeError(f"Semantic prior task failed:\n{res.stderr}")

    require_file(paths["m_sem_tif"], "Semantic prior M_sem.tif")
    return {
        "cmd": cmd,
        "m_sem_tif": paths["m_sem_tif"],
        "m_sem_png": paths["m_sem_png"],
    }


def log_to_mlflow(
    cfg: dict[str, Any],
    run_meta: dict[str, Any],
    semantic_prior_info: dict[str, Any],
    final_inst_shp: str,
    eval_info: dict[str, Any],
) -> None:
    if cfg.get("disable_mlflow", False):
        return
    try:
        import mlflow
    except Exception:
        return

    experiment_name = cfg.get("experiment_name", "forest_agent_dev")
    run_name = cfg.get("run_name", Path(cfg["output_dir"]).name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        for k, v in run_meta.items():
            if k == "terrain_info" and isinstance(v, dict):
                for tk, tv in v.items():
                    mlflow.log_param(f"terrain.{tk}", tv if tv is not None else "")
            else:
                mlflow.log_param(k, v if v is not None else "")

        for k, v in (eval_info.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        artifact_candidates = [
            semantic_prior_info.get("m_sem_tif"),
            semantic_prior_info.get("m_sem_png"),
            final_inst_shp,
            eval_info.get("metrics_json"),
            eval_info.get("details_csv"),
        ]
        shp = Path(final_inst_shp)
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
            p = shp.with_suffix(ext)
            if p.exists():
                artifact_candidates.append(str(p))

        for path in artifact_candidates:
            if path and Path(path).exists():
                try:
                    mlflow.log_artifact(str(path))
                except Exception:
                    pass
