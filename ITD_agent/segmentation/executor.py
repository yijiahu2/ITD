from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ITD_agent.segmentation.contracts import SegmentationExecutionRequest, SegmentationExecutionResult
from tools.process_runner import run_streaming


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _require_file(path: str | Path, desc: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def _run_bash_in_conda_env(
    *,
    command: str,
    conda_sh: str,
    conda_env: str,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    bash_cmd = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && {command}"
    return run_streaming(
        ["bash", "-lc", bash_cmd],
        cwd=cwd,
        print_cmd=True,
        cmd_label="===== BASH CMD =====",
    )


def _get_segmentation_models_block(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("ITD_agent") or {}).get("segmentation_models") or {})


def _entry_name(entry: dict[str, Any]) -> str | None:
    for key in ["name", "algorithm", "id", "script"]:
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _resolve_role_entry(cfg: dict[str, Any], model_role: str, preferred_model: str | None = None) -> dict[str, Any]:
    seg_models = _get_segmentation_models_block(cfg)
    if model_role == "main_model":
        entry = seg_models.get("main_model")
        return entry if isinstance(entry, dict) else {}

    child_candidates = seg_models.get("child_models") or seg_models.get("expert_models") or []
    if isinstance(child_candidates, dict):
        child_candidates = list(child_candidates.values())
    if not isinstance(child_candidates, list):
        child_candidates = []

    normalized_candidates = [item for item in child_candidates if isinstance(item, dict)]
    if preferred_model:
        preferred_model = str(preferred_model).strip().lower()
        for item in normalized_candidates:
            if str(_entry_name(item) or "").strip().lower() == preferred_model:
                return item
    if normalized_candidates:
        return normalized_candidates[0]
    return {}


def _apply_model_entry(cfg: dict[str, Any], model_role: str, model_entry: dict[str, Any]) -> dict[str, Any]:
    if not model_entry:
        return dict(cfg)
    resolved = dict(cfg)
    resolved["selected_model_role"] = model_role
    resolved["selected_model_name"] = _entry_name(model_entry)
    if model_entry.get("script"):
        resolved["segmentation_script"] = str(model_entry["script"])
    if model_entry.get("algorithm"):
        resolved["segmentation_algorithm"] = str(model_entry["algorithm"])
    if model_entry.get("algorithm_module"):
        resolved["segmentation_algorithm_module"] = str(model_entry["algorithm_module"])
    if isinstance(model_entry.get("algorithm_cfg"), dict):
        resolved["segmentation_algorithm_cfg"] = dict(model_entry["algorithm_cfg"])
    for key in ["config_file", "checkpoint", "device"]:
        if model_entry.get(key):
            resolved[key] = model_entry[key]
    if isinstance(model_entry.get("runtime_overrides"), dict):
        resolved.update(model_entry["runtime_overrides"])
    return resolved


def resolve_execution_cfg(
    *,
    cfg: dict[str, Any],
    model_role: str,
    preferred_model: str | None = None,
) -> dict[str, Any]:
    model_entry = _resolve_role_entry(cfg, model_role=model_role, preferred_model=preferred_model)
    resolved = _apply_model_entry(cfg, model_role=model_role, model_entry=model_entry)
    return resolved


def execute_segmentation_model(
    *,
    cfg: dict[str, Any],
    m_sem_tif: str,
    phase: str,
    model_role: str,
    preferred_model: str | None = None,
    plan_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exec_cfg = resolve_execution_cfg(cfg=cfg, model_role=model_role, preferred_model=preferred_model)
    request = SegmentationExecutionRequest(
        phase=phase,
        model_role=model_role,
        algorithm_name=str(exec_cfg.get("segmentation_algorithm") or exec_cfg.get("selected_model_name") or "legacy_cellpose_sam"),
        runtime_cfg={
            "segmentation_algorithm": exec_cfg.get("segmentation_algorithm"),
            "segmentation_algorithm_module": exec_cfg.get("segmentation_algorithm_module"),
            "segmentation_script": exec_cfg.get("segmentation_script"),
            "selected_model_name": exec_cfg.get("selected_model_name"),
        },
        plan_summary=plan_summary or {},
        required_inputs={
            "input_image": exec_cfg.get("input_image"),
            "m_sem_tif": m_sem_tif,
        },
        expected_outputs=["y_inst_shp"],
    )

    segmentation_algorithm = str(exec_cfg.get("segmentation_algorithm", "")).strip().lower()
    if segmentation_algorithm and segmentation_algorithm != "legacy_cellpose_sam":
        config_json = Path(exec_cfg["output_dir"]) / f"{model_role}_{phase}_algorithm_config.json"
        result_json = Path(exec_cfg["output_dir"]) / f"{model_role}_{phase}_algorithm_result.json"
        _ensure_parent(config_json)
        _save_json(exec_cfg, config_json)
        cmd = [
            "python",
            "-m",
            "ITD_agent.segmentation.model_registry.run_algorithm_entry",
            "--config_json",
            str(config_json),
            "--msem_tif",
            str(m_sem_tif),
            "--out_json",
            str(result_json),
        ]
        res = _run_bash_in_conda_env(
            command=" ".join(shlex.quote(x) for x in cmd),
            conda_sh=exec_cfg["conda_sh"],
            conda_env=exec_cfg["conda_env"],
            cwd=str(PROJECT_ROOT),
        )
        if res.returncode != 0:
            raise RuntimeError(f"Segmentation algorithm failed:\n{res.stderr}")
        _require_file(result_json, "Segmentation algorithm result json")
        result = _load_json(result_json)
        _require_file(result["y_inst_shp"], "Segmentation Y_inst.shp")
        execution_result = SegmentationExecutionResult(
            phase=phase,
            model_role=model_role,
            algorithm_name=request.algorithm_name,
            status="completed",
            output_paths={k: v for k, v in result.items() if isinstance(v, str)},
            command=cmd,
            metadata={
                "selected_model_name": exec_cfg.get("selected_model_name"),
                "plan_summary": plan_summary or {},
            },
        ).to_dict()
        result["cmd"] = cmd
        result["execution_request"] = request.to_dict()
        result["execution_result"] = execution_result
        return result

    cmd = [
        "python",
        exec_cfg["segmentation_script"],
        "--in_tif",
        exec_cfg["input_image"],
        "--msem_tif",
        m_sem_tif,
        "--out_dir",
        exec_cfg["output_dir"],
        "--diam_list",
        str(exec_cfg["diam_list"]),
        "--tile",
        str(exec_cfg["tile"]),
        "--overlap",
        str(exec_cfg["overlap"]),
        "--tile_overlap",
        str(exec_cfg["tile_overlap"]),
        "--bsize",
        str(exec_cfg["bsize"]),
        "--iou_merge_thr",
        str(exec_cfg["iou_merge_thr"]),
    ]
    if _normalize_bool(exec_cfg.get("augment", True)):
        cmd.append("--augment")
    else:
        cmd.append("--no_augment")

    res = _run_bash_in_conda_env(
        command=" ".join(shlex.quote(x) for x in cmd),
        conda_sh=exec_cfg["conda_sh"],
        conda_env=exec_cfg["conda_env"],
        cwd=exec_cfg["work_dir"],
    )
    if res.returncode != 0:
        raise RuntimeError(f"Segmentation script failed:\n{res.stderr}")

    output_dir = Path(exec_cfg["output_dir"])
    y_inst_shp = output_dir / "Y_inst.shp"
    _require_file(y_inst_shp, "Segmentation Y_inst.shp")
    result = {
        "cmd": cmd,
        "y_inst_tif": str(output_dir / "Y_inst.tif"),
        "y_inst_shp": str(y_inst_shp),
        "y_inst_color_png": str(output_dir / "Y_inst_color.png"),
    }
    execution_result = SegmentationExecutionResult(
        phase=phase,
        model_role=model_role,
        algorithm_name=request.algorithm_name,
        status="completed",
        output_paths=result,
        command=cmd,
        metadata={
            "selected_model_name": exec_cfg.get("selected_model_name"),
            "plan_summary": plan_summary or {},
        },
    ).to_dict()
    result["execution_request"] = request.to_dict()
    result["execution_result"] = execution_result
    return result
