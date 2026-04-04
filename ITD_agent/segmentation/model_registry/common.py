from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from tools.process_runner import run_streaming
from ITD_agent.segmentation.model_registry.output_utils import (
    ensure_vector_from_label_tif,
    export_rgb_png_from_raster,
    materialize_segmentation_outputs_from_prediction_npz,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_selected_algorithm_name(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("segmentation_algorithm", "")).strip().lower()
    if not name:
        raise ValueError("segmentation_algorithm is empty")
    return name


def resolve_algorithm_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    algorithm_name = get_selected_algorithm_name(cfg)
    merged: dict[str, Any] = {}

    candidate_cfgs = cfg.get("segmentation_candidate_cfgs")
    if isinstance(candidate_cfgs, dict):
        candidate_cfg = candidate_cfgs.get(algorithm_name)
        if isinstance(candidate_cfg, dict):
            merged.update(candidate_cfg)

    inline_cfg = cfg.get("segmentation_algorithm_cfg")
    if isinstance(inline_cfg, dict):
        merged.update(inline_cfg)

    return merged


def build_default_outputs(output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "y_inst_tif": str(out_dir / "Y_inst.tif"),
        "y_inst_shp": str(out_dir / "Y_inst.shp"),
        "y_inst_color_png": str(out_dir / "Y_inst_color.png"),
    }


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def save_json(obj: dict[str, Any], path: str | Path) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(out_path)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_command_context(
    cfg: dict[str, Any],
    m_sem_tif: str,
    algorithm_cfg: dict[str, Any],
    outputs: dict[str, str],
    resolved_cfg_json: str,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    context.update(algorithm_cfg)
    context.update(outputs)
    context.update(
        {
            "project_root": str(PROJECT_ROOT),
            "input_image": str(cfg["input_image"]),
            "m_sem_tif": str(m_sem_tif),
            "output_dir": str(cfg["output_dir"]),
            "algorithm_name": get_selected_algorithm_name(cfg),
            "resolved_cfg_json": str(resolved_cfg_json),
        }
    )
    return context


def format_command_template(template: str, context: dict[str, Any]) -> str:
    return template.format_map(_SafeFormatDict(context))


def run_conda_command(
    *,
    conda_sh: str,
    conda_env: str,
    command: str,
    cwd: str | None = None,
):
    project_root = shlex.quote(str(PROJECT_ROOT))
    bash_cmd = (
        f"source {shlex.quote(conda_sh)} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"export PYTHONNOUSERSITE=1 && "
        f"export MPLCONFIGDIR=/tmp/matplotlib-segmentation && "
        f"export PYTHONPATH={project_root}:${{PYTHONPATH:-}} && "
        f"{command}"
    )
    return run_streaming(
        ["bash", "-lc", bash_cmd],
        cwd=cwd,
        print_cmd=True,
        cmd_label="===== SEGMENTATION ALGORITHM CMD =====",
    )


def require_file(path: str | Path, desc: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def collect_segmentation_result(
    *,
    cfg: dict[str, Any],
    algorithm_cfg: dict[str, Any],
    outputs: dict[str, str],
) -> dict[str, Any]:
    score_json = Path(cfg["output_dir"]) / "instance_scores.json"
    if not Path(outputs["y_inst_shp"]).exists() and Path(outputs["y_inst_tif"]).exists():
        ensure_vector_from_label_tif(
            outputs["y_inst_tif"],
            outputs["y_inst_shp"],
            score_json=str(score_json) if score_json.exists() else None,
            min_area_px=int(algorithm_cfg.get("min_area_px", 1)),
        )

    required_outputs = algorithm_cfg.get("required_outputs") or ["y_inst_shp"]
    for key in required_outputs:
        if key not in outputs:
            raise KeyError(f"Unknown required output key: {key}")
        require_file(outputs[key], f"Segmentation output {key}")

    result = {
        "algorithm": get_selected_algorithm_name(cfg),
        "y_inst_tif": outputs["y_inst_tif"],
        "y_inst_shp": outputs["y_inst_shp"],
        "y_inst_color_png": outputs["y_inst_color_png"],
    }

    if score_json.exists():
        result["instance_scores_json"] = str(score_json)
    return result


def run_external_algorithm(
    cfg: dict[str, Any],
    m_sem_tif: str,
    *,
    default_driver_script: str | None = None,
    default_driver_module: str | None = None,
    default_algorithm_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    algorithm_cfg: dict[str, Any] = {}
    if default_algorithm_cfg:
        algorithm_cfg.update(default_algorithm_cfg)
    algorithm_cfg.update(resolve_algorithm_cfg(cfg))
    outputs = build_default_outputs(cfg["output_dir"])
    resolved_cfg_json = save_json(
        algorithm_cfg,
        Path(cfg["output_dir"]) / "segmentation_algorithm_cfg.resolved.json",
    )

    conda_sh = str(algorithm_cfg.get("conda_sh") or cfg.get("conda_sh") or "").strip()
    conda_env = str(algorithm_cfg.get("conda_env") or cfg.get("conda_env") or "").strip()
    if not conda_sh or not conda_env:
        raise ValueError("conda_sh / conda_env is required for external segmentation algorithm execution")

    cwd = str(
        algorithm_cfg.get("cwd")
        or algorithm_cfg.get("repo_root")
        or cfg.get("work_dir")
        or PROJECT_ROOT
    )

    context = build_command_context(cfg, m_sem_tif, algorithm_cfg, outputs, resolved_cfg_json)
    command_template = str(algorithm_cfg.get("command_template") or "").strip()
    extra_cli_args = str(algorithm_cfg.get("extra_cli_args") or "").strip()

    if command_template:
        command = format_command_template(command_template, context)
    else:
        input_png = str(Path(cfg["output_dir"]) / "segmentation_input_rgb.png")
        pred_npz = str(Path(cfg["output_dir"]) / "segmentation_predictions.npz")
        export_rgb_png_from_raster(str(cfg["input_image"]), input_png)
        driver_script = str(algorithm_cfg.get("driver_script") or default_driver_script or "").strip()
        driver_module = str(algorithm_cfg.get("driver_module") or default_driver_module or "").strip()
        if driver_module:
            cmd = [
                "python",
                "-m",
                driver_module,
                "--config_json",
                resolved_cfg_json,
                "--input_png",
                input_png,
                "--pred_npz",
                pred_npz,
                "--input_image",
                str(cfg["input_image"]),
                "--msem_tif",
                str(m_sem_tif),
                "--output_dir",
                str(cfg["output_dir"]),
                "--algorithm_name",
                get_selected_algorithm_name(cfg),
                "--y_inst_tif",
                outputs["y_inst_tif"],
                "--y_inst_shp",
                outputs["y_inst_shp"],
                "--y_inst_color_png",
                outputs["y_inst_color_png"],
            ]
        elif driver_script:
            cmd = [
                "python",
                driver_script,
                "--config_json",
                resolved_cfg_json,
                "--input_png",
                input_png,
                "--pred_npz",
                pred_npz,
                "--input_image",
                str(cfg["input_image"]),
                "--msem_tif",
                str(m_sem_tif),
                "--output_dir",
                str(cfg["output_dir"]),
                "--algorithm_name",
                get_selected_algorithm_name(cfg),
                "--y_inst_tif",
                outputs["y_inst_tif"],
                "--y_inst_shp",
                outputs["y_inst_shp"],
                "--y_inst_color_png",
                outputs["y_inst_color_png"],
            ]
        else:
            raise ValueError(
                "No external execution entry configured. "
                "Set command_template, driver_module, or driver_script in segmentation_algorithm_cfg/"
                "segmentation_candidate_cfgs."
            )
        command = shell_join(cmd)
        if extra_cli_args:
            command = f"{command} {extra_cli_args}"

    res = run_conda_command(conda_sh=conda_sh, conda_env=conda_env, command=command, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(
            f"External segmentation algorithm failed: algorithm={get_selected_algorithm_name(cfg)}, "
            f"returncode={res.returncode}"
        )

    pred_npz_path = Path(cfg["output_dir"]) / "segmentation_predictions.npz"
    if pred_npz_path.exists():
        max_instances = algorithm_cfg.get("max_instances")
        max_instances = None if max_instances in (None, "", 0) else int(max_instances)
        materialize_segmentation_outputs_from_prediction_npz(
            input_image=str(cfg["input_image"]),
            m_sem_tif=str(m_sem_tif),
            pred_npz=str(pred_npz_path),
            outputs=outputs,
            score_thr=float(algorithm_cfg.get("score_thr", 0.2)),
            min_area_px=int(algorithm_cfg.get("min_area_px", 50)),
            min_sem_overlap_ratio=float(algorithm_cfg.get("min_sem_overlap_ratio", 0.01)),
            clip_to_msem=bool(algorithm_cfg.get("clip_to_msem", True)),
            max_instances=max_instances,
        )

    return collect_segmentation_result(cfg=cfg, algorithm_cfg=algorithm_cfg, outputs=outputs)
