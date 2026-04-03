from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Optional

from tools.process_runner import run_streaming


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_path(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return True


def remove_vector_dataset(path: str | Path) -> list[str]:
    p = Path(path)
    removed: list[str] = []
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        cand = p.with_suffix(ext)
        if cand.exists():
            cand.unlink()
            removed.append(str(cand))
    return removed


def run_cmd(
    cmd: list[str],
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
):
    return run_streaming(cmd, cwd=cwd, env=env)


def run_bash_in_conda_env(
    command: str,
    conda_sh: str,
    conda_env: str,
    cwd: Optional[str] = None,
):
    bash_cmd = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && {command}"
    return run_streaming(
        ["bash", "-lc", bash_cmd],
        cwd=cwd,
        print_cmd=True,
        cmd_label="===== BASH CMD =====",
    )


def require_file(path: str | Path, desc: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def copy_optional_file(src: str | Path | None, dst: str | Path | None) -> str | None:
    if not src or not dst:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_path = Path(dst)
    ensure_parent(dst_path)
    shutil.copy2(src_path, dst_path)
    return str(dst_path)
