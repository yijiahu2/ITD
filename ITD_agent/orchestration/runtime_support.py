from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

from tools.process_runner import run_streaming


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAFE_DELETE_ROOTS = (
    PROJECT_ROOT / "outputs",
    Path("/tmp/itd_agent_runtime"),
    Path("/tmp/itd_agent_data_processing"),
)


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


def _dedupe_resolved_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        if raw_path in (None, ""):
            continue
        resolved = Path(raw_path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def build_cleanup_roots(
    runtime_cfg: dict[str, Any] | None = None,
    *,
    extra_roots: Sequence[str | Path] | None = None,
) -> tuple[Path, ...]:
    candidates: list[str | Path] = list(DEFAULT_SAFE_DELETE_ROOTS)
    if isinstance(runtime_cfg, dict):
        output_dir = runtime_cfg.get("output_dir")
        if output_dir:
            candidates.append(output_dir)
        persistent_output_dir = runtime_cfg.get("persistent_output_dir")
        if persistent_output_dir:
            candidates.append(persistent_output_dir)
        for key in ("metrics_json", "details_csv"):
            value = runtime_cfg.get(key)
            if value:
                candidates.append(Path(value).expanduser().parent)
    if extra_roots:
        candidates.extend(extra_roots)
    return _dedupe_resolved_paths(candidates)


def _validate_cleanup_target(
    path: str | Path,
    allowed_roots: Sequence[str | Path] | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    target = Path(path).expanduser().resolve()
    roots = _dedupe_resolved_paths(allowed_roots or DEFAULT_SAFE_DELETE_ROOTS)
    if not roots:
        raise ValueError("No allowed cleanup roots configured.")
    if any(_is_within_root(target, root) for root in roots):
        return target, roots
    allowed = ", ".join(str(root) for root in roots)
    raise ValueError(f"Refusing to delete path outside allowed roots: {target} (allowed: {allowed})")


def remove_path(
    path: str | Path,
    *,
    allowed_roots: Sequence[str | Path] | None = None,
) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    _validate_cleanup_target(p, allowed_roots=allowed_roots)
    if p.is_symlink():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return True


def remove_vector_dataset(
    path: str | Path,
    *,
    allowed_roots: Sequence[str | Path] | None = None,
) -> list[str]:
    p = Path(path)
    removed: list[str] = []
    candidates: list[Path]
    if p.suffix.lower() == ".shp":
        candidates = [p.with_suffix(ext) for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]]
    else:
        candidates = [p]
    for cand in candidates:
        if not cand.exists():
            continue
        _validate_cleanup_target(cand, allowed_roots=allowed_roots)
        if cand.is_symlink() or cand.is_file():
            cand.unlink()
        else:
            shutil.rmtree(cand)
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
    try:
        if src_path.resolve() == dst_path.resolve():
            return str(dst_path)
    except Exception:
        pass
    ensure_parent(dst_path)
    shutil.copy2(src_path, dst_path)
    return str(dst_path)
