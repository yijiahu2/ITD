from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ITD_agent.evolution.review.io_utils import write_json
from ITD_agent.training_loop.contracts import TrainingPlan, TrainingRunResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_training_plan(plan: TrainingPlan, *, execute: bool) -> TrainingRunResult:
    job_dir = Path(plan.output_dir)
    stdout_log = job_dir / "stdout.log"
    stderr_log = job_dir / "stderr.log"
    if not execute:
        result = TrainingRunResult(
            training_job_id=plan.training_job_id,
            training_mode=plan.training_mode,
            status="skipped",
            returncode=None,
            command=plan.command,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            best_checkpoint_path=None,
            training_metrics_path=None,
            metadata={"reason": "runner.execute_training is false"},
        )
        write_json(job_dir / "training_job_summary.json", result.to_dict())
        write_json(job_dir / "training_metrics.json", {"status": "skipped"})
        return result

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    with stdout_log.open("w", encoding="utf-8") as stdout_f, stderr_log.open("w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(plan.command, cwd=str(PROJECT_ROOT), stdout=stdout_f, stderr=stderr_f, text=True, env=env)
    best_checkpoint = _find_best_checkpoint(job_dir)
    status = "completed" if completed.returncode == 0 and best_checkpoint else "failed"
    metrics_path = job_dir / "training_metrics.json"
    write_json(metrics_path, {"status": status, "returncode": completed.returncode, "best_checkpoint": str(best_checkpoint) if best_checkpoint else None})
    result = TrainingRunResult(
        training_job_id=plan.training_job_id,
        training_mode=plan.training_mode,
        status=status,
        returncode=int(completed.returncode),
        command=plan.command,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        best_checkpoint_path=str(best_checkpoint) if best_checkpoint else None,
        training_metrics_path=str(metrics_path),
        metadata={},
    )
    write_json(job_dir / "training_job_summary.json", result.to_dict())
    return result


def _find_best_checkpoint(root: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ["best*.pth", "model_final.pth", "latest*.pth", "epoch_*.pth", "model_*.pth", "*.pth", "*.pt"]:
        candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (("best" in path.name.lower()) + ("final" in path.name.lower()), path.stat().st_mtime), reverse=True)[0]
