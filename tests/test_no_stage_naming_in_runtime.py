from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("run_evolve_infer_v1", "review_v2", "training_v3", "child_model", "v2_review_dir")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".jsonl", ".md", ".toml", ".txt"}


def test_runtime_and_config_files_do_not_expose_legacy_stage_names() -> None:
    offenders: list[str] = []
    for root in ["ITD_agent", "input_layer", "output_layer", "configs"]:
        for path in (PROJECT_ROOT / root).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(token in text for token in FORBIDDEN):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
