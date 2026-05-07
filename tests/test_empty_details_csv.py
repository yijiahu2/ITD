from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv


def test_summarize_details_csv_handles_empty_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    summary = summarize_details_csv(str(path), top_k=3)

    assert summary["exists"] is True
    assert summary["num_units"] == 0
    assert summary["top_k_reference_units"] == []
