from typing import Any, Dict

from ITD_agent.evaluation_analysis.detail_ranker import summarize_details_csv as _summarize_details_csv


def summarize_details_csv(details_csv_path: str, top_k: int = 3) -> Dict[str, Any]:
    return _summarize_details_csv(details_csv_path, top_k=top_k)
