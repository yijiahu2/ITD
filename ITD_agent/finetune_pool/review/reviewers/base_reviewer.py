from __future__ import annotations

from typing import Any

from ITD_agent.finetune_pool.review.review_context_builder import ReviewContext
from ITD_agent.finetune_pool.review.review_policy import ReviewDecision


class BaseReviewer:
    candidate_type = "candidate"

    def review_many(self, candidates: list[dict[str, Any]], context: ReviewContext, cfg: dict[str, Any]) -> list[ReviewDecision]:
        return [self.review(candidate, context, cfg) for candidate in candidates]

    def review(self, candidate: dict[str, Any], context: ReviewContext, cfg: dict[str, Any]) -> ReviewDecision:
        raise NotImplementedError
