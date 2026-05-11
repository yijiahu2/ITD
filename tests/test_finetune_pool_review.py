from __future__ import annotations

from ITD_agent.finetune_pool.review import ReviewWriteAction, run_review_stage


def test_finetune_pool_review_has_formal_entrypoint_and_write_actions() -> None:
    assert callable(run_review_stage)
    assert ReviewWriteAction.WRITE_MEMORY == "write_memory"
    assert ReviewWriteAction.WRITE_FINETUNE_SAMPLE == "write_finetune_sample"
