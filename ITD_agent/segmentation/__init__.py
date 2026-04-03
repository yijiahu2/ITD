"""Segmentation execution, registry, training, and finetuning modules."""

from ITD_agent.segmentation.contracts import (
    SegmentationExecutionRequest,
    SegmentationExecutionResult,
    SegmentationFinetuneRequest,
    SegmentationFinetuneResult,
)
from ITD_agent.segmentation.executor import execute_segmentation_model, resolve_execution_cfg

__all__ = [
    "execute_segmentation_model",
    "resolve_execution_cfg",
    "SegmentationExecutionRequest",
    "SegmentationExecutionResult",
    "SegmentationFinetuneRequest",
    "SegmentationFinetuneResult",
]
