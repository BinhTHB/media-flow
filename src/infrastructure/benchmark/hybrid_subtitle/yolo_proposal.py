"""
YOLO Text Proposal Provider (Benchmark Wrapper / Backward Compatibility)

Re-exports YoloTextProposalProvider, ProposalBundle, and DecodedFrame from production namespace.
"""

from src.application.dto.proposal_bundle import DecodedFrame, ProposalBundle
from src.infrastructure.detection.yolo_proposal_provider import (
    CROP_PADDING,
    YoloTextProposalProvider,
    _resize_720p,
)

__all__ = [
    "YoloTextProposalProvider",
    "ProposalBundle",
    "DecodedFrame",
    "CROP_PADDING",
    "_resize_720p",
]
