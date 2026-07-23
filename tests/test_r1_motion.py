import pytest
from unittest.mock import MagicMock
import numpy as np
from src.infrastructure.benchmark.hybrid_subtitle.r1_motion import R1MotionCoupling, R1Result

def test_r1_motion_coupling_neutral_on_insufficient_frames():
    r1 = R1MotionCoupling()
    track = {
        'track_id': 'T1',
        'groups': [
            {'frame_index': 0, 'union_bbox_xyxy': [100, 100, 200, 150]}
        ]
    }
    result = r1.evaluate(track, MagicMock())
    assert result.rejected is False
    assert result.neutral is True
    assert 'insufficient' in result.reason

def test_r1_motion_coupling_neutral_on_missing_frames():
    r1 = R1MotionCoupling()
    track = {
        'track_id': 'T1',
        'groups': [
            {'frame_index': 0, 'union_bbox_xyxy': [100, 100, 200, 150]},
            {'frame_index': 1, 'union_bbox_xyxy': [102, 100, 202, 150]}
        ]
    }
    bundle = MagicMock()
    # Mocking frame_for_index to raise KeyError
    bundle.frame_for_index.side_effect = KeyError('Not found')
    result = r1.evaluate(track, bundle)
    assert result.rejected is False
    assert result.neutral is True
