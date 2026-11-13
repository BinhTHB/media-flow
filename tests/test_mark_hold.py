"""A face credit detected on either side of a short hole stays covered."""

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.features.visual.processors.watermark import WatermarkProcessor
from src.features.visual.tracker import ClassAwareTracker

FW, FH = 1920, 1080


def _mark(frame: int, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(
        class_id=1,
        class_name="watermark",
        confidence=0.2,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=FW, frame_height=FH),
        frame_index=frame,
        timestamp_seconds=frame / 30.0,
    )


class _RecordBlur:
    def __init__(self) -> None:
        self.boxes: list[BBox] = []

    def execute(self, frame, bbox: BBox, padding_px: int = 0):
        self.boxes.append(bbox)
        return frame


def test_credit_gap_is_blurred_between_the_two_sightings():
    # 西瓜君 on the Hancock shot: seen at 13.43s and again at 15.63s.
    detections = [
        _mark(403, 506, 381, 668, 419),
        _mark(469, 564, 414, 723, 453),
    ]
    tracks = [track for track in ClassAwareTracker().track(detections) if track.class_name == "watermark"]
    assert len(tracks) == 1

    blur = _RecordBlur()
    WatermarkProcessor(action_name="blur", blur_action=blur).process_frame_tracks(
        frame=None,
        frame_index=450,
        tracks=tracks,
    )
    assert len(blur.boxes) == 1
    box = blur.boxes[0]
    center_y = (box.y1 + box.y2) / 2.0
    # Straight line between the sightings, not a snap onto the later box.
    assert abs(center_y - 423.9) < 8
    assert abs(center_y - 433.5) > 5


def test_distant_marks_do_not_cover_the_gap_between_them():
    detections = [
        _mark(100, 40, 40, 180, 80),
        _mark(160, 1400, 900, 1560, 940),
    ]
    tracks = ClassAwareTracker().track(detections)
    blur = _RecordBlur()
    WatermarkProcessor(action_name="blur", blur_action=blur).process_frame_tracks(
        frame=None,
        frame_index=130,
        tracks=tracks,
    )
    assert blur.boxes == []
