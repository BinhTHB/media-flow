from scripts.targeted_ocr_localization import _frame_report, _timestamps
from src.application.services.adaptive_subtitle_region_service import (
    AdaptiveSubtitleRegionService,
)


def test_parses_explicit_review_timestamps():
    assert _timestamps("5,10,15") == [5.0, 10.0, 15.0]


def test_frame_report_uses_real_dimensions_and_local_mask():
    record = {
        "frame_timestamp": 10.0,
        "frame_width": 1440,
        "frame_height": 2560,
        "boxes": [{
            "text": "subtitle",
            "confidence": 0.95,
            "polygon": [],
            "bbox": {"left": 300, "top": 1200, "right": 1100, "bottom": 1300},
            "source": "paddleocr",
        }],
    }

    report = _frame_report(record, AdaptiveSubtitleRegionService())

    assert report["actual_sampled_timestamp"] == 10.0
    assert (report["frame_width"], report["frame_height"]) == (1440, 2560)
    assert report["strategy"] == "expanded_line_bbox"
    assert report["mask_bbox"]["left"] > 0
    assert report["mask_bbox"]["right"] < 1440
    assert report["mask_area_ratio"] < 0.12
