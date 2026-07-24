from scripts.regenerate_clean_artifacts import SAMPLE_VIDEOS, _write_srt
from scripts.targeted_ocr_localization import _static_region_report


def _frame(timestamp: float, bbox: dict | None, *, track_mode: str = "stationary_track"):
    candidate = {
        "selected": bbox is not None,
        "final_candidate_score": 0.9,
        "ignored_reason": None,
    }
    return {
        "actual_sampled_timestamp": timestamp,
        "frame_width": 720,
        "frame_height": 1280,
        "selected_source_bbox": bbox,
        "track_mode": track_mode,
        "candidates": [candidate],
    }


def test_static_region_accepts_stable_non_bottom_lane():
    reports = {
        f"{index}s": _frame(
            index,
            {"left": 180 + index, "top": 500, "right": 540 + index, "bottom": 570},
        )
        for index in range(5)
    }

    report = _static_region_report(reports)

    assert report["stable_region_found"] is True
    assert report["stable_region_position"] == "middle"
    assert report["stability_score"] >= 0.65
    assert report["frame_coverage_ratio"] == 1.0


def test_static_region_rejects_moving_track():
    reports = {
        f"{index}s": _frame(
            index,
            {"left": 180, "top": 200 + index * 80, "right": 540, "bottom": 270 + index * 80},
            track_mode="moving_track",
        )
        for index in range(5)
    }

    report = _static_region_report(reports)

    assert report["stable_region_found"] is False
    assert report["stable_region_bbox"] is None
    assert report["reason_if_no_stable_region"] == "dynamic_or_moving_region"


def test_mvp_sample_registry_excludes_dynamic_sample1():
    assert set(SAMPLE_VIDEOS) == {"sample2", "sample4", "sample5"}


def test_sidecar_uses_explicit_placeholders(tmp_path):
    path = tmp_path / "output.srt"
    _write_srt(
        path,
        [{"start": 0.0, "end": 1.0, "translated_text": "[REVIEW]"}],
    )

    assert "DỊCH_1" in path.read_text(encoding="utf-8")
