from scripts.blur_watermarks import _padded_bbox


def _candidate():
    return {"bbox": {"left": 20, "top": 30, "right": 80, "bottom": 90}}


def test_bottom_padding_expands_only_lower_bound_safely():
    box = _padded_bbox(_candidate(), 100, 100, 0.0, 0.20)

    assert box == {"left": 20, "top": 30, "right": 80, "bottom": 100.0}


def test_uniform_and_bottom_padding_are_clamped_to_frame():
    box = _padded_bbox(_candidate(), 100, 100, 0.20, 0.20)

    assert 0 <= box["left"] < box["right"] <= 100
    assert 0 <= box["top"] < box["bottom"] <= 100
