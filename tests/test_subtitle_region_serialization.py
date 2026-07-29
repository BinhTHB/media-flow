import json
import pytest
from src.domain.entities.subtitle_region import SubtitleRegion


def test_subtitle_region_provenance_round_trip():
    """Provenance fields survive serialization round-trip (to_dict with include_provenance=True)."""
    original = SubtitleRegion(
        start_time=1.0,
        end_time=3.0,
        x=0.1,
        y=0.8,
        width=0.5,
        height=0.1,
        confidence=0.9,
        track_id="T00001",
    )

    # Without include_provenance: only base 7 fields
    basic = original.to_dict()
    assert set(basic.keys()) == {
        "start_time", "end_time", "x", "y", "width", "height", "confidence"
    }, "to_dict() default must not expose provenance fields"

    # With include_provenance: full 14 fields
    full = original.to_dict(include_provenance=True)
    assert full["track_id"] == "T00001"

    # Round-trip through JSON
    serialized = json.dumps(full)
    loaded = json.loads(serialized)
    restored = SubtitleRegion(
        start_time=loaded["start_time"],
        end_time=loaded["end_time"],
        x=loaded["x"],
        y=loaded["y"],
        width=loaded["width"],
        height=loaded["height"],
        confidence=loaded["confidence"],
        track_id=loaded["track_id"],
    )
    assert restored.track_id == original.track_id

    # None provenance fields are accepted
    bare = SubtitleRegion(1.0, 3.0, 0.1, 0.8, 0.5, 0.1, 0.9)
    bare_dict = bare.to_dict(include_provenance=True)
    assert bare_dict["track_id"] is None
