import json
import os

from PIL import Image
import pytest

from src.domain.entities.frame import Frame
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.infrastructure.providers.subtitle.pillow_subtitle_placement_debug_provider import (
    PillowSubtitlePlacementDebugProvider,
)
from src.shared.exceptions import ProviderError


def _paths(tmp_path):
    return {
        "before_path": str(tmp_path / "debug" / "before_render.jpg"),
        "overlay_path": str(tmp_path / "debug" / "overlay_debug.jpg"),
        "after_path": str(tmp_path / "debug" / "after_render.jpg"),
        "regions_path": str(tmp_path / "debug" / "subtitle_regions.json"),
        "placements_path": str(tmp_path / "debug" / "subtitle_placements.json"),
    }


def _frame(tmp_path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 180), "white").save(image_path)
    return Frame(timestamp=1.0, image_path=str(image_path))


def _rendered_video(tmp_path):
    path = tmp_path / "output.mp4"
    path.write_bytes(b"rendered video")
    return str(path)


def _extract_rendered_frame(_, timestamp, output_path):
    Image.new("RGB", (320, 180), "black").save(output_path)
    return f"ffmpeg_output_seek:{timestamp}"


def test_debug_provider_writes_metadata_and_required_json_fields(tmp_path, monkeypatch):
    frame = _frame(tmp_path)
    region = SubtitleRegion(1.0, 2.0, 0.1, 0.6, 0.5, 0.2, 0.9)
    placement = SubtitlePlacement(alignment=8, x=672, y=888)
    paths = _paths(tmp_path)
    rendered_video_path = _rendered_video(tmp_path)
    monkeypatch.setattr(
        PillowSubtitlePlacementDebugProvider,
        "_extract_rendered_frame",
        staticmethod(_extract_rendered_frame),
    )

    PillowSubtitlePlacementDebugProvider().generate(
        [frame],
        [region],
        [placement],
        rendered_video_path=rendered_video_path,
        **paths,
    )

    assert all(os.path.exists(path) for path in paths.values())
    with open(tmp_path / "debug" / "debug_metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["selected_timestamp"] == frame.timestamp
    assert metadata["source_frame_path"] == frame.image_path
    assert metadata["rendered_video_path"] == rendered_video_path
    assert metadata["extraction_method"] == "ffmpeg_output_seek:1.0"
    assert metadata["before_render_size"] > 0
    assert metadata["after_render_size"] > 0

    with open(paths["regions_path"], encoding="utf-8") as file:
        regions = json.load(file)
    with open(paths["placements_path"], encoding="utf-8") as file:
        placements = json.load(file)
    assert regions[0]["timestamp"] == frame.timestamp
    assert placements[0]["ass_position"] == {"x": 672, "y": 888}


def test_debug_provider_handles_no_region_and_global_fallback(tmp_path, monkeypatch):
    frame = _frame(tmp_path)
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        PillowSubtitlePlacementDebugProvider,
        "_extract_rendered_frame",
        staticmethod(_extract_rendered_frame),
    )

    PillowSubtitlePlacementDebugProvider().generate(
        [frame],
        [],
        [SubtitlePlacement(alignment=2, margin_v=15)],
        rendered_video_path=_rendered_video(tmp_path),
        **paths,
    )

    with open(paths["regions_path"], encoding="utf-8") as file:
        regions = json.load(file)
    with open(paths["placements_path"], encoding="utf-8") as file:
        placements = json.load(file)
    assert regions == []
    assert placements[0]["strategy"] == "global_style_fallback"


def test_debug_provider_rejects_missing_rendered_video(tmp_path):
    with pytest.raises(ProviderError, match="rendered output.mp4 is missing"):
        PillowSubtitlePlacementDebugProvider().generate(
            [_frame(tmp_path)],
            [],
            [],
            rendered_video_path=str(tmp_path / "missing.mp4"),
            **_paths(tmp_path),
        )
