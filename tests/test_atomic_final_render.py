import json
from pathlib import Path

import pytest

from src.application.services.final_render_manifest_service import (
    FinalRenderManifestService,
)
from src.application.services.final_timeline_builder import FinalTimelineBuilder
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.shared.exceptions import ValidationError


def test_final_timeline_has_separate_narration_and_display_tracks():
    narration = [
        Segment("dialogue", 0, 4, "source", "complete narration"),
        Segment("visual", 4, 6, "source", "visual caption"),
    ]
    display = [narration[1]]
    placement = SubtitlePlacement(
        alignment=5,
        x=100,
        y=100,
        mask_bbox={"left": 10, "top": 10, "right": 190, "bottom": 190},
    )

    timeline = FinalTimelineBuilder().build(
        narration, display, [], [placement], replacement_required=True
    )

    assert [item.segment_id for item in timeline.narration] == ["dialogue", "visual"]
    assert [item.segment_id for item in timeline.display] == ["visual"]
    assert timeline.replacement_regions[0]["mask_bbox"]["left"] == 10


def test_replace_source_subtitle_requires_mask_bbox():
    segment = Segment("visual", 0, 1, "source", "caption")

    with pytest.raises(ValidationError, match="requires mask_bbox"):
        FinalTimelineBuilder().build(
            [segment],
            [segment],
            [],
            [SubtitlePlacement(alignment=2)],
            replacement_required=True,
        )


def test_final_render_manifest_is_written_and_detects_stale_artifact(tmp_path):
    paths = {}
    for name in FinalRenderManifestService.REQUIRED_ARTIFACTS:
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = str(path)
    service = FinalRenderManifestService()
    manifest = service.build("job", "timeline-one", paths)
    manifest_path = tmp_path / "final_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert manifest_path.exists()
    service.validate(manifest, "timeline-one")
    Path(paths["rendered_video"]).write_bytes(b"stale")
    with pytest.raises(ValidationError, match="changed after manifest"):
        service.validate(manifest, "timeline-one")


def test_missing_placement_report_fails_final_render_gate(tmp_path):
    paths = {}
    for name in FinalRenderManifestService.REQUIRED_ARTIFACTS:
        if name == "placement_quality_report":
            continue
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)

    with pytest.raises(ValidationError, match="placement_quality_report"):
        FinalRenderManifestService().build("job", "timeline", paths)
