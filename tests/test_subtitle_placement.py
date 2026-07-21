from unittest.mock import Mock

import pytest

from src.application.services.ass_generator import ASSGenerator
from src.application.services.subtitle_placement_service import SubtitlePlacementService
from src.application.use_cases.calculate_subtitle_placement import (
    CalculateSubtitlePlacementUseCase,
)
from src.application.use_cases.detect_subtitle_region import DetectSubtitleRegionUseCase
from src.domain.entities.frame import Frame
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.placement_config import PlacementConfig
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle
from src.infrastructure.providers.subtitle.mock_subtitle_region_detector_provider import (
    MockSubtitleRegionDetectorProvider,
)
from src.shared.exceptions import ValidationError


def _region(**overrides):
    values = {
        "start_time": 1.0,
        "end_time": 3.0,
        "x": 0.1,
        "y": 0.4,
        "width": 0.5,
        "height": 0.2,
        "confidence": 0.9,
    }
    values.update(overrides)
    return SubtitleRegion(**values)


def _segment(identifier="s1", start=1.0, end=2.0):
    return Segment(
        id=identifier,
        start=start,
        end=end,
        text=f"text-{identifier}",
        source="speech",
        confidence=0.9,
    )


def _config(**overrides):
    values = {
        "padding_px": 20,
        "preferred_relation": "below_source",
        "fallback_alignment": 2,
        "fallback_margin_v": 30,
        "min_region_confidence": 0.8,
        "video_width": 1920,
        "video_height": 1080,
        "estimated_height_px": 50,
        "placement_mode": "avoid_source_subtitle",
        "safe_zone_top": 0.05,
        "safe_zone_bottom": 0.95,
        "safe_zone_left": 0.05,
        "safe_zone_right": 0.95,
    }
    values.update(overrides)
    return PlacementConfig(**values)


def test_subtitle_region_validates_values_and_serializes():
    region = _region()

    assert region.to_dict() == {
        "start_time": 1.0,
        "end_time": 3.0,
        "x": 0.1,
        "y": 0.4,
        "width": 0.5,
        "height": 0.2,
        "confidence": 0.9,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_time", -0.1),
        ("x", -0.1),
        ("x", 1.1),
        ("y", -0.1),
        ("y", 1.1),
        ("width", -0.1),
        ("width", 1.1),
        ("height", -0.1),
        ("height", 1.1),
        ("confidence", -0.1),
        ("confidence", 1.1),
    ],
)
def test_subtitle_region_rejects_invalid_times_coordinates_and_confidence(field, value):
    with pytest.raises(ValidationError):
        _region(**{field: value})


def test_subtitle_region_rejects_end_before_start():
    with pytest.raises(ValidationError):
        _region(start_time=2.0, end_time=1.0)


def test_subtitle_placement_validates_alignment_and_serializes():
    placement = SubtitlePlacement(
        alignment=5, x=960, y=540, margin_v=20, margin_l=10, margin_r=15
    )

    assert placement.to_dict() == {
        "alignment": 5,
        "x": 960,
        "y": 540,
        "margin_v": 20,
        "margin_l": 10,
        "margin_r": 15,
        "is_fallback": False,
    }


@pytest.mark.parametrize("alignment", [0, 10, -1, "2"])
def test_subtitle_placement_rejects_invalid_alignment(alignment):
    with pytest.raises(ValidationError):
        SubtitlePlacement(alignment=alignment)


def test_placement_config_validates_values_and_serializes():
    config = _config()

    assert config.to_dict() == {
        "padding_px": 20,
        "preferred_relation": "below_source",
        "fallback_alignment": 2,
        "fallback_margin_v": 30,
        "min_region_confidence": 0.8,
        "video_width": 1920,
        "video_height": 1080,
        "estimated_height_px": 50,
        "placement_mode": "avoid_source_subtitle",
        "safe_zone_top": 0.05,
        "safe_zone_bottom": 0.95,
        "safe_zone_left": 0.05,
        "safe_zone_right": 0.95,
        "min_source_band_width_ratio": 0.80,
        "vertical_start_ratio": 0.72,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("padding_px", -1),
        ("preferred_relation", "beside_source"),
        ("fallback_alignment", 0),
        ("fallback_alignment", 10),
        ("fallback_margin_v", -1),
        ("min_region_confidence", -0.1),
        ("min_region_confidence", 1.1),
        ("video_width", 0),
        ("video_height", 0),
        ("placement_mode", "unknown"),
        ("safe_zone_top", 0.96),
        ("safe_zone_bottom", 0.04),
        ("safe_zone_left", 0.96),
        ("safe_zone_right", 0.04),
    ],
)
def test_placement_config_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        _config(**{field: value})


def test_match_regions_selects_highest_confidence_overlap_and_none_when_unmatched():
    service = SubtitlePlacementService(_config())
    subtitles = [_segment("matched", 1.0, 2.0), _segment("unmatched", 5.0, 6.0)]
    low_confidence = _region(confidence=0.7)
    high_confidence = _region(confidence=0.95)

    matched = service.match_regions_to_subtitles(
        [low_confidence, high_confidence], subtitles
    )

    assert matched == [
        (subtitles[0], high_confidence),
        (subtitles[1], None),
    ]


def test_match_regions_prefers_spatial_overlap():
    """SubtitlePlacementService must select regions that have spatial
    overlap with the subtitle track's bbox rather than just highest confidence."""
    service = SubtitlePlacementService(_config())
    # Subtitle is at bottom: [900, 960]
    subtitle = Segment(
        id="sub_bottom",
        start=1.0,
        end=2.0,
        text="bottom text",
        source="ocr",
        confidence=0.9,
        bbox={"left": 100, "right": 500, "top": 900, "bottom": 960},
    )
    # Region 1: middle of screen (y=0.4, h=0.2 -> [432, 648]), high confidence 0.95
    region_mid = _region(
        start_time=1.0,
        end_time=2.0,
        x=0.1,
        y=0.4,
        width=0.5,
        height=0.2,
        confidence=0.95,
    )
    # Region 2: bottom of screen (y=0.8, h=0.1 -> [864, 972]), lower confidence 0.85 (intersects [900, 960])
    region_bot = _region(
        start_time=1.0,
        end_time=2.0,
        x=0.1,
        y=0.8,
        width=0.5,
        height=0.1,
        confidence=0.85,
    )

    matched = service.match_regions_to_subtitles(
        [region_mid, region_bot], [subtitle]
    )
    assert matched == [(subtitle, region_bot)]


def test_compute_placement_below_source_uses_top_alignment_and_region_center_x():
    config = _config(preferred_relation="below_source")
    placement = SubtitlePlacementService(config).compute_placement(_region(), 0)

    assert placement.alignment == 8
    assert placement.x == 672
    assert placement.y == 668


def test_compute_placement_above_source_uses_bottom_alignment_and_target_y():
    config = _config(preferred_relation="above_source")
    placement = SubtitlePlacementService(config).compute_placement(_region(), 0)

    assert placement.alignment == 2
    assert placement.x == 672
    assert placement.y == 412


def test_compute_placement_override_uses_region_center():
    config = _config(preferred_relation="override")
    placement = SubtitlePlacementService(config).compute_placement(_region(), 0)

    assert placement.alignment == 5
    assert placement.x == 672
    assert placement.y == 540

def test_replace_source_subtitle_places_caption_over_detected_region():
    placement = SubtitlePlacementService(
        _config(placement_mode="replace_source_subtitle")
    ).compute_placement(_region(), 0)

    assert placement.alignment == 5
    assert (placement.x, placement.y) == (960, 540)
    assert placement.mask_bbox is not None

def test_fixed_safe_zone_ignores_detected_region_and_uses_configured_anchor():
    placement = SubtitlePlacementService(
        _config(
            placement_mode="fixed_safe_zone",
            safe_zone_left=0.1,
            safe_zone_right=0.9,
            safe_zone_bottom=0.8,
        )
    ).compute_placement(_region(), 0)

    assert placement == SubtitlePlacement(alignment=2, x=960, y=864)

def test_avoid_source_subtitle_stays_inside_safe_zone_without_overlap():
    service = SubtitlePlacementService(
        _config(
            placement_mode="avoid_source_subtitle",
            safe_zone_top=0.1,
            safe_zone_bottom=0.9,
        )
    )
    region = _region(y=0.8, height=0.08)
    placement = service.compute_placement(region, 0)
    report = service.build_quality_report(
        [region], [_segment()], [placement], SubtitleStyle()
    )

    assert placement == SubtitlePlacement(alignment=2, x=672, y=844)
    assert report["safe_zone_violations"] == 0
    assert report["source_overlap_count"] == 0
    assert report["contrast_ratio"] == 21.0
    assert report["quality_gate_status"] == "PASS"


@pytest.mark.parametrize("region", [None, _region(confidence=0.79)])
def test_compute_placement_falls_back_for_missing_or_low_confidence_region(region):
    config = _config()

    placement = SubtitlePlacementService(config).compute_placement(region, 0)

    assert placement == SubtitlePlacement(alignment=2, margin_v=30)


def test_calculate_placements_preserves_subtitle_order_and_length():
    service = SubtitlePlacementService(_config(preferred_relation="override"))
    subtitles = [_segment("first", 1.0, 2.0), _segment("second", 5.0, 6.0)]
    regions = [_region(start_time=1.0, end_time=2.0)]

    placements = service.calculate_placements(regions, subtitles)

    assert len(placements) == len(subtitles)
    assert (placements[0].alignment, placements[0].x, placements[0].y) == (5, 672, 540)
    assert placements[0].rendered_bbox is not None
    assert (placements[1].alignment, placements[1].margin_v) == (2, 30)


def test_bottom_edge_region_prefers_safe_above_source_placement():
    placement = SubtitlePlacementService(_config()).compute_placement(
        _region(y=0.88, height=0.1), 0
    )

    assert placement == SubtitlePlacement(alignment=2, x=672, y=930)


def test_top_edge_region_prefers_safe_below_source_placement():
    placement = SubtitlePlacementService(
        _config(preferred_relation="above_source")
    ).compute_placement(_region(y=0.01, height=0.1), 0)

    assert placement == SubtitlePlacement(alignment=8, x=672, y=138)


def test_left_and_right_edge_regions_keep_horizontal_center_in_bounds():
    service = SubtitlePlacementService(_config())

    left = service.compute_placement(_region(x=0.0, width=0.05), 0)
    right = service.compute_placement(_region(x=0.95, width=0.05), 0)

    assert 0 <= left.x <= 1920
    assert 0 <= right.x <= 1920


def test_overlapping_events_stack_without_overlapping_source_region():
    service = SubtitlePlacementService(_config())
    subtitles = [_segment("first", 1.0, 3.0), _segment("second", 1.0, 3.0)]
    region = _region(y=0.88, height=0.1, start_time=1.0, end_time=3.0)

    placements = service.calculate_placements([region], subtitles)

    assert [(placement.alignment, placement.x, placement.y) for placement in placements] == [
        (2, 672, 930),
        (2, 672, 880),
    ]


def test_long_two_line_subtitle_uses_configured_safe_height():
    placement = SubtitlePlacementService(
        _config(estimated_height_px=100)
    ).compute_placement(_region(y=0.8, height=0.1), 0)

    assert placement == SubtitlePlacement(alignment=2, x=672, y=844)


def test_mock_subtitle_region_detector_returns_region_for_each_frame():
    frames = [
        Frame(timestamp=1.5, image_path="frame-1.jpg"),
        Frame(timestamp=4.0, image_path="frame-2.jpg"),
    ]

    regions = MockSubtitleRegionDetectorProvider().detect_regions(frames)

    assert [region.to_dict() for region in regions] == [
        {
            "start_time": 1.5,
            "end_time": 3.5,
            "x": 0.1,
            "y": 0.8,
            "width": 0.8,
            "height": 0.08,
            "confidence": 0.95,
        },
        {
            "start_time": 4.0,
            "end_time": 6.0,
            "x": 0.1,
            "y": 0.8,
            "width": 0.8,
            "height": 0.08,
            "confidence": 0.95,
        },
    ]


def test_detect_subtitle_region_use_case_delegates_to_provider():
    frames = [Frame(timestamp=1.0, image_path="frame.jpg")]
    regions = [_region()]
    provider = Mock()
    provider.detect_regions.return_value = regions

    result = DetectSubtitleRegionUseCase(provider).execute(frames)

    assert result == regions
    provider.detect_regions.assert_called_once_with(frames)


def test_calculate_subtitle_placement_use_case_delegates_to_service():
    regions = [_region()]
    subtitles = [_segment()]
    placements = [SubtitlePlacement(alignment=2, margin_v=30)]
    service = Mock()
    service.calculate_placements.return_value = placements

    result = CalculateSubtitlePlacementUseCase(service).execute(regions, subtitles)

    assert result == placements
    service.calculate_placements.assert_called_once_with(regions, subtitles)


def test_ass_generator_applies_placement_margins_alignment_and_position_tags():
    segment = _segment()
    placement = SubtitlePlacement(
        alignment=8, x=100, y=200, margin_l=11, margin_r=12, margin_v=13
    )

    content = ASSGenerator.generate(
        [segment], SubtitleStyle(alignment=2), placements=[placement]
    )

    assert (
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,11,12,13,,"
        r"{\an8\pos(100,200)}text-s1"
    ) in content


def test_ass_generator_omits_alignment_tag_when_placement_matches_style():
    content = ASSGenerator.generate(
        [_segment()],
        SubtitleStyle(alignment=2),
        placements=[SubtitlePlacement(alignment=2, x=100, y=200)],
    )

    assert r"{\pos(100,200)}text-s1" in content
    assert r"\an2" not in content


def test_ass_generator_without_placement_uses_default_dialogue_format():
    content = ASSGenerator.generate([_segment()], SubtitleStyle(alignment=2))

    assert "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,text-s1" in content

def test_rendered_bbox_for_long_subtitle_stays_inside_safe_zone():
    service = SubtitlePlacementService(_config())
    segment = _segment()
    segment.translated_text = "Một câu phụ đề rất dài nhưng vẫn phải nằm hoàn toàn trong vùng hiển thị an toàn."

    placement = service.calculate_placements([], [segment], SubtitleStyle(font_size=32))[0]

    assert placement.rendered_bbox["left"] >= service._safe_left
    assert placement.rendered_bbox["right"] <= service._safe_right
    assert placement.rendered_bbox["top"] >= service._safe_top
    assert placement.rendered_bbox["bottom"] <= service._safe_bottom

def test_replace_mode_generates_source_mask_before_subtitle_text():
    service = SubtitlePlacementService(_config(placement_mode="replace_source_subtitle"))
    segment = _segment()
    placement = service.calculate_placements([_region()], [segment])[0]

    content = ASSGenerator.generate([segment], SubtitleStyle(), [placement])

    assert placement.mask_bbox is not None
    assert r"\p1\1c&H000000&\1a&H00&" in content
    assert "Dialogue: 1," in content

def test_replace_mode_mask_covers_source_and_rendered_text():
    service = SubtitlePlacementService(
        _config(placement_mode="replace_source_subtitle")
    )
    segment = _segment()
    segment.translated_text = (
        "Một câu phụ đề dài hơn đáng kể so với vùng chữ nguồn được phát hiện."
    )
    region = _region(x=0.45, width=0.1, y=0.75, height=0.05)

    placement = service.calculate_placements(
        [region], [segment], SubtitleStyle(font_size=24)
    )[0]

    mask = placement.mask_bbox
    rendered = placement.rendered_bbox
    assert mask["left"] <= rendered["left"] - service.config.padding_px
    assert mask["right"] >= rendered["right"] + service.config.padding_px
    assert mask["top"] <= rendered["top"] - service.config.padding_px
    assert mask["bottom"] >= rendered["bottom"] + service.config.padding_px

def test_replace_mode_masks_fallback_caption_region():
    placement = SubtitlePlacementService(
        _config(placement_mode="replace_source_subtitle")
    ).compute_placement(None, 0)

    assert placement.mask_bbox is not None
    assert placement.x is not None
    assert placement.y is not None

def test_quality_report_fails_for_clipped_rendered_bbox():
    service = SubtitlePlacementService(_config())
    placement = SubtitlePlacement(
        alignment=2,
        x=10,
        y=100,
        rendered_bbox={"left": -10, "top": 50, "right": 100, "bottom": 100},
    )

    report = service.build_quality_report([], [_segment()], [placement], SubtitleStyle())

    assert report["clipping_violations"] == [1]
    assert report["final_placement_verdict"] == "FAIL"

def test_quality_report_fails_when_avoid_bbox_overlaps_source():
    service = SubtitlePlacementService(_config(placement_mode="avoid_source_subtitle"))
    region = _region()
    placement = SubtitlePlacement(
        alignment=5,
        x=672,
        y=540,
        rendered_bbox={"left": 500, "top": 450, "right": 800, "bottom": 620},
    )

    report = service.build_quality_report(
        [region], [_segment()], [placement], SubtitleStyle()
    )

    assert report["source_overlap_count"] == 1
    assert report["final_placement_verdict"] == "FAIL"

def test_feature_config_defaults_subtitle_placement_to_false():
    assert FeatureConfig().subtitle_placement is False
    assert FeatureConfig(subtitle_placement=True).to_dict()["subtitle_placement"] is True

def test_config_loader_loads_subtitle_placement_defaults():
    from src.interfaces.config.config_loader import ConfigLoader

    placement = ConfigLoader().load().subtitle_placement

    assert placement.padding_px == 24
    assert placement.preferred_relation == "override"
    assert placement.placement_mode == "replace_source_subtitle"
    assert placement.fallback_alignment == 2
    assert placement.fallback_margin_v == 15
    assert placement.min_region_confidence == 0.70
    assert placement.video_width == 1920
    assert placement.video_height == 1080

def test_provider_registry_and_factory_create_mock_region_detector():
    from src.infrastructure.factory.provider_factory import ProviderFactory
    from src.infrastructure.registry.provider_registry import ProviderRegistry

    registry_provider = ProviderRegistry().create(
        "subtitle_region_detector", "mock", {}
    )
    factory_provider = ProviderFactory(
        config={
            "defaults": {"subtitle_region_detector": "mock"},
            "providers": {"subtitle_region_detector": {"mock": {}}},
        }
    ).create_subtitle_region_detector_provider()

    assert registry_provider.__class__.__name__ == "MockSubtitleRegionDetectorProvider"
    assert factory_provider.__class__.__name__ == "MockSubtitleRegionDetectorProvider"


def test_placement_config_default_is_replace_source_subtitle():
    from src.domain.entities.placement_config import PlacementConfig

    # Domain dataclass should default to replace_source_subtitle even when
    # constructed with only the required positional fields (no placement_mode key).
    # This guards against yaml-less or minimal-config paths falling back to avoid.
    cfg = PlacementConfig(
        padding_px=24,
        preferred_relation="override",
        fallback_alignment=2,
        fallback_margin_v=15,
        min_region_confidence=0.70,
        video_width=1920,
        video_height=1080,
    )
    assert cfg.placement_mode == "replace_source_subtitle"
    assert cfg.preferred_relation == "override"


def test_default_config_replaces_source_subtitle():
    # Verify default config has replace_source_subtitle + override
    from src.interfaces.config.config_loader import ConfigLoader
    from src.interfaces.config.app_config import AppConfig

    loader = ConfigLoader()
    config = loader.load(overrides={})
    assert config.subtitle_placement.placement_mode == "replace_source_subtitle"
    assert config.subtitle_placement.preferred_relation == "override"

    app_config = AppConfig()
    assert app_config.subtitle_placement.placement_mode == "replace_source_subtitle"
    assert app_config.subtitle_placement.preferred_relation == "override"
