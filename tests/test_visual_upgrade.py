import numpy as np
import pytest
from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.infrastructure.stage_cache import JsonStageCache, CachedDetector, CachedOCR, CachedTranslation
from src.features.visual.subtitle.region_planner import SubtitleRegion, SubtitleRegionPlanner
from src.features.visual.actions.text_overlay import TextOverlayAction


def det(frame, box, confidence=.9):
    return Detection(0, "subtitle", confidence, BBox(*box, 1280, 720), frame, frame / 30)


def test_partial_detection_uses_temporal_full_width():
    full = det(0, (240, 610, 1040, 700))
    partial = det(1, (300, 620, 650, 685))
    planner = SubtitleRegionPlanner([full, full, full, partial])
    region = planner.region_for(1, partial.bbox)
    assert region.x2 >= 1040
    assert region.x1 <= 240


def test_sparse_fragment_region_cannot_override_stable_lane():
    anchor = BBox(276, 619, 591, 668, 1280, 720)
    stable = BBox(251, 613, 1014, 696, 1280, 720)
    sparse = BBox(277, 619, 591, 668, 1280, 720)
    planner = SubtitleRegionPlanner([])
    planner.regions = {
        0: [
            SubtitleRegion(scene=0, bbox=stable, observations=1073),
            SubtitleRegion(scene=0, bbox=sparse, observations=4),
        ]
    }

    region = planner.region_for(450, anchor)

    assert region.x1 <= stable.x1
    assert region.x2 >= stable.x2
    assert region.y1 <= stable.y1
    assert region.y2 >= stable.y2


def test_scene_cut_does_not_carry_previous_region():
    old = det(0, (100, 610, 1180, 700))
    new = det(10, (400, 610, 800, 700))
    planner = SubtitleRegionPlanner([old, new], [10])
    assert planner.region_for(10, new.bbox).width == 400


def test_top_and_bottom_and_side_lanes_stay_separate():
    rows = [det(0, (30, 30, 400, 80)), det(0, (800, 30, 1200, 80)),
            det(0, (200, 620, 1000, 700))]
    planner = SubtitleRegionPlanner(rows)
    assert len(planner.regions[0]) == 3
    assert planner.region_for(0, rows[0].bbox).y2 == 80


def test_local_overlay_preserves_unrelated_pixels_and_hides_source():
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    frame[40:80, 50:400] = 255
    box = BBox(50, 40, 400, 80, 1280, 720)
    out = TextOverlayAction("Caption", anchor_to_bbox=True, bg_color=(10, 12, 16, 255)).execute(frame, box)
    assert np.array_equal(out[300:], frame[300:])
    assert int(out[45:75, 55:100].max()) < 30


def test_detector_cache_invalidates_content_and_identity(tmp_path):
    class Provider:
        calls = 0
        def detect_video(self, video_path, stride=1):
            self.calls += 1
            return [det(1, (20.123456, 40, 300, 80))]
    video = tmp_path / "video"
    video.write_bytes(b"one")
    provider = Provider()
    cached = CachedDetector(provider, JsonStageCache(tmp_path / "cache", {"weights": "a"}))
    first = cached.detect_video(str(video))
    second = cached.detect_video(str(video))
    assert provider.calls == 1
    assert second[0].bbox.x1 == first[0].bbox.x1
    video.write_bytes(b"two")
    cached.detect_video(str(video))
    assert provider.calls == 2
    CachedDetector(provider, JsonStageCache(tmp_path / "cache", {"weights": "b"})).detect_video(str(video))
    assert provider.calls == 3


def test_translation_cache_keeps_batch_context_and_language(tmp_path):
    class Provider:
        calls = 0
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            self.calls += 1
            return [target_language + text for text in texts]
    provider = Provider()
    cached = CachedTranslation(provider, JsonStageCache(tmp_path, {}))
    cached.translate_texts(["a", "b"])
    cached.translate_texts(["a", "b"])
    cached.translate_texts(["b", "a"])
    cached.translate_texts(["a", "b"], target_language="en")
    assert provider.calls == 3


def test_bad_batch_is_not_cached(tmp_path):
    class Provider:
        def translate_texts(self, **kwargs):
            return ["one"]
    cache = JsonStageCache(tmp_path, {})
    with pytest.raises(ValueError):
        CachedTranslation(Provider(), cache).translate_texts(["one", "two"])
    assert not list(tmp_path.rglob("*.json"))


def test_corrupted_cache_recomputes(tmp_path):
    cache = JsonStageCache(tmp_path, {})
    cache.put("test", "input", ["good"])
    path = cache._path("test", "input")
    path.write_text('{"payload":["bad"],"checksum":"wrong"}')
    assert cache.get("test", "input") is None


def test_ocr_pixel_change_invalidates(tmp_path):
    class Provider:
        calls = 0
        def read_crops(self, crops):
            self.calls += 1
            return ["text"] * len(crops)
    provider = Provider()
    cached = CachedOCR(provider, JsonStageCache(tmp_path, {}))
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    cached.read_crops([crop])
    cached.read_crops([crop.copy()])
    crop[0, 0] = 255
    cached.read_crops([crop])
    assert provider.calls == 2
