"""Tests for runtime optimizations: single-pass H.264 encoding, scene-cut
caching, and persisted per-stage timings."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.contracts import VideoUtilPort
from src.infrastructure.stage_cache import CachedSceneCuts, JsonStageCache
from src.infrastructure.video.ffmpeg import FFmpegVideoAdapter, is_web_compatible_video


def _blank_frame(width: int, height: int, value: int = 96) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_ffmpeg_frame_writer_produces_web_compatible_video(tmp_path) -> None:
    out = tmp_path / "rendered.mp4"
    writer = FFmpegVideoAdapter().open_encoded_frame_writer(str(out), fps=25.0, width=64, height=48)
    try:
        for _ in range(30):
            writer.write(_blank_frame(64, 48))
    finally:
        writer.release()

    assert out.is_file() and out.stat().st_size > 0
    # Single-pass H.264 output must already be web-compatible: the runner's
    # ensure_web_compatible_video re-encode becomes a no-op for this file.
    assert is_web_compatible_video(out)


def test_port_default_frame_writer_is_not_implemented() -> None:
    class BareVideoUtil(VideoUtilPort):
        def extract_audio(self, video_path: str, output_audio_path: str) -> str:
            return video_path

        def mux_audio_video(self, video_no_audio_path: str, audio_source_path: str, output_path: str) -> str:
            return output_path

        def probe_video(self, video_path: str) -> dict:
            return {}

    with pytest.raises(NotImplementedError):
        BareVideoUtil().open_encoded_frame_writer("out.mp4", 25.0, 64, 48)


class _CountingSceneDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect_cuts(self, video_path: str) -> list[int]:
        self.calls += 1
        return [10, 240]


def test_cached_scene_cuts_hit_cache_on_second_call(tmp_path) -> None:
    detector = _CountingSceneDetector()
    cache = JsonStageCache(tmp_path, {"identity": "test"})
    cached = CachedSceneCuts(detector, cache)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    assert cached.detect_cuts(str(video)) == [10, 240]
    assert cached.detect_cuts(str(video)) == [10, 240]
    assert detector.calls == 1


def test_processing_result_persists_stage_timings() -> None:
    from src.application.result import ProcessingResult

    result = ProcessingResult(
        job_id="j1",
        status="SUCCESS",
        input_video_path="in.mp4",
        output_video_path="out.mp4",
        stage_timings={"visual": {"rendering": 12.5}, "audio": {}},
    )
    payload = result.to_dict()
    assert payload["stage_timings"]["visual"]["rendering"] == 12.5

    default_payload = ProcessingResult(
        job_id="j2", status="SUCCESS", input_video_path="a", output_video_path="b"
    ).to_dict()
    assert default_payload["stage_timings"] == {}


def test_render_loop_releases_writer_and_capture_on_mid_render_failure(monkeypatch, tmp_path) -> None:
    # Regression: a mid-render exception used to skip cap.release() and
    # writer.release(), leaking the ffmpeg encoder process and the capture
    # handle — both hold file locks on Windows until interpreter exit.
    import cv2

    class _FakeCap:
        def __init__(self, _path):
            self._frames = [np.full((48, 64, 3), 8, dtype=np.uint8) for _ in range(3)]
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_FRAME_COUNT: 3,
            }.get(prop, 0)

        def read(self):
            return (True, self._frames.pop(0)) if self._frames else (False, None)

        def release(self):
            self.released = True

    caps: list[_FakeCap] = []
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda path: (caps.append(_FakeCap(path)) or caps[-1])
    )

    class _RecordingWriter:
        released = False

        def write(self, _frame) -> None:
            return None

        def release(self) -> None:
            self.released = True

    writer = _RecordingWriter()

    from src.features.visual.pipeline import VisualPipeline, VisualRequest

    class _NoDetectionsDetector:
        def detect_video(self, _video_path, stride=1):
            return []

    pipeline = VisualPipeline(detector=_NoDetectionsDetector())
    monkeypatch.setattr(VisualPipeline, "_open_frame_writer", lambda self, *a, **k: writer)

    def _crash_on_second_frame(self, frame, frame_index, request, *args, **kwargs):
        if frame_index >= 1:
            raise RuntimeError("mid-render crash")
        return frame

    monkeypatch.setattr(VisualPipeline, "_apply_subtitle_actions", _crash_on_second_frame)

    request = VisualRequest(
        video_path="fake.mp4",
        subtitle_enabled=False,
        logo_enabled=False,
        watermark_enabled=False,
        other_text_enabled=False,
    )

    with pytest.raises(RuntimeError, match="mid-render crash"):
        pipeline.process(request, str(tmp_path), str(tmp_path / "out.mp4"))

    assert writer.released
    assert caps and caps[-1].released
