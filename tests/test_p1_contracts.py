from __future__ import annotations
import json
import pytest
from src.application.process_video import ProcessVideoUseCase
from src.application.request import ProcessingRequest
from src.core.errors import ConfigurationError, ValidationError
from src.features.audio.models import AudioResult
from src.features.audio.pipeline import AudioRequest
from src.features.visual.pipeline import VisualRequest, VisualResult
from src.interfaces.composition.config import ProviderConfig, load_provider_config
from src.interfaces.jobs.store import JobStore


def test_provider_precedence_runtime_env_config_default():
    config = ProviderConfig(defaults={"translation": "gemini"})
    assert config.resolve("translation", "openrouter", {"MEDIAFLOW_TRANSLATION_PROVIDER": "env"}) == "openrouter"
    assert config.resolve("translation", "auto", {"MEDIAFLOW_TRANSLATION_PROVIDER": "env"}) == "env"
    assert config.resolve("translation", "auto", {}) == "gemini"
    assert ProviderConfig().resolve("translation", "auto", {}) == "auto"


def test_provider_config_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text("defaults:\n  speech: auto\n  speech: gemini\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Duplicate"): load_provider_config(path)


def test_job_store_recovery_and_secret_redaction(tmp_path):
    store = JobStore(tmp_path / "jobs")
    store.save_state("job-1", {"id": "job-1", "status": "RUNNING", "api_key": "secret"})
    store.append_event("job-1", {"type": "started", "token": "secret"})
    store.save_result("job-1", {"status": "SUCCESS"})
    restored = JobStore(tmp_path / "jobs").get("job-1")
    assert restored["api_key"] == restored["events"][0]["token"] == "[REDACTED]"
    assert restored["result"]["status"] == "SUCCESS"
    assert JobStore(tmp_path / "jobs").list()[0]["id"] == "job-1"
    assert not list((tmp_path / "jobs").rglob("*.tmp"))


def test_use_case_preflight_creates_no_artifacts(tmp_path):
    request = ProcessingRequest(job_id="invalid", input_video=str(tmp_path / "missing.mp4"), output_video=str(tmp_path / "new/out.mp4"), audio=AudioRequest(enabled=False), visual=VisualRequest(enabled=False))
    with pytest.raises(ValidationError): ProcessVideoUseCase(workspace_base_dir=str(tmp_path / "jobs")).execute(request)
    assert not (tmp_path / "jobs").exists()
    assert not (tmp_path / "new").exists()


class _Audio:
    def __init__(self, status): self.status = status
    def process(self, request, job_workspace_dir): return AudioResult(enabled=True, status=self.status, warnings=["audio warning"])
class _Visual:
    def __init__(self, status): self.status = status
    def process(self, request, job_workspace_dir, output_video_path): return VisualResult(enabled=True, status=self.status, rendered_video_path="visual.mp4", warnings=["visual warning"])
class _Video:
    def mux_audio_video(self, video_no_audio_path, audio_source_path, output_path): return output_path
    def probe_video(self, path): return {"duration": 1.0, "fps": 30.0, "width": 10, "height": 10}


@pytest.mark.parametrize(("audio", "visual", "expected"), [("SUCCESS","SUCCESS","SUCCESS"),("PARTIAL","SUCCESS","PARTIAL"),("SUCCESS","PARTIAL","PARTIAL"),("FAILED","SUCCESS","FAILED"),("SUCCESS","BLOCKED","BLOCKED")])
def test_status_aggregation(tmp_path, audio, visual, expected):
    source = tmp_path / "source.mp4"; source.write_bytes(b"x")
    request = ProcessingRequest(job_id=f"status-{audio}-{visual}", input_video=str(source), output_video=str(tmp_path / "out.mp4"), audio=AudioRequest(enabled=True), visual=VisualRequest(enabled=True), overwrite=True)
    result = ProcessVideoUseCase(_Audio(audio), _Visual(visual), _Video(), str(tmp_path / "jobs")).execute(request)
    assert result.status == expected
    assert result.warnings == ["audio warning", "visual warning"]
    assert json.loads((tmp_path / "jobs" / request.job_id / "result.json").read_text())["status"] == expected


def test_independent_audio_and_visual_run_together(tmp_path):
    import threading

    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    gate = threading.Barrier(2, timeout=2)
    seen: list[str] = []

    class Audio:
        def process(self, request, job_workspace_dir):
            seen.append("audio")
            gate.wait()
            return AudioResult(enabled=True, status="SUCCESS", dubbed_audio_path=str(source))

    class Visual:
        def process(self, request, job_workspace_dir, output_video_path):
            seen.append("visual")
            gate.wait()
            return VisualResult(enabled=True, status="SUCCESS", rendered_video_path=str(source))

    request = ProcessingRequest(
        job_id="parallel-pipelines",
        input_video=str(source),
        output_video=str(tmp_path / "out.mp4"),
        audio=AudioRequest(enabled=True),
        visual=VisualRequest(enabled=True, use_speech_subtitles=False),
        overwrite=True,
    )
    ProcessVideoUseCase(Audio(), Visual(), _Video(), str(tmp_path / "jobs")).execute(request)
    assert set(seen) == {"audio", "visual"}


def test_reup_speech_subtitles_are_ready_before_the_visual_pass(tmp_path):
    from src.core.models.speech_segment import SpeechSegment

    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    segment = SpeechSegment("s1", 0.0, 1.2, "hello", translated_text="xin chào")
    seen = []

    class Audio:
        def process(self, request, job_workspace_dir):
            seen.append("audio")
            return AudioResult(enabled=True, status="SUCCESS", segments=[segment], dubbed_audio_path=str(source))

    class Visual:
        def process(self, request, job_workspace_dir, output_video_path):
            seen.append(("visual", request.segments))
            return VisualResult(enabled=True, status="SUCCESS", rendered_video_path=str(source))

    request = ProcessingRequest(
        job_id="speech-first",
        input_video=str(source),
        output_video=str(tmp_path / "out.mp4"),
        audio=AudioRequest(enabled=True),
        visual=VisualRequest(enabled=True, use_speech_subtitles=True),
        overwrite=True,
    )
    ProcessVideoUseCase(Audio(), Visual(), _Video(), str(tmp_path / "jobs")).execute(request)
    assert seen[0] == "audio"
    assert seen[1][0] == "visual"
    assert seen[1][1][0]["translated_text"] == "xin chào"


def test_empty_burned_in_line_takes_the_spoken_translation():
    from src.core.models.bbox import BBox
    from src.core.models.subtitle_event import SubtitleEvent
    from src.features.visual.pipeline import VisualPipeline

    event = SubtitleEvent(
        event_id="e1",
        track_id="t1",
        start_time=1.0,
        end_time=2.0,
        anchor_bbox=BBox(0, 0, 10, 10, frame_width=100, frame_height=100),
        replacement_status="OCR_EMPTY",
    )
    ready = SubtitleEvent(
        event_id="e2",
        track_id="t2",
        start_time=3.0,
        end_time=4.0,
        anchor_bbox=BBox(0, 0, 10, 10, frame_width=100, frame_height=100),
        source_text="đã dịch",
        translated_text="đã dịch",
        replacement_status="READY",
    )
    pipeline = object.__new__(VisualPipeline)
    pipeline._fill_empty_events_from_speech(
        [event, ready],
        [{"start_time": 1.1, "end_time": 1.8, "translated_text": "lời mới"}],
    )
    assert event.translated_text == "lời mới"
    assert event.replacement_status == "READY"
    assert event.overlay_applied is True
    assert ready.translated_text == "đã dịch"


def test_speech_captions_are_created_even_when_ocr_already_read_a_line():
    from src.core.models.bbox import BBox
    from src.core.models.subtitle_event import SubtitleEvent
    from src.features.visual.pipeline import VisualPipeline

    ready = SubtitleEvent(
        event_id="burned",
        track_id="t0",
        start_time=1.0,
        end_time=2.0,
        anchor_bbox=BBox(80, 600, 640, 680, frame_width=720, frame_height=720),
        source_text="这是原字幕",
        translated_text="Đây là phụ đề gốc",
        replacement_status="READY",
        overlay_applied=True,
        blur_applied=True,
    )
    pipeline = object.__new__(VisualPipeline)
    pipeline.subtitle_processor = None
    events = pipeline._captions_from_speech(
        [ready],
        [{"start_time": 0.0, "end_time": 3.4, "translated_text": "Một người chưa từng đến trường"}],
        "vi",
    )
    assert [event.translated_text for event in events] == ["Một người chưa từng đến trường"]
    assert events[0].split_reason == "no_source_sub"
    assert events[0].overlay_applied is True
    assert events[0].anchor_bbox.y1 >= int(720 * 0.8)


def test_brand_chip_does_not_count_as_a_readable_caption():
    from src.core.models.bbox import BBox
    from src.core.models.subtitle_event import SubtitleEvent
    from src.features.visual.pipeline import VisualPipeline

    brand = SubtitleEvent(
        event_id="logo",
        track_id="t0",
        start_time=9.5,
        end_time=10.0,
        anchor_bbox=BBox(261, 682, 331, 708, frame_width=720, frame_height=720),
        source_text="OLEDPRO",
        translated_text="OLEDPRO",
        replacement_status="TRANSLATION_UNCHANGED",
    )
    pipeline = object.__new__(VisualPipeline)
    pipeline.subtitle_processor = None
    events = pipeline._captions_from_speech(
        [brand],
        [
            {"start_time": 0.0, "end_time": 3.4, "translated_text": "Một người chưa từng đến trường"},
            {"start_time": 4.0, "end_time": 8.0, "translated_text": "Chiếc xe do chú tự tay sáng chế"},
        ],
        "vi",
    )
    assert [event.translated_text for event in events] == [
        "Một người chưa từng đến trường",
        "Chiếc xe do chú tự tay sáng chế",
    ]
    assert all(event.split_reason == "no_source_sub" for event in events)
    assert all(event.overlay_applied for event in events)
    assert all(event.anchor_bbox.y1 >= int(720 * 0.8) for event in events)


def test_speech_captions_use_source_text_when_translation_is_empty():
    from unittest.mock import MagicMock
    from src.features.visual.pipeline import VisualPipeline

    pipeline = object.__new__(VisualPipeline)
    processor = MagicMock()
    processor.translation_service.translate_texts.side_effect = AssertionError(
        "speech captions must not call live translation"
    )
    pipeline.subtitle_processor = processor
    events = pipeline._captions_from_speech(
        [],
        [{
            "start_time": 0.0,
            "end_time": 1.2,
            "source_text": "Câu gốc tiếng Việt",
            "translated_text": "",
            "spoken_text": "",
        }],
        "vi",
    )
    assert len(events) == 1
    assert events[0].translated_text == "Câu gốc tiếng Việt"
    assert events[0].replacement_status == "READY"
    assert events[0].overlay_applied is True
    assert events[0].split_reason == "no_source_sub"
    processor.translation_service.translate_texts.assert_not_called()


def test_blank_burn_in_tracks_become_bottom_captions_from_speech():
    from src.core.models.bbox import BBox
    from src.core.models.subtitle_event import SubtitleEvent
    from src.features.visual.pipeline import VisualPipeline

    sliver = SubtitleEvent(
        event_id="edge",
        track_id="t1",
        start_time=15.9,
        end_time=16.5,
        anchor_bbox=BBox(20, 713, 586, 720, frame_width=720, frame_height=720),
        replacement_status="OCR_EMPTY",
    )
    pipeline = object.__new__(VisualPipeline)
    pipeline.subtitle_processor = None
    events = pipeline._captions_from_speech(
        [sliver],
        [{"start_time": 0.0, "end_time": 3.4, "translated_text": "5 giờ sáng, một tiếng kêu"}],
        "vi",
    )
    assert len(events) == 1
    assert events[0].split_reason == "no_source_sub"
    assert events[0].translated_text == "5 giờ sáng, một tiếng kêu"
    assert events[0].overlay_applied is True
    assert events[0].anchor_bbox.y1 >= int(720 * 0.8)
    assert events[0].anchor_bbox.height > 40
