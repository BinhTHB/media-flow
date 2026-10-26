from pathlib import Path
from unittest.mock import MagicMock


from src.application.request import ProcessingRequest
from src.core.models.speech_segment import SpeechSegment
from src.features.audio.models import AudioRequest
from src.features.audio.pipeline import AudioPipeline
from src.interfaces.jobs.runner import JobOptions, JobRunner
from src.interfaces.web.server import build_dubbing_plan


def test_speech_segment_from_web_dict():
    web_segment = {
        "id": "seg_01",
        "start": 1.25,
        "end": 3.5,
        "source_text": "Hello world",
        "target_text": "Xin chào thế giới",
        "speaker": "S02",
    }
    seg = SpeechSegment.from_dict(web_segment)
    assert seg.segment_id == "seg_01"
    assert seg.start_time == 1.25
    assert seg.end_time == 3.5
    assert seg.source_text == "Hello world"
    assert seg.translated_text == "Xin chào thế giới"
    assert seg.spoken_text == "Xin chào thế giới"
    assert seg.speaker_id == "S02"


def test_processing_request_deserializes_segments_and_speaker_voices():
    data = {
        "input_video": "input/test.mp4",
        "output_video": "outputs/test.mp4",
        "audio": {
            "enabled": True,
            "dubbing_enabled": True,
            "segments": [
                {
                    "id": "seg_custom_1",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "Original text",
                    "target_text": "Bản dịch tùy biến",
                    "speaker_id": "S01",
                }
            ],
            "speaker_voices": {
                "S01": "elevenlabs:EXAVITQu4vr4xnSDxMaL",
                "S02": "elevenlabs:nPczCjzI2devNBz1zQrb",
            },
        },
    }
    req = ProcessingRequest.from_dict(data)
    assert req.audio.segments is not None
    assert len(req.audio.segments) == 1
    assert req.audio.segments[0].segment_id == "seg_custom_1"
    assert req.audio.segments[0].spoken_text == "Bản dịch tùy biến"
    assert req.audio.speaker_voices["S02"] == "elevenlabs:nPczCjzI2devNBz1zQrb"


def test_audio_pipeline_skips_asr_when_segments_provided(tmp_path):
    mock_asr = MagicMock()
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = str(tmp_path / "clip.mp3")
    mock_composer = MagicMock()
    mock_composer.trim_tts_silence.return_value = tmp_path / "clip.wav"
    mock_composer.probe_duration.return_value = 1.0
    mock_composer.max_speedup = 1.3

    pipeline = AudioPipeline(
        asr_provider=mock_asr,
        tts_provider=mock_tts,
        composer=mock_composer,
    )

    custom_segments = [
        SpeechSegment(
            segment_id="seg_01",
            start_time=0.0,
            end_time=2.0,
            source_text="Test source",
            translated_text="Bản dịch",
            spoken_text="Bản dịch",
            speaker_id="S02",
        )
    ]

    audio_req = AudioRequest(
        enabled=True,
        dubbing_enabled=True,
        audio_path=str(tmp_path / "dummy.wav"),
        voice="default_voice",
        segments=custom_segments,
        speaker_voices={"S02": "voice_for_s02"},
    )

    result = pipeline.process(audio_req, str(tmp_path))

    # ASR should NEVER be called when segments are provided!
    mock_asr.transcribe.assert_not_called()

    # TTS should be called with S02's specific voice
    mock_tts.synthesize.assert_called_once_with(
        text="Bản dịch",
        output_path=str(tmp_path / "audio" / "tts" / "tts_seg_01.mp3"),
        voice="voice_for_s02",
    )
    assert result.status == "SUCCESS"


def test_visual_request_deserializes_segments_and_layout():

    data = {
        "input_video": "input/test.mp4",
        "output_video": "outputs/test.mp4",
        "visual": {
            "enabled": True,
            "subtitle_layout": "in-place",
            "hide_source_subtitle": False,
            "segments": [
                {
                    "id": "s1",
                    "start": 1.0,
                    "end": 2.5,
                    "source_text": "Hello",
                    "target_text": "Xin chào",
                }
            ],
        },
    }
    req = ProcessingRequest.from_dict(data)
    assert req.visual.subtitle_layout == "in-place"
    assert req.visual.hide_source_subtitle is False
    assert req.visual.segments is not None
    assert len(req.visual.segments) == 1
    assert req.visual.segments[0].translated_text == "Xin chào"


def test_job_runner_preflight_allows_segments_without_gemini(monkeypatch, tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy video")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("src.interfaces.jobs.runner.shutil.which", lambda name: name)
    monkeypatch.setattr(
        "src.interfaces.jobs.runner.JobRunner._require_dependency",
        lambda self, mod, pkg: MagicMock(),
    )

    runner = JobRunner()
    options = JobOptions(
        input_path=str(video),
        output_path=str(tmp_path / "out.mp4"),
        preset="translate-subtitle",
        device="cpu",
        yolo_weights=str(video),
        segments=[{"id": "s1", "start": 0, "end": 1, "target_text": "Sub"}],
        translation_provider="auto",
    )

    # When segments are present and already translated, OCR is NOT required.
    # Preflight should pass without ConfigurationError for GEMINI_API_KEY.
    runner._preflight(options)


def test_visual_pipeline_builds_events_from_segments():
    from src.core.models.bbox import BBox
    from src.core.models.detection import Detection
    from src.features.visual.pipeline import VisualPipeline

    pipeline = VisualPipeline()
    segments = [
        SpeechSegment(
            segment_id="seg_01",
            start_time=1.0,
            end_time=3.0,
            source_text="Source sentence",
            translated_text="Câu dịch tiếng Việt",
        ),
        SpeechSegment(
            segment_id="seg_02",
            start_time=4.0,
            end_time=6.0,
            source_text="No detection sentence",
            translated_text="Câu dịch không có box",
        ),
    ]

    mock_dets = [
        Detection(
            class_id=0,
            class_name="subtitle",
            confidence=0.92,
            bbox=BBox(100, 500, 400, 560, 1280, 720),
            frame_index=45,  # at 30 fps, 1.5s is frame 45
            timestamp_seconds=1.5,
        )
    ]

    events = pipeline._build_events_from_segments(
        segments=segments,
        detections=mock_dets,
        width=1280,
        height=720,
        fps=30.0,
        subtitle_layout="in-place",
    )

    assert len(events) == 2
    ev1, ev2 = events[0], events[1]

    # Event 1 matched YOLO detection in frame 45
    assert ev1.event_id == "seg_01"
    assert ev1.replacement_status == "READY"
    assert ev1.anchor_bbox.x1 == 100
    assert ev1.anchor_bbox.y1 == 500
    assert ev1.translated_text == "Câu dịch tiếng Việt"
    assert ev1.blur_applied is True
    assert ev1.overlay_applied is True

    # Event 2 has no detection: fallback to standard bottom cinema anchor
    assert ev2.event_id == "seg_02"
    assert ev2.replacement_status == "READY"
    assert ev2.anchor_bbox.y1 >= int(720 * 0.8)
    assert ev2.translated_text == "Câu dịch không có box"

    # Event with bottom layout: fallback to cinema bottom anchor
    events_bottom = pipeline._build_events_from_segments(
        segments=segments,
        detections=mock_dets,
        width=1280,
        height=720,
        fps=30.0,
        subtitle_layout="bottom",
    )
    assert events_bottom[1].anchor_bbox.y1 >= int(720 * 0.8)


def test_visual_pipeline_applies_subtitle_layout_inplace_and_bottom():
    import numpy as np
    from src.core.models.bbox import BBox
    from src.core.models.subtitle_event import SubtitleEvent
    from src.features.visual.pipeline import VisualPipeline, VisualRequest

    mock_proc = MagicMock()
    mock_proc.render_region_events.side_effect = lambda frame, events, region, **kwargs: "rendered_inplace"
    mock_proc.render_frame_events.side_effect = lambda frame, idx, events: "rendered_bottom"
    mock_proc.select_render_events.side_effect = lambda evs: evs

    pipeline = VisualPipeline(subtitle_processor=mock_proc)

    test_box = BBox(100, 500, 400, 560, 1280, 720)
    ev = SubtitleEvent(
        event_id="ev1",
        track_id="tr1",
        start_time=1.0,
        end_time=3.0,
        anchor_bbox=test_box,
        source_text="Hello",
        translated_text="Xin chào",
        replacement_status="READY",
        blur_applied=True,
        overlay_applied=True,
    )

    events_by_frame = {45: [ev]}
    dets_by_frame = {45: []}
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Test IN-PLACE layout: should route to render_region_events
    req_inplace = VisualRequest(
        enabled=True,
        subtitle_enabled=True,
        subtitle_mode="translate_replace",
        subtitle_layout="in-place",
    )
    pipeline._region_planner = MagicMock()
    pipeline._region_planner.region_for.return_value = test_box

    res_inplace = pipeline._apply_subtitle_actions(
        frame=frame,
        frame_index=45,
        request=req_inplace,
        detections_by_frame=dets_by_frame,
        events_by_frame=events_by_frame,
    )
    assert res_inplace == "rendered_inplace"
    mock_proc.render_region_events.assert_called_once()

    # 2. Test BOTTOM layout: should route to render_frame_events
    mock_proc.reset_mock()
    req_bottom = VisualRequest(
        enabled=True,
        subtitle_enabled=True,
        subtitle_mode="translate_replace",
        subtitle_layout="bottom",
    )
    res_bottom = pipeline._apply_subtitle_actions(
        frame=frame,
        frame_index=45,
        request=req_bottom,
        detections_by_frame=dets_by_frame,
        events_by_frame=events_by_frame,
    )
    assert res_bottom == "rendered_bottom"
    mock_proc.render_frame_events.assert_called_once()
    mock_proc.render_region_events.assert_not_called()


def test_visual_pipeline_extracts_bbox_from_segment_dict():
    from src.features.visual.pipeline import VisualPipeline

    pipeline = VisualPipeline()
    segments = [
        {
            "id": "seg_01",
            "start": 1.0,
            "end": 2.5,
            "source_text": "Original subtitle",
            "target_text": "Phụ đề dịch có box",
            "bbox": {
                "x1": 250,
                "y1": 580,
                "x2": 850,
                "y2": 640,
                "frame_width": 1280,
                "frame_height": 720,
            },
        }
    ]

    events = pipeline._build_events_from_segments(
        segments=segments,
        detections=[],  # No detector matches for this specific frame
        width=1280,
        height=720,
        fps=30.0,
        target_language="vi",
    )

    assert len(events) == 1
    ev = events[0]
    assert ev.anchor_bbox.x1 == 250
    assert ev.anchor_bbox.y1 == 580
    assert ev.anchor_bbox.x2 == 850
    assert ev.anchor_bbox.y2 == 640
    assert ev.replacement_status == "READY"


def test_build_dubbing_plan_attaches_yolo_bboxes(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from src.core.models.bbox import BBox
    from src.core.models.detection import Detection
    from src.interfaces.web import server

    mock_detector = MagicMock()
    fake_bbox = BBox(150, 600, 750, 660, 1280, 720)
    fake_det = Detection(
        class_id=0,
        class_name="subtitle",
        confidence=0.92,
        bbox=fake_bbox,
        frame_index=30,
        timestamp_seconds=1.0,
    )
    mock_detector.detect_video.return_value = [fake_det]

    monkeypatch.setattr(
        "src.infrastructure.detection.yolo.YoloDetectorAdapter",
        lambda *args, **kwargs: mock_detector,
    )

    # Mock asr provider to return a segment
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = [
        SpeechSegment(
            segment_id="s1",
            start_time=0.5,
            end_time=2.0,
            source_text="Hello",
            translated_text="Xin chào",
        )
    ]
    monkeypatch.setattr(
        "src.interfaces.composition.providers.build_asr_provider",
        lambda *args, **kwargs: mock_asr,
    )
    monkeypatch.setattr(
        "src.infrastructure.video.ffmpeg.extract_audio",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.infrastructure.video.ffmpeg.probe_video",
        lambda *args, **kwargs: {"duration": "10.0"},
    )
    monkeypatch.setattr(
        server,
        "dubbing_capabilities",
        lambda: {
            "ready": True,
            "providers": {
                "asr": {"available": True, "reason": ""},
                "translation": {"available": False, "reason": ""},
                "tts": {"available": True, "reason": ""},
                "composer": {"available": True, "reason": ""},
                "diarization": {"available": False, "reason": ""},
            },
        },
    )
    dummy_video = server.PROJECT_ROOT / "outputs" / "test_sample_dummy.mp4"
    dummy_video.parent.mkdir(parents=True, exist_ok=True)
    dummy_video.write_bytes(b"dummy")
    try:
        monkeypatch.setattr(
            server,
            "resolve_project_video",
            lambda *args, **kwargs: dummy_video,
        )

        plan = server.build_dubbing_plan({
            "input_path": "test_videos/sample_10s.mp4",
            "source_language": "en",
            "target_language": "vi",
            "voice_id": "vi-VN-HoaiMyNeural",
        })
    finally:
        dummy_video.unlink(missing_ok=True)

    assert plan["default_subtitle_bbox"] is not None
    assert plan["default_subtitle_bbox"]["x1"] == 150
    assert plan["default_subtitle_bbox"]["y1"] == 600
    assert len(plan["segments"]) == 1
    assert "bbox" in plan["segments"][0]
    assert plan["segments"][0]["bbox"]["x1"] == 150
    assert plan["segments"][0]["bbox"]["y1"] == 600


def test_speech_segment_preserves_bbox():
    raw = {
        "segment_id": "seg_bbox_test",
        "start": 1.5,
        "end": 3.0,
        "source_text": "Text in box",
        "translated_text": "Chu trong hop",
        "bbox": {
            "x1": 250.0,
            "y1": 780.0,
            "x2": 550.0,
            "y2": 840.0,
            "frame_width": 1920,
            "frame_height": 1080,
        },
    }
    seg = SpeechSegment.from_dict(raw)
    assert seg.bbox is not None
    assert seg.bbox["x1"] == 250.0
    assert seg.bbox["y1"] == 780.0
    assert seg.bbox["x2"] == 550.0
    assert seg.bbox["y2"] == 840.0

    d = seg.to_dict()
    assert "bbox" in d
    assert d["bbox"]["x1"] == 250.0


def test_visual_pipeline_builds_events_prioritizes_speech_segment_bbox():
    seg = SpeechSegment(
        segment_id="seg_priority",
        start_time=0.5,
        end_time=2.0,
        source_text="Original subtitle",
        translated_text="Phụ đề dịch",
        bbox={
            "x1": 300,
            "y1": 850,
            "x2": 700,
            "y2": 910,
            "frame_width": 1920,
            "frame_height": 1080,
        },
    )

    from src.features.visual.pipeline import VisualPipeline
    pipeline = VisualPipeline()
    events = pipeline._build_events_from_segments(
        segments=[seg],
        detections=[],
        width=1920,
        height=1080,
        fps=30.0,
        subtitle_layout="in-place",
    )

    assert len(events) == 1
    ev = events[0]
    assert ev.anchor_bbox.x1 == 300
    assert ev.anchor_bbox.y1 == 850
    assert ev.anchor_bbox.x2 == 700
    assert ev.anchor_bbox.y2 == 910


def test_build_job_options_maps_speaker_mode_and_ambience():
    from src.interfaces.web.server import build_job_options
    job_dict = {
        "id": "job_test_123",
        "input_path": "test_videos/sample.mp4",
        "output_path": "outputs/sample-out.mp4",
        "preset": "translate-video",
        "device": "auto",
        "overwrite": True,
        "profile": "quality",
        "source_language": "zh",
        "target_language": "vi",
        "voice_id": "elevenlabs:EXAVITQu4vr4xnSDxMaL",
        "asr_provider": "auto",
        "translation_provider": "auto",
        "tts_provider": "auto",
        "subtitle_layout": "temporal",
        "hide_source_subtitle": True,
        "segments": [{"id": "s1", "start": 0.0, "end": 2.0, "source_text": "hello", "target_text": "xin chao"}],
        "speaker_voices": {"S01": "elevenlabs:EXAVITQu4vr4xnSDxMaL"},
        "blur_extra": True,
        "speaker_mode": "auto",
        "preserve_ambience": True,
        "mix_original_volume": 0.18,
    }
    opts = build_job_options(job_dict)
    assert opts.speaker_mode == "auto"
    assert opts.preserve_ambience is True
    assert opts.mix_original_volume == 0.18
    assert opts.segments == job_dict["segments"]
    assert opts.speaker_voices == job_dict["speaker_voices"]
    assert opts.blur_extra is True


def test_job_runner_assigns_visual_segments_when_provided(tmp_path):
    from unittest.mock import patch
    dummy_in = tmp_path / "sample.mp4"
    dummy_in.write_bytes(b"dummy")
    custom_segments = [
        {"id": "seg_1", "start": 1.0, "end": 3.0, "source_text": "Ni hao", "target_text": "Xin chao ban"}
    ]
    opts = JobOptions(
        input_path=str(dummy_in),
        output_path=str(tmp_path / "test_out.mp4"),
        preset="translate-video",
        segments=custom_segments,
        preserve_ambience=True,
        mix_original_volume=0.20,
    )
    runner = JobRunner()
    with patch.object(runner, "_preflight", return_value="cpu"):
        normalized, request = runner.prepare(opts)
        # Visual request MUST receive custom segments so editing on UI is rendered to video
        assert request.visual.segments == custom_segments
        # Audio request MUST also receive parsed segments
        assert request.audio.segments is not None
        assert len(request.audio.segments) == 1
        assert request.audio.preserve_ambience is True
        assert request.audio.mix_original_volume == 0.20


def test_build_dubbing_plan_does_not_block_translate_subtitle_for_missing_tts():
    from unittest.mock import patch
    fake_capabilities = {
        "status": "partial",
        "ready": False,
        "mode": "hybrid-free",
        "providers": {
            "asr": {"provider": "groq", "available": True, "reason": "ok"},
            "translation": {"provider": "gemini", "available": True, "reason": "ok"},
            "tts": {"provider": None, "available": False, "reason": "Cần ELEVENLABS_API_KEY"},
            "composer": {"provider": "ffmpeg", "available": True, "reason": "ok"},
            "diarization": {"provider": None, "available": False, "reason": "no"},
        },
        "limits": {},
    }
    with patch("src.interfaces.web.server.dubbing_capabilities", return_value=fake_capabilities):
        with patch("src.interfaces.web.server.resolve_project_video", return_value=Path("test_videos/sample.mp4")):
            plan = build_dubbing_plan({
                "input_path": "test_videos/sample.mp4",
                "preset": "translate-subtitle",
                "source_language": "zh",
                "target_language": "vi",
            })
            # translate-subtitle only needs asr, translation, composer (not tts)
            assert plan["ready"] is True
            assert len(plan["blockers"]) == 0



