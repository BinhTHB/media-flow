import os
from unittest.mock import Mock
import pytest
from src.domain.entities.video_job import VideoJob
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.job_status import JobStatus
from src.domain.entities.segment import Segment
from src.application.orchestrator.job_orchestrator import JobOrchestrator
from src.application.orchestrator.orchestrator_dependencies import OrchestratorDependencies
from src.application.use_cases.extract_audio import ExtractAudioUseCase
from src.application.use_cases.filter_ocr_segments import FilterOCRSegmentsUseCase
from src.application.use_cases.get_video_duration import GetVideoDurationUseCase
from src.application.use_cases.group_segments import GroupSegmentsUseCase
from src.application.use_cases.transcribe_speech import TranscribeSpeechUseCase
from src.application.use_cases.detect_onscreen_text import DetectOnscreenTextUseCase
from src.application.use_cases.translate_segments import TranslateSegmentsUseCase
from src.application.use_cases.generate_subtitle import GenerateSubtitleUseCase
from src.application.use_cases.render_video import RenderVideoUseCase
from src.application.use_cases.detect_subtitle_region import DetectSubtitleRegionUseCase
from src.application.use_cases.calculate_subtitle_placement import CalculateSubtitlePlacementUseCase
from src.domain.entities.placement_config import PlacementConfig
from src.shared.exceptions import ValidationError

from src.infrastructure.providers.speech.mock_speech_provider import MockSpeechProvider
from src.infrastructure.providers.ocr.mock_ocr_provider import MockOCRProvider
from src.infrastructure.providers.translation.mock_translation_provider import MockTranslationProvider
from src.infrastructure.providers.subtitle.mock_subtitle_provider import MockSubtitleProvider
from src.infrastructure.providers.video.mock_video_provider import MockVideoProvider
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.infrastructure.providers.subtitle.mock_subtitle_region_detector_provider import MockSubtitleRegionDetectorProvider
from src.application.services.subtitle_placement_service import SubtitlePlacementService

from src.application.use_cases.extract_frames import ExtractFramesUseCase
from src.application.use_cases.merge_segments import MergeSegmentsUseCase
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService
from src.infrastructure.providers.video.mock_frame_extraction_provider import MockFrameExtractionProvider

@pytest.fixture
def orchestrator() -> JobOrchestrator:
    speech_provider = MockSpeechProvider()
    ocr_provider = MockOCRProvider()
    translation_provider = MockTranslationProvider()
    subtitle_provider = MockSubtitleProvider()
    video_provider = MockVideoProvider()
    frame_extraction_provider = MockFrameExtractionProvider()

    extract_audio_uc = ExtractAudioUseCase(video_provider)
    transcribe_speech_uc = TranscribeSpeechUseCase(speech_provider)
    detect_onscreen_text_uc = DetectOnscreenTextUseCase(ocr_provider)
    translate_segments_uc = TranslateSegmentsUseCase(translation_provider)
    generate_subtitle_uc = GenerateSubtitleUseCase(subtitle_provider)
    render_video_uc = RenderVideoUseCase(video_provider)
    extract_frames_uc = ExtractFramesUseCase(frame_extraction_provider)
    merge_segments_uc = MergeSegmentsUseCase()
    from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
    normalize_segments_uc = NormalizeSegmentsUseCase()
    subtitle_segmentation_uc = SubtitleSegmentationUseCase(SubtitleSegmentationService())
    
    artifact_manager = ArtifactManager(base_dir="output")

    region_detector_provider = MockSubtitleRegionDetectorProvider()
    placement_service = SubtitlePlacementService(
        PlacementConfig(
            padding_px=20,
            preferred_relation="below_source",
            fallback_alignment=2,
            fallback_margin_v=30,
            min_region_confidence=0.8,
            video_width=1920,
            video_height=1080,
        )
    )
    
    detect_subtitle_region_uc = DetectSubtitleRegionUseCase(region_detector_provider)
    calculate_subtitle_placement_uc = CalculateSubtitlePlacementUseCase(placement_service)

    deps = OrchestratorDependencies(
        extract_audio_uc=extract_audio_uc,
        filter_ocr_segments_uc=FilterOCRSegmentsUseCase(),
        group_segments_uc=GroupSegmentsUseCase(),
        get_video_duration_uc=GetVideoDurationUseCase(video_provider),
        transcribe_speech_uc=transcribe_speech_uc,
        detect_onscreen_text_uc=detect_onscreen_text_uc,
        translate_segments_uc=translate_segments_uc,
        generate_subtitle_uc=generate_subtitle_uc,
        render_video_uc=render_video_uc,
        extract_frames_uc=extract_frames_uc,
        merge_segments_uc=merge_segments_uc,
        normalize_segments_uc=normalize_segments_uc,
        subtitle_segmentation_uc=subtitle_segmentation_uc,
        detect_subtitle_region_uc=detect_subtitle_region_uc,
        calculate_subtitle_placement_uc=calculate_subtitle_placement_uc,
        artifact_provider=artifact_manager,
    )
    return JobOrchestrator(deps)

def test_orchestrator_only_stt(orchestrator):
    features = FeatureConfig(
        speech_to_text=True,
        ocr=False,
        translate=False,
        subtitle=False,
        render_video=False,
    )
    job = VideoJob(
        job_id="stt_only",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=features,
    )
    result = orchestrator.execute(job)
    assert len(result.segments) == 2
    assert result.segments[0].source == "speech"
    assert result.segments[0].translated_text == ""
    assert result.status == JobStatus.COMPLETED
    assert result.subtitle_content == ""
    report = orchestrator.artifact_provider.read_json(
        os.path.join(
            orchestrator.artifact_provider.debug_dir(job.job_id),
            "speech_quality_report.json",
        )
    )
    assert report["asr_segment_count"] == 2
    assert report["speech_translation_input_count"] == 2
    assert report["speech_translation_output_count"] == 2
    vietnamese_report = orchestrator.artifact_provider.read_json(
        os.path.join(
            orchestrator.artifact_provider.debug_dir(job.job_id),
            "vietnamese_text_quality_report.json",
        )
    )
    assert vietnamese_report["mode"] == "report-first"
    assert len(vietnamese_report["captions"]) == len(result.segments)

def test_orchestrator_applies_enabled_vietnamese_corrections_downstream(orchestrator):
    orchestrator.normalize_segments_uc.config.enable_vietnamese_text_correction = True
    orchestrator.transcribe_speech_uc.speech_provider.transcribe = lambda *_: [
        Segment(
            id="speech_0",
            start=0.0,
            end=3.0,
            text="Viên cảnh xác bắt ngờ.",
            source="speech",
            confidence=0.9,
        )
    ]
    job = VideoJob(
        job_id="vietnamese_correction_enabled",
        input_video="test.mp4",
        features=FeatureConfig(speech_to_text=True, translate=False, subtitle=True),
    )

    result = orchestrator.execute(job)

    assert result.segments[0].text == "Viên cảnh sát bất ngờ."
    assert "Viên cảnh sát bất ngờ." in result.subtitle_content
    report = orchestrator.artifact_provider.read_json(
        os.path.join(
            orchestrator.artifact_provider.debug_dir(job.job_id),
            "vietnamese_text_quality_report.json",
        )
    )
    assert report["corrections_applied"] == 2
    assert report["captions"][0]["original_text"] == "Viên cảnh xác bắt ngờ."
    assert report["captions"][0]["actual_action"] == "correct"

def test_orchestrator_only_ocr(orchestrator):
    features = FeatureConfig(
        speech_to_text=False,
        ocr=True,
        translate=False,
        subtitle=False,
        render_video=False,
    )
    job = VideoJob(
        job_id="ocr_only",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=features,
    )
    result = orchestrator.execute(job)
    assert len(result.segments) == 1
    assert result.segments[0].source == "ocr"
    assert result.segments[0].translated_text == ""
    assert result.status == JobStatus.COMPLETED

def test_orchestrator_full_flow(orchestrator):
    features = FeatureConfig(
        speech_to_text=True,
        ocr=True,
        translate=True,
        subtitle=True,
        render_video=True,
    )
    job = VideoJob(
        job_id="full_flow",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=features,
    )
    result = orchestrator.execute(job)
    assert len(result.segments) == 3
    assert result.segments[0].source == "speech"
    assert result.segments[0].translated_text == "Chào mọi người, chào mừng đến với kênh của\ntôi."
    assert result.segments[1].source == "ocr"
    assert result.segments[2].source == "speech"
    assert result.subtitle_path == os.path.normpath("output/full_flow/subtitles/subtitle.srt")
    assert "00:00:00,000" in result.subtitle_content
    assert result.output_video_path == os.path.normpath("output/full_flow/videos/output.mp4")
    assert result.status == JobStatus.COMPLETED


def test_orchestrator_render_no_subtitle_raises(orchestrator):
    from src.shared.exceptions import ValidationError
    features = FeatureConfig(
        speech_to_text=True,
        ocr=False,
        translate=True,
        subtitle=False,
        render_video=True,
    )
    job = VideoJob(
        job_id="render_no_sub",
        input_video="test.mp4",
        source_language="zh",
        features=features,
    )
    with pytest.raises(ValidationError) as excinfo:
        orchestrator.execute(job)
    assert "Cannot render video without subtitles enabled" in str(excinfo.value)


def test_orchestrator_fails_source_leakage_quality_gate(orchestrator, monkeypatch):
    from src.application.services.source_text_leakage_detector import (
        SourceTextLeakageDetector,
    )

    monkeypatch.setattr(
        SourceTextLeakageDetector,
        "detect",
        lambda segments, source_language: [{"text": "你好"}],
    )
    job = VideoJob(
        job_id="source_leakage_gate",
        input_video="test.mp4",
        source_language="zh",
        features=FeatureConfig(
            speech_to_text=True,
            translate=True,
            subtitle=False,
            render_video=False,
        ),
    )

    with pytest.raises(ValidationError, match="Quality Gate Failed"):
        orchestrator.execute(job)

    metadata = orchestrator.artifact_provider.read_json(
        orchestrator.artifact_provider.debug_metadata_json_path(job.job_id)
    )
    assert metadata["quality_metrics"]["source_text_leakage_count"] == 1
    assert metadata["quality_metrics"]["quality_gate_status"] == "FAIL"

def _read_ass(orchestrator, job_id):
    with open(orchestrator.artifact_provider.subtitle_ass_path(job_id), encoding="utf-8") as f:
        return f.read()

def test_orchestrator_placement_disabled_preserves_legacy_ass(orchestrator):
    orchestrator.subtitle_format = "ass"
    job = VideoJob(
        job_id="placement_disabled",
        input_video="test.mp4",
        features=FeatureConfig(
            speech_to_text=True,
            translate=True,
            subtitle=True,
            subtitle_placement=False,
        ),
    )

    orchestrator.execute(job)

    content = _read_ass(orchestrator, job.job_id)
    assert r"\pos" not in content
    assert r"\an" not in content

def test_orchestrator_placement_enabled_adds_ass_position_tags(orchestrator):
    orchestrator.subtitle_format = "ass"
    job = VideoJob(
        job_id="placement_enabled",
        input_video="test.mp4",
        features=FeatureConfig(
            speech_to_text=True,
            translate=True,
            subtitle=True,
            subtitle_placement=True,
        ),
    )

    orchestrator.execute(job)

    content = _read_ass(orchestrator, job.job_id)
    assert r"\pos" in content
    assert r"\an" in content

def test_orchestrator_placement_without_regions_uses_global_style(orchestrator, monkeypatch):
    orchestrator.subtitle_format = "ass"
    monkeypatch.setattr(orchestrator.detect_subtitle_region_uc, "execute", lambda frames: [])
    job = VideoJob(
        job_id="placement_without_regions",
        input_video="test.mp4",
        features=FeatureConfig(
            speech_to_text=True,
            translate=True,
            subtitle=True,
            subtitle_placement=True,
        ),
    )

    orchestrator.execute(job)

    content = _read_ass(orchestrator, job.job_id)
    assert r"\pos" not in content
    assert r"\an" not in content


def test_orchestrator_skips_debug_artifacts_when_debug_is_disabled(orchestrator):
    orchestrator.generate_subtitle_placement_debug_uc = Mock()
    job = VideoJob(
        job_id="placement_debug_disabled",
        input_video="test.mp4",
        features=FeatureConfig(
            speech_to_text=True,
            translate=True,
            subtitle=True,
            subtitle_placement=True,
            subtitle_placement_debug=False,
        ),
    )

    orchestrator.execute(job)

    orchestrator.generate_subtitle_placement_debug_uc.execute.assert_not_called()

def test_orchestrator_persists_artifacts(orchestrator):
    job = VideoJob(
        job_id="test_artifact_persistence",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=FeatureConfig(
            speech_to_text=True,
            ocr=True,
            translate=True,
            subtitle=True,
            render_video=False,
        ),
    )
    orchestrator.execute(job)

    debug_dir = orchestrator.artifact_provider.debug_dir(job.job_id)
    raw_ocr_path = os.path.join(debug_dir, "raw_ocr.json")
    filtered_ocr_path = os.path.join(debug_dir, "filtered_ocr.json")
    grouping_provenance_path = os.path.join(debug_dir, "grouping_provenance.json")

    assert os.path.exists(raw_ocr_path)
    assert os.path.exists(filtered_ocr_path)
    assert os.path.exists(grouping_provenance_path)

    raw_ocr_data = orchestrator.artifact_provider.read_json(raw_ocr_path)
    assert "segments" in raw_ocr_data
    assert "dropped_tokens" in raw_ocr_data
    assert "drop_reasons" in raw_ocr_data
    assert "filtered_tokens" in raw_ocr_data

    filtered_ocr_data = orchestrator.artifact_provider.read_json(filtered_ocr_path)
    assert "segments" in filtered_ocr_data
    assert "dropped_tokens" in filtered_ocr_data
    assert "drop_reasons" in filtered_ocr_data
    assert "filtered_tokens" in filtered_ocr_data

    grouping_provenance_data = orchestrator.artifact_provider.read_json(grouping_provenance_path)
    assert "groups" in grouping_provenance_data
    assert "dropped_tokens" in grouping_provenance_data
    assert "filtered_tokens" in grouping_provenance_data

def test_orchestrator_content_quality_report(orchestrator):
    job = VideoJob(
        job_id="test_content_quality_report_job",
        input_video="test.mp4",
        source_language="zh",
        target_language="vi",
        features=FeatureConfig(
            speech_to_text=True,
            ocr=True,
            translate=True,
            subtitle=True,
            render_video=False,
        ),
    )
    orchestrator.execute(job)

    debug_dir = orchestrator.artifact_provider.debug_dir(job.job_id)
    report_path = os.path.join(debug_dir, "content_quality_report.json")
    assert os.path.exists(report_path)

    report_data = orchestrator.artifact_provider.read_json(report_path)
    assert "captions" in report_data
    assert len(report_data["captions"]) > 0
    
    caption = report_data["captions"][0]
    assert "caption_index" in caption
    assert "timestamp" in caption
    assert "original_text" in caption
    assert "score" in caption
    assert "findings" in caption
    assert "recommended_action" in caption
    assert "actual_action" in caption
    assert "reason" in caption

def test_orchestrator_report_only_keeps_low_quality_caption(orchestrator, monkeypatch):
    from src.domain.entities.segment import Segment
    controlled = [Segment(id="low", start=0.0, end=1.0, text="...", source="speech")]
    monkeypatch.setattr(orchestrator.subtitle_segmentation_uc, "execute", lambda *args, **kwargs: controlled)
    orchestrator.normalize_segments_uc.config.enable_content_quality_filtering = False
    job = VideoJob(job_id="cq_report_only", input_video="test.mp4")
    result = orchestrator.execute(job)
    report = orchestrator.artifact_provider.read_json(os.path.join(orchestrator.artifact_provider.debug_dir(job.job_id), "content_quality_report.json"))
    assert [segment.id for segment in result.segments] == ["low"]
    assert report["captions"][0]["findings"] == ["placeholder_only"]
    assert report["captions"][0]["actual_action"] == "keep"


def test_orchestrator_filtering_drops_low_quality_caption(orchestrator, monkeypatch):
    from src.domain.entities.segment import Segment
    controlled = [Segment(id="low", start=0.0, end=1.0, text="...", source="speech")]
    monkeypatch.setattr(orchestrator.subtitle_segmentation_uc, "execute", lambda *args, **kwargs: controlled)
    orchestrator.normalize_segments_uc.config.enable_content_quality_filtering = True
    job = VideoJob(job_id="cq_filtering", input_video="test.mp4")
    result = orchestrator.execute(job)
    report = orchestrator.artifact_provider.read_json(os.path.join(orchestrator.artifact_provider.debug_dir(job.job_id), "content_quality_report.json"))
    assert result.segments == []
    assert report["captions"][0]["actual_action"] == "drop"