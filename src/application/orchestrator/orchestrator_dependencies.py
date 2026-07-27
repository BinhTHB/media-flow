from dataclasses import dataclass
from typing import Optional
from src.application.ports.artifact_provider import ArtifactProvider
from src.application.use_cases.extract_audio import ExtractAudioUseCase
from src.application.use_cases.get_video_duration import GetVideoDurationUseCase
from src.application.use_cases.transcribe_speech import TranscribeSpeechUseCase
from src.application.use_cases.detect_onscreen_text import DetectOnscreenTextUseCase
from src.application.use_cases.translate_segments import TranslateSegmentsUseCase
from src.application.use_cases.generate_subtitle import GenerateSubtitleUseCase
from src.application.use_cases.render_video import RenderVideoUseCase
from src.application.use_cases.extract_frames import ExtractFramesUseCase
from src.application.use_cases.merge_segments import MergeSegmentsUseCase
from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
from src.application.use_cases.filter_ocr_segments import FilterOCRSegmentsUseCase
from src.application.use_cases.group_segments import GroupSegmentsUseCase
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.application.use_cases.detect_subtitle_region import DetectSubtitleRegionUseCase
from src.application.use_cases.calculate_subtitle_placement import CalculateSubtitlePlacementUseCase
from src.application.use_cases.generate_subtitle_placement_debug import GenerateSubtitlePlacementDebugUseCase
from src.application.use_cases.generate_dubbed_audio import GenerateDubbedAudioUseCase
from src.application.use_cases.mux_dubbed_video import MuxDubbedVideoUseCase
from src.domain.entities.subtitle_style import SubtitleStyle
from src.application.services.final_timeline_builder import FinalTimelineBuilder
from src.application.services.final_render_manifest_service import FinalRenderManifestService
from src.application.services.mask_authority import DecisionStore

@dataclass
class OrchestratorDependencies:
    extract_audio_uc: ExtractAudioUseCase
    get_video_duration_uc: GetVideoDurationUseCase
    transcribe_speech_uc: TranscribeSpeechUseCase
    detect_onscreen_text_uc: DetectOnscreenTextUseCase
    translate_segments_uc: TranslateSegmentsUseCase
    generate_subtitle_uc: GenerateSubtitleUseCase
    render_video_uc: RenderVideoUseCase
    extract_frames_uc: ExtractFramesUseCase
    merge_segments_uc: MergeSegmentsUseCase
    normalize_segments_uc: NormalizeSegmentsUseCase
    filter_ocr_segments_uc: FilterOCRSegmentsUseCase
    group_segments_uc: GroupSegmentsUseCase
    subtitle_segmentation_uc: SubtitleSegmentationUseCase
    detect_subtitle_region_uc: DetectSubtitleRegionUseCase
    calculate_subtitle_placement_uc: CalculateSubtitlePlacementUseCase
    artifact_provider: ArtifactProvider
    generate_subtitle_placement_debug_uc: Optional[GenerateSubtitlePlacementDebugUseCase] = None
    generate_dubbed_audio_uc: Optional[GenerateDubbedAudioUseCase] = None
    mux_dubbed_video_uc: Optional[MuxDubbedVideoUseCase] = None
    final_timeline_builder: Optional[FinalTimelineBuilder] = None
    final_render_manifest_service: Optional[FinalRenderManifestService] = None
    keep_audio: bool = False
    frame_interval: float = 1.0
    subtitle_format: str = "srt"
    subtitle_style: Optional[SubtitleStyle] = None
    mask_decision_store: Optional[DecisionStore] = None
