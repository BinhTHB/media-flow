import os
from typing import Any, Dict
from dataclasses import asdict

from src.interfaces.config.config_loader import ConfigLoader
from src.interfaces.config.app_config import AppConfig
from src.application.orchestrator.job_orchestrator import JobOrchestrator
from src.application.orchestrator.orchestrator_dependencies import OrchestratorDependencies
from src.application.services.subtitle_placement_service import SubtitlePlacementService
from src.application.services.subtitle_segmentation_service import SubtitleSegmentationService
from src.application.services.final_timeline_builder import FinalTimelineBuilder
from src.application.services.final_render_manifest_service import FinalRenderManifestService
from src.application.use_cases.calculate_subtitle_placement import CalculateSubtitlePlacementUseCase
from src.application.use_cases.detect_onscreen_text import DetectOnscreenTextUseCase
from src.application.use_cases.detect_subtitle_region import DetectSubtitleRegionUseCase
from src.application.use_cases.extract_audio import ExtractAudioUseCase
from src.application.use_cases.get_video_duration import GetVideoDurationUseCase
from src.application.use_cases.extract_frames import ExtractFramesUseCase
from src.application.use_cases.generate_subtitle import GenerateSubtitleUseCase
from src.application.use_cases.generate_dubbed_audio import GenerateDubbedAudioUseCase
from src.application.use_cases.mux_dubbed_video import MuxDubbedVideoUseCase
from src.application.use_cases.generate_subtitle_placement_debug import (
    GenerateSubtitlePlacementDebugUseCase,
)
from src.application.use_cases.merge_segments import MergeSegmentsUseCase
from src.application.use_cases.normalize_segments import NormalizeSegmentsUseCase
from src.application.use_cases.filter_ocr_segments import FilterOCRSegmentsUseCase
from src.application.use_cases.group_segments import GroupSegmentsUseCase
from src.application.use_cases.render_video import RenderVideoUseCase
from src.application.use_cases.subtitle_segmentation import SubtitleSegmentationUseCase
from src.application.use_cases.transcribe_speech import TranscribeSpeechUseCase
from src.application.use_cases.translate_segments import TranslateSegmentsUseCase
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.domain.entities.subtitle_style import SubtitleStyle
from src.domain.entities.subtitle_style_preset import SubtitleStylePreset
from src.infrastructure.artifacts.artifact_manager import ArtifactManager
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.registry.provider_registry import ProviderRegistry
from src.interfaces.application.media_flow_application import MediaFlowApplication
from src.interfaces.composition.application_context import ApplicationContext

class ApplicationBuilder:
    def __init__(
        self,
        config_path: str | None = None,
        subtitle_styles_path: str | None = None,
        output_dir: str = "output",
        video_provider: str | None = None,
        speech_provider: str | None = None,
        translation_provider: str | None = None,
        subtitle_provider: str | None = None,
        ocr_provider: str | None = None,
        subtitle_region_detector_provider: str | None = None,
        frame_provider: str | None = None,
        whisper_model: str | None = None,
        target_language: str = "vi",
        frame_interval: float = 1.0,
        keep_audio: bool = False,
        subtitle_format: str = "srt",
        subtitle_style: str = "default",
        font: str | None = None,
        font_size: int | None = None,
        outline: float | None = None,
        shadow: float | None = None,
        margin_v: int | None = None,
        max_chars_per_line: int | None = None,
        max_lines_per_subtitle: int | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        no_deduplicate: bool = False,
        disable_segmentation: bool = False,
        prefer_ocr_over_speech: bool = False,
        enable_vietnamese_text_correction: bool = False,
        ocr_device: str = "gpu",
        placement_mode: str | None = None,
        resource_profile: str = "balanced",
        ffmpeg_encoder: str = "auto",
        ffmpeg_gpu_decode: bool = False,
        ffmpeg_nvenc_cq: int = 23,
        ffmpeg_nvenc_preset: str = "p5",
        use_v21_pipeline: bool = False,
        gemini_cache: str = "on",
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.config_path = config_path or os.path.join(base_dir, "config", "providers.yaml")
        self.subtitle_styles_path = subtitle_styles_path or os.path.join(base_dir, "config", "subtitle_styles.yaml")
        self.output_dir = output_dir
        self.video_provider = video_provider
        self.speech_provider = speech_provider
        self.translation_provider = translation_provider
        self.subtitle_provider = subtitle_provider
        self.ocr_provider = ocr_provider
        self.subtitle_region_detector_provider = getattr(
            args, "subtitle_region_detector", None
        ) if 'args' in locals() else subtitle_region_detector_provider
        self.frame_provider = frame_provider
        self.whisper_model = whisper_model
        self.target_language = target_language
        self.frame_interval = frame_interval
        self.keep_audio = keep_audio
        self.subtitle_format = subtitle_format
        self.subtitle_style_name = subtitle_style
        self.style_overrides = {
            "font_family": font,
            "font_size": font_size,
            "outline_width": outline,
            "shadow": shadow,
            "margin_v": margin_v,
        }
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.no_deduplicate = no_deduplicate
        self.disable_segmentation = disable_segmentation
        self.prefer_ocr_over_speech = prefer_ocr_over_speech
        self.enable_vietnamese_text_correction = enable_vietnamese_text_correction
        self.ocr_device = ocr_device
        self.placement_mode = placement_mode
        self.resource_profile = resource_profile
        self.ffmpeg_encoder = ffmpeg_encoder
        self.ffmpeg_gpu_decode = ffmpeg_gpu_decode
        self.ffmpeg_nvenc_cq = ffmpeg_nvenc_cq
        self.ffmpeg_nvenc_preset = ffmpeg_nvenc_preset
        self.use_v21_pipeline = use_v21_pipeline
        self.gemini_cache = gemini_cache

    @classmethod
    def from_cli_args(cls, args: Any) -> "ApplicationBuilder":
        return cls(
            output_dir=args.output_dir,
            video_provider=args.video_provider,
            speech_provider=args.speech_provider,
            translation_provider=args.translation_provider,
            subtitle_provider=args.subtitle_provider,
            ocr_provider=args.ocr_provider,
            subtitle_region_detector_provider=getattr(
                args, "subtitle_region_detector", None
            ),
            frame_provider=args.frame_provider,
            whisper_model=args.whisper_model,
            target_language=args.target_lang,
            frame_interval=args.frame_interval,
            keep_audio=args.keep_audio,
            subtitle_format=args.subtitle_format,
            subtitle_style=args.subtitle_style,
            font=args.font,
            font_size=args.font_size,
            outline=args.outline,
            shadow=args.shadow,
            margin_v=args.margin_v,
            max_chars_per_line=args.max_chars_per_line,
            max_lines_per_subtitle=args.max_lines_per_subtitle,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            no_deduplicate=args.no_deduplicate,
            disable_segmentation=args.disable_segmentation,
            prefer_ocr_over_speech=args.prefer_ocr_over_speech,
            enable_vietnamese_text_correction=(
                getattr(args, "enable_vietnamese_text_correction", False)
            ),
            ocr_device=getattr(args, "ocr_device", "gpu"),
            placement_mode=getattr(args, "placement_mode", None),
            resource_profile=getattr(args, "resource_profile", "balanced"),
            ffmpeg_encoder=getattr(args, "ffmpeg_encoder", "auto"),
            ffmpeg_gpu_decode=getattr(args, "ffmpeg_gpu_decode", False),
            ffmpeg_nvenc_cq=getattr(args, "ffmpeg_nvenc_cq", 23),
            ffmpeg_nvenc_preset=getattr(args, "ffmpeg_nvenc_preset", "p5"),
            use_v21_pipeline=getattr(args, "use_v21_pipeline", False),
            gemini_cache=getattr(args, "gemini_cache", "on"),
        )

    def build(self) -> MediaFlowApplication:
        loader = ConfigLoader(
            providers_path=self.config_path,
            subtitle_styles_path=self.subtitle_styles_path,
        )
        quality_overrides = {
            key: value
            for key, value in {
                "max_chars_per_line": self.max_chars_per_line,
                "max_lines_per_subtitle": self.max_lines_per_subtitle,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
            }.items()
            if value is not None
        }
        if self.no_deduplicate:
            quality_overrides["deduplicate"] = False
        if self.prefer_ocr_over_speech:
            quality_overrides["prefer_speech_over_ocr"] = False
        if self.enable_vietnamese_text_correction:
            quality_overrides["enable_vietnamese_text_correction"] = True
        overrides = {
            "subtitle_quality": quality_overrides,
            "gemini": {"cache_enabled": self.gemini_cache == "on"},
        }
        overrides["resource_policy"] = {"profile": self.resource_profile}
        overrides["ffmpeg"] = {
            "encoder": self.ffmpeg_encoder,
            "decoder": "cuda" if self.ffmpeg_gpu_decode else "auto",
            "use_gpu_decode": self.ffmpeg_gpu_decode,
            "nvenc_cq": self.ffmpeg_nvenc_cq,
            "nvenc_preset": self.ffmpeg_nvenc_preset,
        }
        if self.placement_mode:
            overrides["subtitle_placement"] = {"placement_mode": self.placement_mode}
        app_config = loader.load(overrides=overrides)

        registry = ProviderRegistry()
        config_dict = {
            "defaults": app_config.defaults,
            "providers": app_config.providers,
        }
        factory = ProviderFactory(
            config=config_dict,
            registry=registry,
            gemini_config=app_config.gemini,
            gemini_api_key=loader.gemini_api_key,
            provider_fallback_config=app_config.provider_fallback,
        )
        artifact_manager = ArtifactManager(base_dir=self.output_dir)
        context = ApplicationContext(
            config=app_config,
            provider_registry=registry,
            provider_factory=factory,
            artifact_manager=artifact_manager,
        )

        video_provider = factory.create_video_provider(
            name=self.video_provider,
            overrides={
                "output_dir": self.output_dir,
                "ffmpeg_config": app_config.ffmpeg,
                "resource_policy": app_config.resource_policy,
                "ocr_device": self.ocr_device,
                "asr_device": app_config.providers.get("speech", {}).get(
                    self.speech_provider or app_config.defaults.get("speech", "mock"),
                    {},
                ).get("device", "not_used"),
            },
        )
        speech_provider = factory.create_speech_provider(
            name=self.speech_provider,
            overrides={"model_size": self.whisper_model},
        )
        translation_provider = factory.create_translation_provider(
            name=self.translation_provider,
            overrides={"target_language": self.target_language},
        )
        context.gemini_runtime = factory.gemini_runtime
        subtitle_provider = factory.create_subtitle_provider(name=self.subtitle_provider)
        ocr_provider = factory.create_ocr_provider(
            name=self.ocr_provider,
            overrides={"use_gpu": self.ocr_device == "gpu"},
        )
        region_detector_provider = factory.create_subtitle_region_detector_provider(
            name=self.subtitle_region_detector_provider,
            overrides={
                "use_gpu": self.ocr_device == "gpu",
                "use_v21_pipeline": self.use_v21_pipeline,
            },
        )
        debug_provider = factory.create_subtitle_placement_debug_provider()
        frame_extraction_provider = factory.create_frame_extraction_provider(name=self.frame_provider)
        tts_provider = factory.create_tts_provider()
        dubbing_provider = factory.create_dubbing_provider()

        quality_config = app_config.subtitle_quality
        segmentation_service = SubtitleSegmentationService(
            config=quality_config,
            enabled=not self.disable_segmentation,
        )
        placement_service = SubtitlePlacementService(app_config.subtitle_placement)
        subtitle_style = self._resolve_subtitle_style(app_config)
        dependencies = OrchestratorDependencies(
            extract_audio_uc=ExtractAudioUseCase(video_provider),
            get_video_duration_uc=GetVideoDurationUseCase(video_provider),
            transcribe_speech_uc=TranscribeSpeechUseCase(speech_provider),
            detect_onscreen_text_uc=DetectOnscreenTextUseCase(ocr_provider),
            translate_segments_uc=TranslateSegmentsUseCase(translation_provider),
            generate_subtitle_uc=GenerateSubtitleUseCase(subtitle_provider),
            render_video_uc=RenderVideoUseCase(video_provider),
            extract_frames_uc=ExtractFramesUseCase(frame_extraction_provider),
            merge_segments_uc=MergeSegmentsUseCase(),
            normalize_segments_uc=NormalizeSegmentsUseCase(quality_config),
            filter_ocr_segments_uc=FilterOCRSegmentsUseCase(quality_config),
            group_segments_uc=GroupSegmentsUseCase(quality_config),
            subtitle_segmentation_uc=SubtitleSegmentationUseCase(segmentation_service),
            detect_subtitle_region_uc=DetectSubtitleRegionUseCase(
                region_detector_provider
            ),
            calculate_subtitle_placement_uc=CalculateSubtitlePlacementUseCase(
                placement_service
            ),
            artifact_provider=artifact_manager,
            generate_subtitle_placement_debug_uc=GenerateSubtitlePlacementDebugUseCase(
                debug_provider
            ),
            generate_dubbed_audio_uc=GenerateDubbedAudioUseCase(
                tts_provider, dubbing_provider, artifact_manager
            ),
            mux_dubbed_video_uc=MuxDubbedVideoUseCase(
                dubbing_provider, artifact_manager
            ),
            final_timeline_builder=FinalTimelineBuilder(),
            final_render_manifest_service=FinalRenderManifestService(),
            keep_audio=self.keep_audio,
            frame_interval=self.frame_interval,
            subtitle_format=self.subtitle_format,
            subtitle_style=subtitle_style,
        )
        return MediaFlowApplication(JobOrchestrator(dependencies), context)

    def _resolve_subtitle_style(self, app_config: AppConfig) -> SubtitleStyle:
        clean_overrides = {k: v for k, v in self.style_overrides.items() if v is not None}
        preset_dict = app_config.styles.get(self.subtitle_style_name, {})
        merged_style = dict(SubtitleStylePreset.presets()["default"])
        merged_style.update(preset_dict)
        merged_style.update(clean_overrides)
        return SubtitleStyle(**merged_style)
