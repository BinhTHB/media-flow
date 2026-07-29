from typing import Any, Callable, Dict, List

class ProviderRegistry:
    def __init__(self):
        self._factories: Dict[str, Dict[str, Callable[[Dict[str, Any]], Any]]] = {}
        self._classes: Dict[str, Dict[str, Any]] = {}
        register_default_providers(self)

    def register(self, category: str, name: str, factory_or_class: Any) -> None:
        if callable(factory_or_class) and not isinstance(factory_or_class, type):
            factory_fn = factory_or_class
            # store factory itself as the class in case get is called
            self._classes.setdefault(category, {})[name] = factory_or_class
        else:
            # It's a class type
            self._classes.setdefault(category, {})[name] = factory_or_class
            def default_factory(options: Dict[str, Any]) -> Any:
                try:
                    return factory_or_class(**options)
                except TypeError:
                    # Fallback for classes that do not accept these options in constructor
                    return factory_or_class()
            factory_fn = default_factory
            
        self._factories.setdefault(category, {})[name] = factory_fn

    def get(self, category: str, name: str) -> Any:
        try:
            return self._classes[category][name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {category}.{name}") from exc

    def create(self, category: str, name: str, options: Dict[str, Any]) -> Any:
        try:
            factory_fn = self._factories[category][name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider factory: {category}.{name}") from exc
        return factory_fn(options)

    def categories(self) -> List[str]:
        return sorted(self._factories.keys())

    def names(self, category: str) -> List[str]:
        return sorted(self._factories.get(category, {}).keys())



def _create_hybrid_gemini_ocr(options):
    from src.infrastructure.benchmark.hybrid_subtitle.config import DEFAULT_GEMINI_MODEL
    from src.infrastructure.benchmark.hybrid_subtitle.gemini_verifier import GeminiTrackVerifier
    from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
    from src.infrastructure.providers.ocr.hybrid_gemini_ocr_provider import HybridGeminiOCRProvider

    device = options.get("device", "cpu")
    if "use_gpu" in options:
        device = "cuda" if options["use_gpu"] else "cpu"

    runtime = options.get("runtime")
    model = options.get("gemini_model", DEFAULT_GEMINI_MODEL)
    verifier = GeminiTrackVerifier(runtime, model=model) if runtime else None
    proposal = YoloTextProposalProvider(
        weights_path=options.get("weights", "best.pt"),
        conf=options.get("conf", 0.05),
        iou=options.get("iou", 0.70),
        imgsz=options.get("imgsz", 1024),
        device=device,
    )
    return HybridGeminiOCRProvider(
        proposal_provider=proposal,
        semantic_verifier=verifier,
        min_confidence=float(options.get("min_confidence", 0.40)),
        crop_batch_classifier=None,
    )


def _create_hybrid_yolo(options):
    from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
        HybridSubtitleRegionDetectorProvider,
    )
    from src.infrastructure.benchmark.hybrid_subtitle.config import DEFAULT_GEMINI_MODEL
    from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider

    device = options.get("device", "cpu")
    if "use_gpu" in options:
        device = "cuda" if options["use_gpu"] else "cpu"

    runtime = options.get("runtime")
    crop_batch_classifier = None
    if runtime is not None:
        from src.infrastructure.benchmark.hybrid_subtitle.crop_batch_classifier import (
            CropBatchClassifier,
        )
        model = options.get("gemini_model", DEFAULT_GEMINI_MODEL)
        crop_batch_classifier = CropBatchClassifier(
            runtime,
            model=model,
            budget_limit=3,
            artifact_dir=options.get("crop_classifier_artifact_dir"),
            job_id=str(options.get("job_id", "")),
            run_id=str(options.get("run_id", "")),
        )
    proposal_provider = YoloTextProposalProvider(
        weights_path=options.get("weights", "best.pt"),
        conf=options.get("conf", 0.05),
        iou=options.get("iou", 0.70),
        imgsz=options.get("imgsz", 1024),
        device=device,
    )
    return HybridSubtitleRegionDetectorProvider(
        weights=options.get("weights", "best.pt"),
        conf=options.get("conf", 0.05),
        iou=options.get("iou", 0.70),
        imgsz=options.get("imgsz", 1024),
        min_confidence=float(options.get("min_confidence", 0.40)),
        band_gate_threshold=float(options.get("band_gate_threshold", 0.68)),
        static_variance_threshold=float(
            options.get("static_variance_threshold", 827.12)
        ),
        device=device,
        crop_batch_classifier=crop_batch_classifier,
        proposal_provider=proposal_provider,
    )


def register_default_providers(registry: ProviderRegistry) -> None:
    from src.infrastructure.providers.ocr.mock_ocr_provider import MockOCRProvider
    from src.infrastructure.providers.ocr.gemini_multistage_ocr_provider import GeminiMultiStageOCRProvider
    from src.infrastructure.providers.ocr.gemini_vision_ocr_provider import GeminiVisionOCRProvider
    from src.infrastructure.providers.ocr.gemini_video_direct_provider import GeminiDirectVideoOCRProvider
    from src.infrastructure.providers.speech.faster_whisper_speech_provider import FasterWhisperSpeechProvider
    from src.infrastructure.providers.speech.gemini_audio_speech_provider import GeminiAudioSpeechProvider
    from src.infrastructure.providers.speech.mock_speech_provider import MockSpeechProvider
    from src.infrastructure.providers.storage.mock_storage_provider import MockStorageProvider
    from src.infrastructure.providers.subtitle.mock_subtitle_provider import MockSubtitleProvider
    from src.infrastructure.providers.subtitle.ass_subtitle_provider import ASSSubtitleProvider
    from src.infrastructure.providers.subtitle.mock_subtitle_region_detector_provider import MockSubtitleRegionDetectorProvider
    from src.infrastructure.providers.subtitle.pillow_subtitle_placement_debug_provider import PillowSubtitlePlacementDebugProvider
    from src.infrastructure.providers.video.ffmpeg_video_provider import FFmpegVideoProvider
    from src.infrastructure.providers.video.mock_video_provider import MockVideoProvider
    from src.infrastructure.providers.video.ffmpeg_frame_extraction_provider import FFmpegFrameExtractionProvider
    from src.infrastructure.providers.video.mock_frame_extraction_provider import MockFrameExtractionProvider
    from src.infrastructure.providers.translation.mock_translation_provider import MockTranslationProvider
    from src.infrastructure.providers.translation.openai_translation_provider import OpenAITranslationProvider
    from src.infrastructure.providers.translation.gemini_translation_provider import GeminiTranslationProvider
    from src.infrastructure.providers.tts.edge_tts_provider import EdgeTTSProvider
    from src.infrastructure.providers.tts.ffmpeg_dubbing_provider import FFmpegDubbingProvider

    # video
    registry.register("video", "mock", MockVideoProvider)
    registry.register(
        "video", "ffmpeg",
        lambda opt: FFmpegVideoProvider(
            output_dir=opt.get("output_dir", "output"),
            ffmpeg_config=opt.get("ffmpeg_config"),
            resource_policy=opt.get("resource_policy"),
            ocr_device=opt.get("ocr_device", "not_used"),
            asr_device=opt.get("asr_device", "not_used"),
        ),
    )

    # speech
    registry.register("speech", "mock", MockSpeechProvider)
    registry.register("speech", "faster-whisper", lambda opt: FasterWhisperSpeechProvider(
        model_size=opt.get("model_size", "base"),
        device=opt.get("device", "cpu"),
        compute_type=opt.get("compute_type", "int8"),
    ))
    registry.register("speech", "gemini_audio", lambda opt: GeminiAudioSpeechProvider(
        runtime=opt.get("runtime"),
        model=opt.get("model", "gemini-3.5-flash-lite"),
        escalation_model=opt.get("escalation_model", "gemini-3.5-flash"),
        api_key=opt.get("api_key"),
        prompt_version=opt.get("prompt_version", "gemini_asr_v1"),
        schema_version=opt.get("schema_version", "gemini_asr_segments_v1"),
        full_audio_max_seconds=float(opt.get("full_audio_max_seconds", 300.0)),
        chunk_fallback_enabled=bool(opt.get("chunk_fallback_enabled", True)),
        chunk_duration_seconds=float(opt.get("chunk_duration_seconds", 60.0)),
        chunk_overlap_seconds=float(opt.get("chunk_overlap_seconds", 2.0)),
        escalation_enabled=bool(opt.get("escalation_enabled", False)),
        max_escalation_requests=int(opt.get("max_escalation_requests", 2)),
        inline_payload_safety_limit_bytes=int(opt.get("inline_payload_safety_limit_bytes", 19_000_000)),
    ))

    # translation
    registry.register("translation", "mock", MockTranslationProvider)
    registry.register("translation", "openai", lambda opt: OpenAITranslationProvider(
        model=opt.get("model", "gpt-4o-mini"),
        max_retries=int(opt.get("max_retries", 3)),
    ))
    registry.register("translation", "gemini", lambda opt: GeminiTranslationProvider(
        runtime=opt.get("runtime"),
        model=opt.get("model", "gemini-3.5-flash-lite"),
        max_retries=int(opt.get("max_retries", 3)),
    ))

    # subtitle
    registry.register("subtitle", "mock", MockSubtitleProvider)
    registry.register("subtitle", "ass", ASSSubtitleProvider)

    # subtitle_region_detector
    registry.register(
        "subtitle_region_detector", "mock", MockSubtitleRegionDetectorProvider
    )
    registry.register(
        "subtitle_region_detector",
        "hybrid_yolo",
        _create_hybrid_yolo,
    )

    # subtitle_placement_debug
    registry.register(
        "subtitle_placement_debug",
        "pillow",
        PillowSubtitlePlacementDebugProvider,
    )
    registry.register(
        "subtitle_placement_debug",
        "mock",
        PillowSubtitlePlacementDebugProvider,
    )

    # ocr
    registry.register("ocr", "mock", MockOCRProvider)
    registry.register("ocr", "hybrid_gemini", _create_hybrid_gemini_ocr)
    registry.register("ocr", "gemini_vision", lambda opt: GeminiVisionOCRProvider(
        runtime=opt.get("runtime"),
        model=opt.get("model", "gemini-3.5-flash-lite"),
        escalation_model=opt.get("escalation_model", "gemini-3.5-flash"),
        api_key=opt.get("api_key"),
        batch_size=int(opt.get("batch_size", 4)),
        prompt_version=opt.get("prompt_version", "gemini_ocr_v1"),
        schema_version=opt.get("schema_version", "ocr_detection_v1"),
        max_escalation_requests=int(opt.get("max_escalation_requests", 2)),
        escalation_enabled=bool(opt.get("escalation_enabled", False)),
        crop_retry_enabled=bool(opt.get("crop_retry_enabled", True)),
        crop_padding_ratio=float(opt.get("crop_padding_ratio", 0.05)),
        crop_upscale_factor=float(opt.get("crop_upscale_factor", 2.0)),
    ))
    registry.register("ocr", "gemini_vision_multistage", lambda opt: GeminiMultiStageOCRProvider(
        runtime=opt.get("runtime"),
        model=opt.get("model", "gemini-3.5-flash-lite"),
        stage_a_model=opt.get("stage_a_model", opt.get("region_detection_model")),
        stage_b_model=opt.get("stage_b_model", opt.get("region_classification_model")),
        stage_c_model=opt.get("stage_c_model", opt.get("crop_recognition_model")),
        stage_d_model=opt.get("stage_d_model", opt.get("verification_model")),
        stage_a_options=opt.get("stage_a_options"),
        stage_b_options=opt.get("stage_b_options"),
        stage_c_options=opt.get("stage_c_options"),
        stage_d_options=opt.get("stage_d_options"),
        api_key=opt.get("api_key"),
        tracking_iou_threshold=float(opt.get("tracking_iou_threshold", 0.3)),
        crop_padding_ratio=float(opt.get("crop_padding_ratio", 0.05)),
        window_size=int(opt.get("window_size", opt.get("region_window_frame_count", 3))),
        window_overlap=int(opt.get("window_overlap", 1)),
        prompt_version=opt.get("prompt_version", "gemini_ocr_multistage_v1"),
        schema_version=opt.get("schema_version", "ocr_multistage_v1"),
    ))
    registry.register("ocr", "gemini_video_direct", lambda opt: GeminiDirectVideoOCRProvider(
        runtime=opt.get("runtime"),
        model=opt.get("model", "gemini-3.5-flash-lite"),
        api_key=opt.get("api_key"),
        discovery_fps=float(opt.get("direct_video_discovery_fps", 6)),
        classification_fps=float(opt.get("direct_video_classification_fps", 6)),
        segmentation_fps=float(opt.get("direct_video_segmentation_fps", 12)),
        geometry_fps=float(opt.get("direct_video_geometry_fps", 12)),
        transcription_fps=float(opt.get("direct_video_transcription_fps", 12)),
        media_resolution=opt.get("direct_video_media_resolution", "high"),
        clip_padding_seconds=float(opt.get("direct_video_clip_padding_seconds", 0.40)),
        max_retry_per_stage=int(opt.get("direct_video_max_retry_per_stage", 1)),
        max_raw_candidates=int(opt.get("direct_video_max_raw_candidates", 200)),
        max_parseable_raw_candidates=int(opt.get("direct_video_max_parseable_raw_candidates", 200)),
        max_quota_wait_seconds=float(opt.get("direct_video_max_quota_wait_seconds", 120)),
        max_quota_retries_per_stage=int(opt.get("direct_video_max_quota_retries_per_stage", 1)),
        honor_retry_after=str(opt.get("direct_video_honor_retry_after", "True")).lower() == "true",
        fail_fast_on_daily_quota=str(opt.get("direct_video_fail_fast_on_daily_quota", "True")).lower() == "true",
        min_request_spacing_seconds=float(opt.get("direct_video_min_request_spacing_seconds", 5.0)),
        direct_video_active_rpm=int(opt.get("direct_video_active_rpm", 15)),
        quota_jitter_seconds=float(opt.get("direct_video_quota_jitter_seconds", 0.5)),
        unknown_429_backoff_seconds=float(opt.get("direct_video_unknown_429_backoff_seconds", 5.0)),
        fps_certification_ttl_seconds=float(opt.get("direct_video_fps_certification_ttl_seconds", 86400.0)),
        resume_enabled=str(opt.get("direct_video_resume_enabled", "True")).lower() == "true",
        checkpoint_schema_version=str(opt.get("direct_video_checkpoint_schema_version", "1.0.0")),
        batch_size=int(opt.get("direct_video_batch_size", 10)),
        resume_manifest_dir=opt.get("direct_video_resume_manifest_dir"),

        max_deduplicated_candidates=int(opt.get("direct_video_max_deduplicated_candidates", 10)),
        max_accepted_subtitle_tracks=int(opt.get("direct_video_max_accepted_subtitle_tracks", 10)),
        max_events_per_track=int(opt.get("direct_video_max_events_per_track", 20)),
        max_total_stage_requests=int(opt.get("direct_video_max_total_stage_requests", 100)),
        max_counted_input_tokens_per_job=int(opt.get("direct_video_max_counted_input_tokens_per_job", 500000)),
        stage_timeout_seconds=float(opt.get("direct_video_stage_timeout_seconds", 360)),
        no_progress_timeout_seconds=float(opt.get("direct_video_no_progress_timeout_seconds", 300)),
        job_timeout_seconds=float(opt.get("direct_video_job_timeout_seconds", 600)),
    ))

    # storage
    registry.register("storage", "mock", MockStorageProvider)

    registry.register("tts", "edge-tts", EdgeTTSProvider)
    registry.register("dubbing", "ffmpeg", FFmpegDubbingProvider)

    # frame_extraction
    registry.register("frame_extraction", "mock", MockFrameExtractionProvider)
    registry.register("frame_extraction", "ffmpeg", FFmpegFrameExtractionProvider)
