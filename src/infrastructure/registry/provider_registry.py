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


def register_default_providers(registry: ProviderRegistry) -> None:
    from src.infrastructure.providers.ocr.mock_ocr_provider import MockOCRProvider
    from src.infrastructure.providers.ocr.paddle_ocr_provider import PaddleOCRProvider
    from src.infrastructure.providers.speech.faster_whisper_speech_provider import FasterWhisperSpeechProvider
    from src.infrastructure.providers.speech.mock_speech_provider import MockSpeechProvider
    from src.infrastructure.providers.storage.mock_storage_provider import MockStorageProvider
    from src.infrastructure.providers.subtitle.mock_subtitle_provider import MockSubtitleProvider
    from src.infrastructure.providers.subtitle.ass_subtitle_provider import ASSSubtitleProvider
    from src.infrastructure.providers.subtitle.mock_subtitle_region_detector_provider import MockSubtitleRegionDetectorProvider
    from src.infrastructure.providers.subtitle.paddle_subtitle_region_detector_provider import PaddleOCRSubtitleRegionDetectorProvider
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

    # translation
    registry.register("translation", "mock", MockTranslationProvider)
    registry.register("translation", "openai", lambda opt: OpenAITranslationProvider(
        model=opt.get("model", "gpt-4o-mini"),
        max_retries=int(opt.get("max_retries", 3)),
    ))
    registry.register("translation", "gemini", lambda opt: GeminiTranslationProvider(
        model=opt.get("model", "gemini-3.1-flash-lite"),
        max_retries=int(opt.get("max_retries", 3)),
        api_key=opt.get("api_key"),
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
        "paddle",
        lambda opt: PaddleOCRSubtitleRegionDetectorProvider(
            language=opt.get("language", "ch"),
            use_gpu=opt.get("use_gpu", True),
            min_confidence=float(opt.get("min_confidence", 0.70)),
        ),
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
    registry.register("ocr", "paddle", lambda opt: PaddleOCRProvider(
        language=opt.get("language", "ch"),
        use_gpu=opt.get("use_gpu", True),
    ))

    # storage
    registry.register("storage", "mock", MockStorageProvider)

    registry.register("tts", "edge-tts", EdgeTTSProvider)
    registry.register("dubbing", "ffmpeg", FFmpegDubbingProvider)

    # frame_extraction
    registry.register("frame_extraction", "mock", MockFrameExtractionProvider)
    registry.register("frame_extraction", "ffmpeg", FFmpegFrameExtractionProvider)
