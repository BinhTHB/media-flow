"""One structured execution path for MediaFlow CLI and local web jobs."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from src.application.process_video import ProcessVideoUseCase
from src.application.request import ProcessingRequest
from src.application.result import ProcessingResult
from src.core.errors import ConfigurationError, MediaFlowError, ValidationError
from src.features.audio.pipeline import AudioPipeline, AudioRequest
from src.features.visual.pipeline import VisualPipeline, VisualRequest
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.infrastructure.detection.yolo import YoloDetectorAdapter
from src.infrastructure.video.ffmpeg import FFmpegVideoAdapter
from src.interfaces.composition.config import load_provider_config
from src.interfaces.composition.providers import (
    build_asr_corrector,
    build_asr_provider,
    build_spoken_rewriter,
    build_translation_service,
    build_tts_provider,
    has_elevenlabs_keys,
    has_openrouter_keys,
)
from src.interfaces.jobs.store import JobStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ProgressCallback = Callable[["JobEvent"], None]
JobProfile = Literal["quality", "balanced", "fast"]

#: Override the durable job workspace root (tests, alternate deployments).
JOBS_DIR_ENV = "MEDIA_FLOW_JOBS_DIR"


def default_jobs_dir() -> Path:
    """Resolve the durable job workspace root.

    ``MEDIA_FLOW_JOBS_DIR`` wins so tests and alternate deployments can keep
    job state out of the project tree.
    """
    override = os.environ.get(JOBS_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return PROJECT_ROOT / "outputs" / "jobs"

#: Pipeline presets select the DETECTION FLOW only. Everything a job does
#: (translate/blur/preserve subtitles, dub audio, handle extra text) is
#: expressed by the explicit feature fields on JobOptions.
YOLO_FLOW_PRESETS = {"yolo-4class", "yolo-text"}
#: Deprecated feature-combination presets (plus the "-generic" suffix). They
#: map onto a flow preset + feature fields and stay accepted so existing
#: CLI invocations and web clients keep working.
LEGACY_PRESET_FEATURES: dict[str, tuple[str, bool]] = {
    # base preset -> (subtitle_action, dub_audio)
    "translate-video": ("translate", True),
    "translate-subtitle": ("translate", False),
    "dub-audio": ("preserve", True),
    "blur-all": ("blur", False),
    "blur-and-dub": ("blur", True),
    "preserve-all": ("preserve", False),
}
GENERIC_TEXT_WEIGHTS_REL = Path("models") / "yolo" / "text" / "best.pt"
GENERIC_TEXT_WEIGHTS_URL = "https://huggingface.co/RoyRud1902/yolo11n-text"
#: The generic scene-text model is tuned for conf≈0.25; a slightly lower
#: default keeps small caption strokes while still suppressing UI noise.
GENERIC_TEXT_DEFAULT_CONF = 0.20
BURN_IN_WEIGHTS_DEFAULT = str(PROJECT_ROOT / "models" / "yolo" / "best.pt")


def resolve_preset(
    preset: str,
    subtitle_action: str,
    dub_audio: bool,
    extra_text_action: str,
    blur_extra: bool = False,
) -> tuple[str, str, bool, str]:
    """Resolve any accepted preset name onto the canonical contract.

    Returns ``(flow_preset, subtitle_action, dub_audio, extra_text_action)``.
    New-style flow presets pass the feature fields through; deprecated
    feature-combination presets (with optional "-generic" suffix) map onto
    the flow + feature fields they stood for. Raises ``ValueError`` for
    unknown preset names.
    """
    generic_suffix = preset.endswith("-generic")
    base = preset.removesuffix("-generic")
    if base in YOLO_FLOW_PRESETS:
        return base, subtitle_action, bool(dub_audio), extra_text_action
    features = LEGACY_PRESET_FEATURES.get(base)
    if features is None:
        raise ValueError(f"Unsupported preset: {preset}")
    legacy_subtitle, legacy_dub = features
    legacy_extra = "blur" if (base.startswith("blur-") or blur_extra) else "preserve"
    flow = "yolo-text" if generic_suffix else "yolo-4class"
    return flow, legacy_subtitle, legacy_dub, legacy_extra


def _normalize_preset(options: "JobOptions") -> None:
    """Resolve deprecated presets onto the flow preset + feature fields.

    Mutates ``options``. New-style flow presets are left untouched — their
    feature fields are authoritative.
    """
    (
        options.preset,
        options.subtitle_action,
        options.dub_audio,
        options.extra_text_action,
    ) = resolve_preset(
        options.preset,
        options.subtitle_action,
        options.dub_audio,
        options.extra_text_action,
        blur_extra=options.blur_extra,
    )


@dataclass(frozen=True)
class JobEvent:
    type: Literal["started", "progress", "completed", "failed"]
    stage: str
    progress: int
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobOptions:
    input_path: str
    output_path: str
    #: Detection flow: "yolo-4class" (self-trained burn-in model) or
    #: "yolo-text" (single-class yolo11n-text + classifier).
    preset: str = "yolo-4class"
    #: What happens to burned-in subtitles: "translate" | "blur" | "preserve".
    subtitle_action: str = "blur"
    #: Whether the audio pipeline dubs the video.
    dub_audio: bool = False
    #: Action for logo/watermark/other burn-in text: "blur" | "preserve".
    extra_text_action: str = "blur"
    device: str = "auto"
    job_id: str = ""
    overwrite: bool = False
    profile: JobProfile = "balanced"
    source_language: str = "auto"
    target_language: str = "vi"
    voice: str = "elevenlabs:env"
    asr_provider: str = "auto"
    asr_correction: str = "auto"
    translation_provider: str = "auto"
    duration_rewrite: str = "auto"
    max_tts_speedup: float = 1.30
    tts_provider: str = "auto"
    visual_cache_dir: str = str(PROJECT_ROOT / "outputs" / "cache" / "visual")
    no_visual_cache: bool = False
    yolo_weights: str = str(PROJECT_ROOT / "models" / "yolo" / "best.pt")
    yolo_imgsz: int = 1024
    yolo_conf: float = 0.05
    yolo_iou: float = 0.70
    yolo_subtitle_rescue_conf: float = 0.01
    yolo_subtitle_rescue_max_gap_seconds: float = 3.0
    yolo_subtitle_rescue_min_support_ratio: float = 0.25
    subtitle_layout: str = "temporal"
    hide_source_subtitle: bool = True
    timeout_seconds: float = 0.0
    segments: list[dict[str, Any]] | None = None
    speaker_voices: dict[str, str] | None = None
    blur_extra: bool = False
    speaker_mode: str = "single"
    preserve_ambience: bool = True
    mix_original_volume: float = 0.20


class JobError(RuntimeError):
    """Stable failure contract returned by all local entry points."""

    def __init__(self, code: str, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class JobRunner:
    """Validate, compose and execute one MediaFlow processing request."""

    def __init__(
        self,
        *,
        use_case_factory: Callable[[JobOptions, ProcessingRequest], tuple[Any, dict[str, Any]]] | None = None,
        job_store: JobStore | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._use_case_factory = use_case_factory or self._compose
        self._job_store = job_store or JobStore(default_jobs_dir())
        self._cancel_requested = cancel_requested or (lambda: False)

    def run(self, options: JobOptions, on_event: ProgressCallback | None = None) -> ProcessingResult:
        callback = on_event or (lambda event: None)
        started = time.monotonic()
        request: ProcessingRequest | None = None
        claimed_job_id = ""

        def emit(event: JobEvent) -> None:
            callback(event)
            if request and request.job_id:
                self._job_store.append_event(request.job_id, event.to_dict())
                state = self._job_store.get(request.job_id) or {"id": request.job_id}
                state.update({"status": "RUNNING", "stage": event.stage, "progress": event.progress, "updated_at": time.time()})
                self._job_store.save_state(request.job_id, state)

        def checkpoint() -> None:
            persisted_cancel = bool(
                request and request.job_id and self._job_store.cancel_requested(request.job_id)
            )
            if self._cancel_requested() or persisted_cancel:
                raise JobError("cancelled", "Job was cancelled.")
            if options.timeout_seconds > 0 and time.monotonic() - started > options.timeout_seconds:
                raise JobError("timeout", "Job exceeded its configured timeout.")

        try:
            callback(JobEvent("started", "preflight", 5, "Validating local runtime"))
            normalized, request = self.prepare(options)
            candidate_job_id = str(request.job_id)
            if not self._job_store.claim(candidate_job_id):
                request = None  # Do not overwrite state owned by the active runner.
                raise JobError("concurrency_limit", f"Job '{candidate_job_id}' is already running.")
            claimed_job_id = candidate_job_id
            self._job_store.save_state(request.job_id, {
                "id": request.job_id, "status": "RUNNING", "progress": 5,
                "created_at": time.time(), "updated_at": time.time(), "request": request.to_dict(),
                "provenance": {
                    "profile": normalized.profile,
                    "device": normalized.device,
                    "providers": {
                        "asr": normalized.asr_provider,
                        "translation": normalized.translation_provider,
                        "tts": normalized.tts_provider,
                    },
                },
            })
            self._job_store.append_event(request.job_id, JobEvent("started", "preflight", 5, "Validating local runtime").to_dict())
            checkpoint()
            emit(JobEvent("progress", "composition", 25, "Composing providers and pipelines"))
            use_case, caches = self._use_case_factory(normalized, request)
            checkpoint()
            emit(JobEvent("progress", "processing", 40, "Processing media"))
            result = use_case.execute(request)
            checkpoint()
            if result.output_video_path and Path(result.output_video_path).is_file():
                try:
                    from src.infrastructure.video.ffmpeg import ensure_web_compatible_video
                    ensure_web_compatible_video(result.output_video_path)
                except Exception:
                    pass
            emit(JobEvent("progress", "finalizing", 92, "Finalizing output", {"detections": dict(result.detections_summary)}))
            details = {"result": result.to_dict(), "caches": self._cache_stats(caches)}
            self._job_store.save_result(request.job_id, result.to_dict())
            emit(JobEvent("completed", "completed", 100, "Media processing completed", details))
            state = self._job_store.get(request.job_id) or {"id": request.job_id}
            state.update({
                "status": result.status,
                "progress": 100,
                "updated_at": time.time(),
                "metrics": {
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "detections": dict(result.detections_summary),
                    "tracks": dict(result.tracks_summary),
                    "cache": self._cache_stats(caches),
                },
            })
            self._job_store.save_state(request.job_id, state)
            return result
        except Exception as exc:
            error = self.map_error(exc)
            event = JobEvent("failed", "failed", 100, str(error), {"error": error.to_dict()})
            callback(event)
            if request and request.job_id:
                self._job_store.append_event(request.job_id, event.to_dict())
                state = self._job_store.get(request.job_id) or {"id": request.job_id}
                status = "BLOCKED" if error.code in {"configuration_error", "cancelled", "timeout", "concurrency_limit"} else "FAILED"
                state.update({
                    "status": status,
                    "progress": 100,
                    "error": error.to_dict(),
                    "updated_at": time.time(),
                    "metrics": {"duration_seconds": round(time.monotonic() - started, 6)},
                })
                self._job_store.save_state(request.job_id, state)
            raise error from exc
        finally:
            if claimed_job_id:
                self._job_store.release(claimed_job_id)

    def prepare(self, options: JobOptions) -> tuple[JobOptions, ProcessingRequest]:
        if options.preset not in YOLO_FLOW_PRESETS:
            _normalize_preset(options)
        if options.subtitle_action not in {"translate", "blur", "preserve"}:
            raise ValidationError(f"Unsupported subtitle_action: {options.subtitle_action}")
        if options.extra_text_action not in {"blur", "preserve"}:
            raise ValidationError(f"Unsupported extra_text_action: {options.extra_text_action}")
        if options.profile not in {"quality", "balanced", "fast"}:
            raise ValidationError(f"Unsupported profile: {options.profile}")
        if options.device not in {"auto", "cuda", "cpu"}:
            raise ValidationError(f"Unsupported device: {options.device}")

        input_path = Path(options.input_path).resolve()
        if not input_path.is_file():
            raise ValidationError(f"Input video file not found: {input_path}")
        output_path = Path(options.output_path).resolve()
        if output_path.exists() and not options.overwrite:
            raise ValidationError(f"Output video already exists: {output_path}")

        config = load_provider_config()
        options.asr_provider = config.resolve("speech", options.asr_provider)
        options.translation_provider = config.resolve("translation", options.translation_provider)
        options.tts_provider = config.resolve("tts", options.tts_provider)
        if (
            options.preset == "yolo-text"
            and Path(options.yolo_weights).resolve() == Path(BURN_IN_WEIGHTS_DEFAULT).resolve()
        ):
            options.yolo_weights = str(PROJECT_ROOT / GENERIC_TEXT_WEIGHTS_REL)
        effective_device = self._preflight(options)
        options.device = effective_device
        options.input_path = str(input_path)
        options.output_path = str(output_path)
        needs_dubbing = bool(options.dub_audio)
        parsed_segments = None
        if options.segments:
            from src.core.models.speech_segment import SpeechSegment
            parsed_segments = [
                SpeechSegment.from_dict(s) if isinstance(s, dict) else s
                for s in options.segments
            ]

        audio = AudioRequest(
            enabled=needs_dubbing,
            source_language=options.source_language,
            target_language=options.target_language,
            dubbing_enabled=needs_dubbing,
            voice=options.voice,
            segments=parsed_segments,
            speaker_voices=dict(options.speaker_voices or {}),
            preserve_ambience=bool(options.preserve_ambience),
            mix_original_volume=float(options.mix_original_volume),
        )
        visual = self._visual_request(options)
        normalized_layout = "temporal" if options.subtitle_layout in {"in-place", "temporal"} else "bottom"
        visual.subtitle_layout = normalized_layout
        visual.hide_source_subtitle = options.hide_source_subtitle
        if options.segments and options.subtitle_action == "translate":
            visual.segments = options.segments
        request = ProcessingRequest(
            job_id=options.job_id or f"job_v2_{int(time.time())}",
            input_video=options.input_path,
            output_video=options.output_path,
            audio=audio,
            visual=visual,
            device=effective_device,
            overwrite=options.overwrite,
        )
        return options, request

    def _preflight(self, options: JobOptions) -> str:
        needs_visual = not (
            options.subtitle_action == "preserve" and options.extra_text_action == "preserve"
        )
        effective_device = "cpu"
        if needs_visual:
            torch = self._require_dependency("torch", "torch")
            self._require_dependency("cv2", "opencv-python")
            self._require_dependency("ultralytics", "ultralytics")
            if not Path(options.yolo_weights).is_file():
                message = f"YOLO weights not found: {options.yolo_weights}"
                if options.preset == "yolo-text":
                    message += (
                        f". Download best.pt from {GENERIC_TEXT_WEIGHTS_URL} and place it at "
                        "models/yolo/text/best.pt, or pass --yolo-weights / set "
                        "MEDIA_FLOW_YOLO_WEIGHTS."
                    )
                raise ConfigurationError(message)
            if options.device in {"auto", "cuda"}:
                cuda_ok = bool(torch.cuda.is_available())
                if options.device == "cuda" and not cuda_ok:
                    raise ConfigurationError("CUDA was requested but is not available.")
                effective_device = "cuda" if cuda_ok else "cpu"
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ConfigurationError("FFmpeg and FFprobe must be available in PATH.")

        has_pretranslated_segments = bool(
            options.segments and all(
                (s.get("target_text") or s.get("translated_text"))
                if isinstance(s, dict)
                else getattr(s, "translated_text", None)
                for s in options.segments
            )
        )
        needs_visual_translation = options.subtitle_action == "translate" and not options.segments
        needs_audio_translation = bool(options.dub_audio) and not has_pretranslated_segments
        needs_translation = needs_visual_translation or needs_audio_translation
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        has_openrouter = has_openrouter_keys()
        if needs_translation and options.translation_provider == "auto" and not (has_openrouter or has_gemini):
            raise ConfigurationError("Translation requires OPENROUTER_API_KEY(S) or GEMINI_API_KEY.")
        if options.translation_provider == "openrouter" and not has_openrouter:
            raise ConfigurationError("OPENROUTER_API_KEY(S) is required for translation provider openrouter.")
        if options.translation_provider == "gemini" and not has_gemini:
            raise ConfigurationError("GEMINI_API_KEY is required for translation provider gemini.")
        if needs_visual_translation and not has_gemini:
            raise ConfigurationError("GEMINI_API_KEY is required for visual subtitle OCR.")

        if options.dub_audio:
            if not options.segments:
                if options.asr_provider == "auto" and not (os.getenv("GROQ_API_KEY") or has_gemini):
                    raise ConfigurationError("ASR requires GROQ_API_KEY or GEMINI_API_KEY.")
                if options.asr_provider == "groq" and not os.getenv("GROQ_API_KEY"):
                    raise ConfigurationError("GROQ_API_KEY is required for ASR provider groq.")
                if options.asr_provider == "gemini" and not has_gemini:
                    raise ConfigurationError("GEMINI_API_KEY is required for ASR provider gemini.")
            if options.tts_provider in {"auto", "edge"}:
                self._require_dependency("edge_tts", "edge-tts")
            if options.tts_provider == "elevenlabs" and not has_elevenlabs_keys():
                raise ConfigurationError("ELEVENLABS_API_KEY(S) is required for TTS provider elevenlabs.")
            if options.tts_provider == "fish" and not has_openrouter:
                raise ConfigurationError("OpenRouter key pool is required for TTS provider fish.")
            if options.tts_provider == "gemini-live" and not has_gemini:
                raise ConfigurationError("GEMINI_API_KEY is required for TTS provider gemini-live.")
        return effective_device

    @staticmethod
    def _require_dependency(module_name: str, package_name: str):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ConfigurationError(
                f"Missing dependency '{package_name}' in interpreter {sys.executable}."
            ) from exc

    @staticmethod
    def _visual_request(options: JobOptions) -> VisualRequest:
        common = {
            "target_language": options.target_language,
            "subtitle_layout": options.subtitle_layout,
            "hide_source_subtitle": options.hide_source_subtitle,
        }
        # Nothing touches pixels: skip detection entirely (mirrors the old
        # dub-audio preset and makes preserve-only jobs cheap).
        if options.subtitle_action == "preserve" and options.extra_text_action == "preserve":
            return VisualRequest(enabled=False, **common)
        if options.segments and options.subtitle_action == "translate":
            common["segments"] = options.segments
        subtitle_mode = {
            "translate": "translate_replace",
            "blur": "blur",
            "preserve": "preserve",
        }[options.subtitle_action]
        return VisualRequest(enabled=True, subtitle_enabled=True, subtitle_mode=subtitle_mode,
                             logo_enabled=True, logo_action=options.extra_text_action,
                             watermark_enabled=True, watermark_action=options.extra_text_action,
                             other_text_enabled=True, other_text_action=options.extra_text_action,
                             **common)

    @staticmethod
    def _compose(options: JobOptions, request: ProcessingRequest) -> tuple[ProcessVideoUseCase, dict[str, Any]]:
        video_util = FFmpegVideoAdapter()
        # Resolve here too so direct callers (tests, tooling) with a raw
        # legacy preset behave exactly like a prepared JobOptions.
        resolved_preset, _, _, _ = resolve_preset(
            options.preset,
            options.subtitle_action,
            options.dub_audio,
            options.extra_text_action,
            blur_extra=options.blur_extra,
        )
        use_text_flow = resolved_preset == "yolo-text"
        needs_visual_translation = (
            request.visual.enabled and request.visual.subtitle_mode == "translate_replace"
        )
        needs_audio_translation = bool(options.dub_audio)
        visual_translation = build_translation_service("subtitle", options.translation_provider) if needs_visual_translation else None
        audio_translation = build_translation_service("spoken", options.translation_provider) if needs_audio_translation else None
        detector_config = {
            "weights_path": options.yolo_weights,
            # The generic scene-text model needs its own operating point; the
            # 4-class burn-in model keeps its low default for the rescue gate.
            "conf": (
                GENERIC_TEXT_DEFAULT_CONF
                if use_text_flow and float(options.yolo_conf) == 0.05
                else options.yolo_conf
            ),
            "iou": options.yolo_iou,
            # The generic text model is trained and recommended at 640px;
            # matching it speeds detection up sharply on dense frames.
            "imgsz": (
                640
                if use_text_flow and int(options.yolo_imgsz) == 1024
                else options.yolo_imgsz
            ),
            "device": options.device,
            # Class-gated weak-subtitle rescue never applies to the generic
            # single-class model; disable it explicitly.
            "subtitle_rescue_conf": None if use_text_flow else options.yolo_subtitle_rescue_conf,
            "subtitle_rescue_max_gap_seconds": options.yolo_subtitle_rescue_max_gap_seconds,
            "subtitle_rescue_min_support_ratio": options.yolo_subtitle_rescue_min_support_ratio,
        }
        detector = YoloDetectorAdapter(**detector_config) if request.visual.enabled else None
        if use_text_flow and detector is not None:
            from src.features.visual.text_classifier import TextClassifyingDetector

            detector = TextClassifyingDetector(detector)
        if use_text_flow and request.visual.enabled:
            # Classifier edits change detector output semantics; include the
            # classifier source digest so the detection cache invalidates.
            from src.infrastructure.stage_cache import file_digest

            detector_config["classifier_sha256"] = file_digest(
                PROJECT_ROOT / "src" / "features" / "visual" / "text_classifier.py"
            )
        ocr_provider = None
        if request.visual.enabled and request.visual.subtitle_mode == "translate_replace":
            from src.infrastructure.ocr.gemini_ocr import GeminiOCRAdapter

            ocr_provider = GeminiOCRAdapter()
        caches: dict[str, Any] = {}
        if request.visual.enabled and not options.no_visual_cache:
            from src.infrastructure.visual_cache_wiring import wire_visual_caches

            detector, ocr_provider, visual_translation, caches = wire_visual_caches(
                PROJECT_ROOT, options.visual_cache_dir, detector, ocr_provider,
                visual_translation, detector_config, options.translation_provider,
            )
        from src.infrastructure.video.scene_boundaries import FrameDifferenceSceneDetector
        from src.infrastructure.stage_cache import CachedSceneCuts

        scene_detector = FrameDifferenceSceneDetector()
        if request.visual.enabled and "scene" in caches:
            scene_detector = CachedSceneCuts(scene_detector, caches["scene"])

        visual_pipeline = VisualPipeline(
            detector=detector,
            subtitle_processor=SubtitleProcessor(ocr_provider=ocr_provider, translation_service=visual_translation),
            scene_detector=scene_detector,
            video_util=video_util,
        )
        audio_pipeline = AudioPipeline(audio_extractor=video_util)
        if request.audio.enabled:
            from src.features.audio.composer import AudioComposer

            audio_pipeline = AudioPipeline(
                asr_provider=build_asr_provider(options.asr_provider),
                asr_corrector=build_asr_corrector(options.asr_correction, options.asr_provider),
                translation_service=audio_translation,
                spoken_rewriter=build_spoken_rewriter(options.duration_rewrite),
                tts_provider=build_tts_provider(options.tts_provider),
                composer=AudioComposer(max_speedup=options.max_tts_speedup),
                audio_extractor=video_util,
            )
        return ProcessVideoUseCase(audio_pipeline=audio_pipeline, visual_pipeline=visual_pipeline, video_util=video_util), caches

    @staticmethod
    def _cache_stats(caches: dict[str, Any]) -> dict[str, dict[str, int]]:
        return {stage: {"hits": cache.hits, "misses": cache.misses} for stage, cache in caches.items()}

    @staticmethod
    def map_error(exc: Exception) -> JobError:
        if isinstance(exc, JobError):
            return exc
        if isinstance(exc, (ValidationError, FileNotFoundError, FileExistsError, ValueError)):
            return JobError("validation_error", str(exc), cause=exc)
        if isinstance(exc, (ConfigurationError, ModuleNotFoundError, ImportError)):
            return JobError("configuration_error", str(exc), cause=exc)
        if isinstance(exc, MediaFlowError):
            return JobError("pipeline_error", str(exc), cause=exc)
        return JobError("internal_error", str(exc), cause=exc)


def format_event(event: JobEvent) -> str:
    """Human-readable CLI presentation; consumers receive the structured event first."""
    return f"[{event.progress:3d}%] {event.stage}: {event.message}"


def event_as_json(event: JobEvent) -> str:
    return json.dumps(event.to_dict(), ensure_ascii=False)
