#!/usr/bin/env python3
"""
MediaFlow End-to-End Runner (Feature-First Modular Clean Architecture V2)

Executes video processing via ProcessVideoUseCase, AudioPipeline, and VisualPipeline.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _project_python() -> Path:
    """Return the repository virtual-environment interpreter path."""
    if os.name == "nt":
        return BASE_DIR / ".venv" / "Scripts" / "python.exe"
    return BASE_DIR / ".venv" / "bin" / "python"


def _reexec_with_project_python() -> None:
    """Keep bare `python scripts/run_e2e.py` on the project runtime."""
    expected = _project_python()
    try:
        current = Path(sys.executable).resolve()
        expected_resolved = expected.resolve()
    except OSError:
        return
    if not expected.is_file() or current == expected_resolved:
        return
    print(
        f"[Runtime] Switching interpreter: {sys.executable} -> {expected_resolved}",
        flush=True,
    )
    os.execv(str(expected_resolved), [str(expected_resolved), str(Path(__file__).resolve()), *sys.argv[1:]])


# Load .env file automatically if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                # Pool-style credentials are additive: merge project .env with any
                # process-level pool instead of letting one stale environment value
                # shadow the entire configured pool. Scalar settings still respect
                # explicit process environment overrides.
                if k in {"OPENROUTER_API_KEYS", "FISH_API_KEYS", "ELEVENLABS_API_KEYS"} and v:
                    existing = os.environ.get(k, "").strip()
                    os.environ[k] = f"{existing},{v}" if existing else v
                elif k not in os.environ:
                    os.environ[k] = v

from src.application.process_video import ProcessVideoUseCase
from src.application.request import ProcessingRequest
from src.features.audio.pipeline import AudioPipeline, AudioRequest
from src.features.translation.service import TranslationService
from src.features.visual.pipeline import VisualPipeline, VisualRequest
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.infrastructure.detection.yolo import YoloDetectorAdapter
from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter
from src.infrastructure.video.ffmpeg import FFmpegVideoAdapter
from src.interfaces.composition.providers import (
    build_asr_corrector as _shared_build_asr_corrector,
    build_asr_provider as _shared_build_asr_provider,
    build_spoken_rewriter as _shared_build_spoken_rewriter,
    build_translation_service as _shared_build_translation_service,
    build_tts_provider as _shared_build_tts_provider,
    has_elevenlabs_keys as _shared_has_elevenlabs_keys,
    has_openrouter_keys as _shared_has_openrouter_keys,
)
from src.interfaces.jobs.runner import JobOptions, JobRunner, format_event


def require_runtime_dependency(module_name: str, package_name: str):
    """Import a runtime dependency with an actionable interpreter diagnostic."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{package_name}' in interpreter "
            f"{sys.executable}. Use the project interpreter: "
            f"{BASE_DIR}\\.venv\\Scripts\\python.exe -m pip install {package_name}"
        ) from exc
    return module


def _has_openrouter_keys() -> bool:
    from src.infrastructure.translation.openrouter_translation import configured_openrouter_keys
    return bool(configured_openrouter_keys())


def _has_elevenlabs_keys() -> bool:
    from src.infrastructure.elevenlabs_dubbing import configured_elevenlabs_keys
    return bool(configured_elevenlabs_keys())


def _build_translation_service(purpose: str, provider_choice: str) -> TranslationService:
    """Build an independent translation service for subtitle or spoken audio."""
    from src.infrastructure.translation.openrouter_translation import OpenRouterTranslationAdapter
    from src.infrastructure.translation.router import FallbackTranslationAdapter

    providers = []
    # Subtitle quality benefits more from whole-batch domain reasoning than from
    # minimizing latency. Prefer Gemini for on-screen captions; keep OpenRouter
    # first for spoken dubbing where the existing free-model router/rewrite path is
    # tuned for throughput and isolated semantic repair.
    if purpose == "subtitle" and provider_choice == "auto":
        if os.getenv("GEMINI_API_KEY"):
            providers.append(("gemini", GeminiTranslationAdapter(model="gemini-3.5-flash", purpose=purpose)))
        if _has_openrouter_keys():
            providers.append(("openrouter", OpenRouterTranslationAdapter(purpose=purpose)))
    else:
        if provider_choice in {"auto", "openrouter"} and _has_openrouter_keys():
            providers.append(("openrouter", OpenRouterTranslationAdapter(purpose=purpose)))
        if provider_choice in {"auto", "gemini"} and os.getenv("GEMINI_API_KEY"):
            providers.append(("gemini", GeminiTranslationAdapter(model="gemini-3.5-flash", purpose=purpose)))
    if provider_choice == "openrouter" and not _has_openrouter_keys():
        raise RuntimeError("OPENROUTER_API_KEY(S) is required for --translation-provider openrouter.")
    if provider_choice == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for --translation-provider gemini.")
    if not providers:
        raise RuntimeError("No translation provider is configured; set OPENROUTER_API_KEY(S) or GEMINI_API_KEY.")
    return TranslationService(FallbackTranslationAdapter(providers))


def _build_asr_provider(provider_choice: str):
    from src.infrastructure.speech.gemini_audio import GeminiAudioASRAdapter
    from src.infrastructure.speech.groq_whisper import GroqWhisperASRAdapter
    from src.infrastructure.speech.router import FallbackASRAdapter

    if provider_choice == "groq":
        return GroqWhisperASRAdapter(model="whisper-large-v3")
    if provider_choice == "gemini":
        return GeminiAudioASRAdapter()

    providers = []
    if os.getenv("GROQ_API_KEY"):
        providers.append(("groq-whisper-large-v3", GroqWhisperASRAdapter(model="whisper-large-v3")))
    if os.getenv("GEMINI_API_KEY"):
        providers.append(("gemini-audio", GeminiAudioASRAdapter()))
    if not providers:
        raise RuntimeError("No ASR provider is configured; set GROQ_API_KEY or GEMINI_API_KEY.")
    if len(providers) == 1:
        return providers[0][1]
    return FallbackASRAdapter(providers)


def _build_asr_corrector(provider_choice: str, asr_provider_choice: str):
    if provider_choice == "off":
        return None
    if provider_choice in {"auto", "openrouter"} and _has_openrouter_keys():
        from src.infrastructure.speech.openrouter_corrector import OpenRouterASRCorrector
        secondary_provider = None
        primary_is_groq = asr_provider_choice == "groq" or (
            asr_provider_choice == "auto" and bool(os.getenv("GROQ_API_KEY"))
        )
        if primary_is_groq and os.getenv("GEMINI_API_KEY"):
            from src.infrastructure.speech.gemini_audio import GeminiAudioASRAdapter
            secondary_provider = GeminiAudioASRAdapter()
        return OpenRouterASRCorrector(secondary_provider=secondary_provider)
    if provider_choice == "openrouter":
        raise RuntimeError("OpenRouter key pool is required for --asr-correction openrouter.")
    return None


def _build_spoken_rewriter(provider_choice: str):
    if provider_choice == "off":
        return None
    if provider_choice in {"auto", "openrouter"} and _has_openrouter_keys():
        from src.infrastructure.translation.duration_rewriter import OpenRouterDurationRewriter
        return OpenRouterDurationRewriter()
    if provider_choice == "openrouter":
        raise RuntimeError("OpenRouter key pool is required for --duration-rewrite openrouter.")
    return None


def _build_tts_provider(provider_choice: str):
    from src.infrastructure.tts.edge_tts import EdgeTTSAdapter
    from src.infrastructure.tts.openrouter_tts import OpenRouterTTSAdapter
    from src.infrastructure.tts.gemini_flash_live import GeminiFlashLiveTTSAdapter
    from src.infrastructure.tts.elevenlabs_tts import ElevenLabsTTSAdapter
    from src.infrastructure.tts.router import FallbackTTSAdapter

    if provider_choice == "elevenlabs":
        if not _has_elevenlabs_keys():
            raise RuntimeError("ELEVENLABS_API_KEY(S) is required for --tts-provider elevenlabs.")
        return ElevenLabsTTSAdapter()
    if provider_choice == "fish":
        if not _has_openrouter_keys():
            raise RuntimeError("OpenRouter key pool is required for --tts-provider fish.")
        return OpenRouterTTSAdapter()
    if provider_choice == "gemini-live":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required for --tts-provider gemini-live.")
        return GeminiFlashLiveTTSAdapter()
    if provider_choice == "edge":
        return EdgeTTSAdapter()

    # Development route with no paid dependency: ElevenLabs Free API first when
    # configured, then Fish free while quota remains, then Edge as deterministic
    # emergency fallback. Gemini Live stays explicit-only because it is not
    # verbatim-safe for dubbing; Gemini 3.1 Flash TTS Preview stays out of auto due
    # to its tiny free daily quota.
    providers = []
    if _has_elevenlabs_keys():
        providers.append(("elevenlabs-v3", ElevenLabsTTSAdapter()))
    if _has_openrouter_keys():
        providers.append(("openrouter-fish-s2.1-pro-free", OpenRouterTTSAdapter()))
    providers.append(("edge-tts", EdgeTTSAdapter()))
    return FallbackTTSAdapter(providers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run feature-first modular MediaFlow END-TO-END pipeline on a video."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video file (e.g. test_videos/real_mini_benchmark/sample5.mp4)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to final output video file (e.g. outputs/e2e_sample5.mp4)"
    )
    parser.add_argument(
        "--preset",
        choices=["blur-all", "translate-subtitle", "preserve-all", "dub-audio", "translate-video"],
        default="blur-all",
        help="Policy preset (default: blur-all)",
    )
    parser.add_argument(
        "--policy", type=str, default="", help="Path to policy YAML file"
    )
    parser.add_argument(
        "--profile", choices=["quality", "balanced", "fast"], default="balanced",
        help="Runtime profile preserved across CLI and web execution (default: balanced)",
    )
    parser.add_argument("--source-language", default="auto", help="Spoken source language for ASR")
    parser.add_argument("--target-language", default="vi", help="Translation target language")
    parser.add_argument(
        "--voice",
        default="elevenlabs:env",
        help="TTS voice id. elevenlabs:env uses ELEVENLABS_VOICE_ID (or Adam default).",
    )
    parser.add_argument(
        "--asr-provider", choices=["auto", "groq", "gemini"], default="auto",
        help="Remote ASR provider for dubbing presets",
    )
    parser.add_argument(
        "--asr-correction", choices=["auto", "off", "openrouter"], default="auto",
        help="Contextual ASR correction. auto enables OpenRouter correction when a key pool is configured.",
    )
    parser.add_argument(
        "--translation-provider", choices=["auto", "openrouter", "gemini"], default="auto",
        help="Translation provider. auto prefers OpenRouter pool and keeps Gemini as fallback.",
    )
    parser.add_argument(
        "--duration-rewrite", choices=["auto", "off", "openrouter"], default="auto",
        help="Shorten spoken copy only when natural TTS would exceed the speed-up cap.",
    )
    parser.add_argument(
        "--max-tts-speedup", type=float, default=1.30,
        help="Maximum allowed speech speed-up during composition (default: 1.30x).",
    )
    parser.add_argument(
        "--tts-provider", choices=["auto", "elevenlabs", "fish", "gemini-live", "edge"], default="auto",
        help="TTS provider. auto prefers ElevenLabs, then Fish, then Edge; Gemini Live is explicit experimental only.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device (auto|cuda|cpu)"
    )
    parser.add_argument(
        "--job-id", type=str, default="", help="Custom job ID"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file if present"
    )
    parser.add_argument('--visual-cache-dir', default=str(BASE_DIR / 'outputs/cache/visual'), help='Content-addressed detection/OCR/translation cache')
    parser.add_argument('--no-visual-cache', action='store_true', help='Recompute visual provider outputs')
    parser.add_argument('--yolo-weights', default=str(BASE_DIR / 'models/yolo/best.pt'))
    parser.add_argument('--yolo-imgsz', type=int, default=1024)
    parser.add_argument('--yolo-conf', type=float, default=0.05)
    parser.add_argument('--yolo-iou', type=float, default=0.70)
    parser.add_argument('--yolo-subtitle-rescue-conf', type=float, default=0.01)
    parser.add_argument('--yolo-subtitle-rescue-max-gap-seconds', type=float, default=3.0)
    parser.add_argument('--yolo-subtitle-rescue-min-support-ratio', type=float, default=0.25)
    parser.add_argument('--subtitle-layout', choices=['temporal', 'legacy_centered'], default='temporal')
    return parser.parse_args()


def preflight_check(args):
    print("=" * 60)
    print("[1/5] Preflight Check")
    print("=" * 60)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    print(f"  [OK] Input video: {input_path}")

    needs_visual = args.preset != "dub-audio"
    effective_device = "cpu"
    if needs_visual:
        torch = require_runtime_dependency("torch", "torch")
        require_runtime_dependency("cv2", "opencv-python")
        ultralytics = require_runtime_dependency("ultralytics", "ultralytics")
        version = getattr(ultralytics, "__version__", "installed")
        print(f"  [OK] YOLO runtime: ultralytics {version} ({sys.executable})")
        weights_path = Path(args.yolo_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at expected location: {weights_path}")
        print(f"  [OK] YOLO weights: {weights_path}")
        if args.device in ("cuda", "auto"):
            cuda_ok = torch.cuda.is_available()
            effective_device = "cuda" if cuda_ok else "cpu"
            if args.device == "cuda" and not cuda_ok:
                raise RuntimeError("CUDA requested via --device cuda but CUDA is not available.")
            print(f"  [OK] Compute Device: {effective_device} (Requested: {args.device}, CUDA available: {cuda_ok})")
        else:
            print("  [OK] Compute Device: cpu")
    else:
        print("  [OK] Visual pipeline disabled for dub-audio; YOLO preflight skipped")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg executable not found in PATH.")
    if not shutil.which("ffprobe"):
        raise RuntimeError("FFprobe executable not found in PATH.")
    print("  [OK] FFmpeg / FFprobe: Available")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output path already exists (use --overwrite to replace): {output_path}")
    print(f"  [OK] Output directory: {output_path.parent} (writable)")

    needs_translation = args.preset in {"translate-subtitle", "dub-audio", "translate-video"} or (args.policy and "translate" in args.policy)
    if needs_translation:
        has_openrouter = _has_openrouter_keys()
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        if args.translation_provider == "openrouter" and not has_openrouter:
            raise RuntimeError("OPENROUTER_API_KEY(S) is required for --translation-provider openrouter.")
        if args.translation_provider == "gemini" and not has_gemini:
            raise RuntimeError("GEMINI_API_KEY is required for --translation-provider gemini.")
        if args.translation_provider == "auto" and not (has_openrouter or has_gemini):
            raise RuntimeError("Translation requires OPENROUTER_API_KEY(S) or GEMINI_API_KEY.")
        selected_translation = (
            "openrouter+gemini-fallback"
            if args.translation_provider == "auto" and has_openrouter and has_gemini
            else ("openrouter" if (args.translation_provider == "openrouter" or (args.translation_provider == "auto" and has_openrouter)) else "gemini")
        )
        print(f"  [OK] Translation: {selected_translation}")

    # Visual subtitle OCR is still a separate Gemini Vision service.
    if args.preset in {"translate-subtitle", "translate-video"} and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is currently required for visual subtitle OCR.")

    if args.preset in {"dub-audio", "translate-video"}:
        selected_asr = args.asr_provider
        if selected_asr == "auto":
            if os.getenv("GROQ_API_KEY"):
                selected_asr = "groq-whisper-large-v3+gemini-fallback" if os.getenv("GEMINI_API_KEY") else "groq-whisper-large-v3"
            else:
                selected_asr = "gemini"
        if args.asr_provider == "groq" and not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required when --asr-provider groq is selected.")
        if args.asr_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required when --asr-provider gemini is selected.")
        if args.asr_provider == "auto" and not (os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RuntimeError("ASR requires GROQ_API_KEY or GEMINI_API_KEY.")

        if args.tts_provider in {"auto", "edge"}:
            require_runtime_dependency("edge_tts", "edge-tts")
        if args.tts_provider == "gemini-live" and os.getenv("GEMINI_API_KEY"):
            require_runtime_dependency("google.genai", "google-genai")
        if args.tts_provider == "elevenlabs" and not _has_elevenlabs_keys():
            raise RuntimeError("ELEVENLABS_API_KEY(S) is required when --tts-provider elevenlabs is selected.")
        if args.tts_provider == "fish" and not _has_openrouter_keys():
            raise RuntimeError("OpenRouter key pool is required when --tts-provider fish is selected.")
        if args.tts_provider == "gemini-live" and not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required when --tts-provider gemini-live is selected.")
        if args.tts_provider == "auto":
            route = []
            if _has_elevenlabs_keys():
                route.append("elevenlabs-v3")
            if _has_openrouter_keys():
                route.append("fish-s2.1-pro-free")
            route.append("edge-tts")
            selected_tts = "+".join(route)
        elif args.tts_provider == "elevenlabs":
            selected_tts = "elevenlabs-v3"
        elif args.tts_provider == "fish":
            selected_tts = "openrouter-fish-s2.1-pro-free"
        elif args.tts_provider == "gemini-live":
            selected_tts = "gemini-3.1-flash-live"
        else:
            selected_tts = "edge-tts"
        print(f"  [OK] Dubbing providers: ASR={selected_asr}, TTS={selected_tts}")

    print("[Preflight] All checks passed successfully.\n")
    return effective_device, input_path, output_path


def main():
    args = parse_args()
    effective_device, input_path, output_path = preflight_check(args)
    job_id = args.job_id or f"job_v2_{int(time.time())}"

    # Build feature requests based on preset.
    needs_dubbing = args.preset in {"dub-audio", "translate-video"}
    audio_req = AudioRequest(
        enabled=needs_dubbing,
        source_language=args.source_language,
        target_language=args.target_language,
        dubbing_enabled=needs_dubbing,
        voice=args.voice,
    )
    if args.preset in {"translate-subtitle", "translate-video"}:
        visual_req = VisualRequest(
            enabled=True,
            subtitle_enabled=True,
            subtitle_mode="translate_replace",
            target_language=args.target_language,
            logo_enabled=True,
            logo_action="preserve",
            watermark_enabled=True,
            watermark_action="preserve",
            other_text_enabled=True,
            other_text_action="preserve",
        )
    elif args.preset == "preserve-all":
        visual_req = VisualRequest(
            enabled=True,
            subtitle_enabled=True,
            subtitle_mode="preserve",
            target_language=args.target_language,
            logo_enabled=True,
            logo_action="preserve",
            watermark_enabled=True,
            watermark_action="preserve",
            other_text_enabled=True,
            other_text_action="preserve",
        )
    elif args.preset == "dub-audio":
        visual_req = VisualRequest(enabled=False, target_language=args.target_language)
    else:  # blur-all / default
        visual_req = VisualRequest(
            enabled=True,
            subtitle_enabled=True,
            subtitle_mode="blur",
            target_language=args.target_language,
            logo_enabled=True,
            logo_action="blur",
            watermark_enabled=True,
            watermark_action="blur",
            other_text_enabled=True,
            other_text_action="blur",
        )

    visual_req.subtitle_layout = args.subtitle_layout
    request = ProcessingRequest(
        job_id=job_id,
        input_video=str(input_path),
        output_video=str(output_path),
        audio=audio_req,
        visual=visual_req,
        device=effective_device,
        overwrite=args.overwrite,
    )

    print("=" * 60)
    print("[2/5 - 5/5] Running Modular Clean Architecture V2 Pipeline")
    print("=" * 60)
    print(f"  Preset: {args.preset}")
    print(f"  Visual Subtitle Mode: {visual_req.subtitle_mode}\n")

    # Wire replaceable providers at the composition root. Visual and audio translation
    # deliberately use separate service instances and separate prompts/context policies.
    video_util = FFmpegVideoAdapter()
    needs_visual_translation = args.preset in {"translate-subtitle", "translate-video"}
    needs_audio_translation = args.preset in {"dub-audio", "translate-video"}
    visual_translation_service = (
        _build_translation_service("subtitle", args.translation_provider)
        if needs_visual_translation
        else None
    )
    audio_translation_service = (
        _build_translation_service("spoken", args.translation_provider)
        if needs_audio_translation
        else None
    )

    detector_config = dict(
        weights_path=args.yolo_weights,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        device=effective_device,
        subtitle_rescue_conf=args.yolo_subtitle_rescue_conf,
        subtitle_rescue_max_gap_seconds=args.yolo_subtitle_rescue_max_gap_seconds,
        subtitle_rescue_min_support_ratio=args.yolo_subtitle_rescue_min_support_ratio,
    )
    detector = YoloDetectorAdapter(**detector_config) if visual_req.enabled else None
    ocr_provider = None
    if visual_req.enabled and visual_req.subtitle_mode == "translate_replace":
        from src.infrastructure.ocr.gemini_ocr import GeminiOCRAdapter
        ocr_provider = GeminiOCRAdapter()
    visual_caches = {}
    if visual_req.enabled and not args.no_visual_cache:
        from src.infrastructure.visual_cache_wiring import wire_visual_caches
        detector, ocr_provider, visual_translation_service, visual_caches = wire_visual_caches(
            BASE_DIR, args.visual_cache_dir, detector, ocr_provider,
            visual_translation_service, detector_config, args.translation_provider)
    subtitle_processor = SubtitleProcessor(
        ocr_provider=ocr_provider,
        translation_service=visual_translation_service,
    )
    from src.infrastructure.video.scene_boundaries import FrameDifferenceSceneDetector
    visual_pipeline = VisualPipeline(detector=detector, subtitle_processor=subtitle_processor,
                                     scene_detector=FrameDifferenceSceneDetector())

    audio_pipeline = AudioPipeline(audio_extractor=video_util)
    if needs_dubbing:
        from src.features.audio.composer import AudioComposer
        audio_pipeline = AudioPipeline(
            asr_provider=_build_asr_provider(args.asr_provider),
            asr_corrector=_build_asr_corrector(args.asr_correction, args.asr_provider),
            translation_service=audio_translation_service,
            spoken_rewriter=_build_spoken_rewriter(args.duration_rewrite),
            tts_provider=_build_tts_provider(args.tts_provider),
            composer=AudioComposer(max_speedup=args.max_tts_speedup),
            audio_extractor=video_util,
        )

    use_case = ProcessVideoUseCase(
        audio_pipeline=audio_pipeline,
        visual_pipeline=visual_pipeline,
        video_util=video_util,
    )
    result = use_case.execute(request)
    for stage, cache in visual_caches.items():
        print(f'Visual cache {stage}: hits={cache.hits}, misses={cache.misses}')

    print("=" * 60)
    print("E2E EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status:                      {result.status}")
    print(f"Input Video:                 {result.input_video_path}")
    print(f"Final Output Artifact:       {result.output_video_path}")
    print(f"Workspace Directory:         {result.workspace_dir}")
    print(f"Total Duration:              {result.timings.get('total', 0.0):.2f}s")
    print(f"Video Info:                  {result.resolution} @ {result.fps:.2f} fps ({result.duration_sec:.2f}s)")
    print(f"YOLO Detection Summary:")
    for cls_name, count in result.detections_summary.items():
        print(f"  - {cls_name}: {count}")
    print("=" * 60)


def run_job(args, *, runner: JobRunner | None = None):
    """CLI adapter over the same structured runner used by the local web."""
    selected_runner = runner or JobRunner()
    options = JobOptions(
        input_path=args.input,
        output_path=args.output,
        preset=args.preset,
        device=args.device,
        job_id=args.job_id,
        overwrite=args.overwrite,
        profile=args.profile,
        source_language=args.source_language,
        target_language=args.target_language,
        voice=args.voice,
        asr_provider=args.asr_provider,
        asr_correction=args.asr_correction,
        translation_provider=args.translation_provider,
        duration_rewrite=args.duration_rewrite,
        max_tts_speedup=args.max_tts_speedup,
        tts_provider=args.tts_provider,
        visual_cache_dir=args.visual_cache_dir,
        no_visual_cache=args.no_visual_cache,
        yolo_weights=args.yolo_weights,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_subtitle_rescue_conf=args.yolo_subtitle_rescue_conf,
        yolo_subtitle_rescue_max_gap_seconds=args.yolo_subtitle_rescue_max_gap_seconds,
        yolo_subtitle_rescue_min_support_ratio=args.yolo_subtitle_rescue_min_support_ratio,
        subtitle_layout=args.subtitle_layout,
    )
    return selected_runner.run(options, lambda event: print(format_event(event), flush=True))


def shared_main() -> None:
    _reexec_with_project_python()
    result = run_job(parse_args())
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


# Compatibility aliases for focused provider tests and the final dubbing router.
_has_openrouter_keys = _shared_has_openrouter_keys
_has_elevenlabs_keys = _shared_has_elevenlabs_keys
_build_translation_service = _shared_build_translation_service
_build_asr_provider = _shared_build_asr_provider
_build_asr_corrector = _shared_build_asr_corrector
_build_spoken_rewriter = _shared_build_spoken_rewriter
_build_tts_provider = _shared_build_tts_provider


if __name__ == "__main__":
    shared_main()
