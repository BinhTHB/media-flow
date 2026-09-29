"""Local MediaFlow web interface.

Serves the static frontend and runs the shared Python job service on background
threads. The server binds to localhost by default and never invokes a shell.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from src.interfaces.jobs.runner import JobEvent, JobOptions, JobRunner


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_env(path: Path | None = None) -> None:
    """Load project-local .env without overwriting explicit process variables.

    Multi-key pool variables are appended so process-level keys stay first.
    """
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            if key in {"OPENROUTER_API_KEYS", "FISH_API_KEYS", "ELEVENLABS_API_KEYS"} and value:
                existing = os.environ.get(key, "").strip()
                os.environ[key] = f"{existing},{value}" if existing else value
            else:
                os.environ.setdefault(key, value)


load_project_env()

FRONTEND_ROOT = PROJECT_ROOT / "frontend"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
UPLOAD_ROOT = PROJECT_ROOT / "input" / "uploads"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PRESETS = {"blur-all", "translate-subtitle", "preserve-all", "dub-audio", "translate-video", "blur-and-dub",
           "translate-video-generic", "translate-subtitle-generic"}
DEVICES = {"auto", "cuda", "cpu"}
DUBBING_LANGUAGES = {"auto", "zh", "en", "ja", "ko", "vi"}
DUBBING_SPEAKER_MODES = {"single", "auto"}
DUBBING_FIT_POLICIES = {"fit"}
DUBBING_VOICES = {
    "elevenlabs:EXAVITQu4vr4xnSDxMaL": "ElevenLabs v3 · Sarah · Nữ · Free API shortlist #1",
    "elevenlabs:nPczCjzI2devNBz1zQrb": "ElevenLabs v3 · Brian · Nam · Free API shortlist #2",
    "elevenlabs:pNInz6obpgDQGcFmaJgB": "ElevenLabs v3 · Adam · Nam · Free API",
    "elevenlabs:env": "ElevenLabs v3 · Voice ID từ .env",
    "fish:28902432d87243f08638ba59a1569781": "Fish · Nam trẻ tự nhiên · experimental",
    "fish:99dc73af301a419b8043e4fc401bd305": "Fish · Nam kể chuyện · shortlist #2",
    "fish:c5660c5f4f7b4b74be5f248a75d7fee6": "Fish · Nữ GenZ · voice cũ",
    "fish:e246b97671944988a54d53c43b74c64a": "Fish · Nam trầm, bình tĩnh",
    "fish:6ff0bb19af754d7684f9f4a74392e466": "Fish · Nam news-anchor",
    "gemini-live:": "Gemini 3.1 Flash Live · default voice",
    "gemini-live:Sulafat": "Gemini 3.1 Flash Live · Sulafat · Ấm",
    "gemini-live:Achird": "Gemini 3.1 Flash Live · Achird · Thân thiện",
    "gemini-live:Leda": "Gemini 3.1 Flash Live · Leda · Trẻ trung",
    "vi-VN-HoaiMyNeural": "Edge · Hoài My · Nữ · Việt Nam",
    "vi-VN-NamMinhNeural": "Edge · Nam Minh · Nam · Việt Nam",
}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_project_video(raw_path: str, *, must_exist: bool = True) -> Path:
    if not raw_path or not raw_path.strip():
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chưa chọn video đầu vào.")

    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    if not is_within(candidate, PROJECT_ROOT):
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            "Video phải nằm trong thư mục dự án MediaFlow.",
        )
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Định dạng video không được hỗ trợ.")
    if must_exist and not candidate.is_file():
        raise RequestError(HTTPStatus.NOT_FOUND, "Không tìm thấy video đầu vào.")
    return candidate


def safe_output_name(raw_name: str) -> str:
    name = Path(raw_name or "mediaflow-output.mp4").name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name:
        name = "mediaflow-output.mp4"
    if Path(name).suffix.lower() != ".mp4":
        name = f"{Path(name).stem}.mp4"
    return name


def safe_upload_name(raw_name: str) -> str:
    decoded = unquote(raw_name or "").strip()
    name = Path(decoded).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name or Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Tên hoặc định dạng video không hợp lệ.")
    return name


def relative_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def dubbing_capabilities() -> dict[str, Any]:
    """Return truthful capability state for the local dubbing surface."""
    try:
        import edge_tts  # noqa: F401
        edge_available = True
    except ImportError:
        edge_available = False

    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_elevenlabs = bool(
        os.getenv("ELEVENLABS_API_KEY")
        or os.getenv("ELEVENLABS_API_KEYS")
        or any(name.startswith("ELEVENLABS_API_KEY_") and value.strip() for name, value in os.environ.items())
    )
    has_openrouter = bool(
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEYS")
        or any(name.startswith("OPENROUTER_API_KEY_") and value.strip() for name, value in os.environ.items())
    )

    if has_groq:
        asr_provider = "groq-whisper-large-v3+gemini-fallback" if has_gemini else "groq-whisper-large-v3"
    else:
        asr_provider = "gemini-audio" if has_gemini else None

    if has_openrouter:
        translation_provider = "openrouter-free-pool+gemini-fallback" if has_gemini else "openrouter-free-pool"
    else:
        translation_provider = "gemini-3.5-flash" if has_gemini else None

    tts_route = []
    if has_elevenlabs:
        tts_route.append("elevenlabs-v3-free")
    if has_openrouter:
        tts_route.append("fish-s2.1-pro-free")
    if edge_available:
        tts_route.append("edge-tts")
    tts_provider = "+".join(tts_route) if tts_route else None

    ffmpeg_ready = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    ready = bool(asr_provider and translation_provider and tts_provider and ffmpeg_ready)

    return {
        "status": "ready" if ready else "partial",
        "ready": ready,
        "mode": "hybrid-free",
        "providers": {
            "asr": {
                "provider": asr_provider,
                "available": bool(asr_provider),
                "reason": "Ưu tiên Groq Whisper Large V3; Gemini Audio là fallback." if asr_provider else "Cần GEMINI_API_KEY hoặc GROQ_API_KEY.",
            },
            "diarization": {
                "provider": "elevenlabs-dubbing-v2" if has_elevenlabs else None,
                "available": has_elevenlabs,
                "reason": "ElevenLabs Dubbing v2 tự detect/giữ speaker identity; final quality router chỉ nhận output nếu transcript qua semantic gate." if has_elevenlabs else "Cần ELEVENLABS_API_KEYS để bật multi-speaker quality path.",
            },
            "translation": {
                "provider": translation_provider,
                "available": bool(translation_provider),
                "reason": "Ưu tiên OpenRouter free-model pool; Gemini là quality fallback." if translation_provider else "Cần OPENROUTER_API_KEY(S) hoặc GEMINI_API_KEY.",
            },
            "tts": {
                "provider": tts_provider,
                "available": bool(tts_provider),
                "reason": "Auto ưu tiên ElevenLabs v3 + language_code=vi khi có Free API key, sau đó Fish rồi Edge. Gemini Live chỉ explicit experimental vì không verbatim-safe." if tts_provider else "Cần ELEVENLABS_API_KEY, OpenRouter key pool, hoặc package edge-tts.",
            },
            "composer": {
                "provider": "ffmpeg",
                "available": ffmpeg_ready,
                "reason": "FFmpeg duration-fit/timeline composer sẵn sàng." if ffmpeg_ready else "Cần FFmpeg và FFprobe trong PATH.",
            },
        },
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "recommended_scope": "quality-auto: ElevenLabs Dubbing v2 multi-speaker + semantic gate, fallback MediaFlow khi transcript nguồn lệch",
        },
    }


def build_dubbing_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a dubbing request and describe the next pipeline stages.

    This is deliberately a planning contract, not a fake successful render.
    """
    input_path = resolve_project_video(str(payload.get("input_path", "")))
    source_language = str(payload.get("source_language", "auto"))
    target_language = str(payload.get("target_language", "vi"))
    speaker_mode = str(payload.get("speaker_mode", "single"))
    fit_policy = str(payload.get("fit_policy", "fit"))
    voice_id = str(payload.get("voice_id", "elevenlabs:EXAVITQu4vr4xnSDxMaL"))

    if source_language not in DUBBING_LANGUAGES:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Ngôn ngữ nguồn không được hỗ trợ.")
    if target_language not in DUBBING_LANGUAGES or target_language == "auto":
        raise RequestError(HTTPStatus.BAD_REQUEST, "Ngôn ngữ đích không hợp lệ.")
    if speaker_mode not in DUBBING_SPEAKER_MODES:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chế độ nhân vật không hợp lệ.")
    if fit_policy not in DUBBING_FIT_POLICIES:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chính sách khớp thời lượng không hợp lệ.")
    if voice_id not in DUBBING_VOICES:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Voice không được hỗ trợ.")

    preset = str(payload.get("preset", "translate-video")).removesuffix("-generic")
    capabilities = dubbing_capabilities()
    if preset == "translate-subtitle":
        required_provider_ids = ("asr", "translation", "composer")
    elif preset in {"blur-all", "preserve-all"}:
        required_provider_ids = ("composer",)
    else:
        required_provider_ids = ("asr", "translation", "tts", "composer")

    blockers = [
        capabilities["providers"][provider_id]["reason"]
        for provider_id in required_provider_ids
        if not capabilities["providers"][provider_id]["available"]
    ]
    if speaker_mode in {"multi", "diarization_required"} and not capabilities["providers"]["diarization"]["available"]:
        blockers.append(capabilities["providers"]["diarization"]["reason"])
    ready = not blockers
    stages = [
        {"id": "ingest", "label": "Nạp và kiểm tra media", "status": "ready"},
        {"id": "asr", "label": "Nhận diện lời thoại + timestamp", "status": "ready" if capabilities["providers"]["asr"]["available"] else "blocked"},
        {"id": "translate", "label": "Dịch theo từng đoạn", "status": "ready" if capabilities["providers"]["translation"]["available"] else "blocked"},
        {"id": "tts", "label": "Tạo voice", "status": "ready" if capabilities["providers"]["tts"]["available"] else "blocked"},
        {"id": "compose", "label": "Khớp thời lượng + ghép timeline", "status": "ready" if capabilities["providers"]["composer"]["available"] else "blocked"},
        {"id": "mux", "label": "Mux audio vào video", "status": "ready" if capabilities["providers"]["composer"]["available"] else "blocked"},
    ]

    segments: list[dict[str, Any]] = []
    speakers: list[dict[str, Any]] = []
    duration = 0.0

    try:
        from src.infrastructure.video.ffmpeg import probe_video
        probe = probe_video(str(input_path))
        duration = float(probe.get("duration", 0.0))
    except Exception:
        duration = 0.0

    # Phân tích audio ASR thực tế khi provider sẵn sàng
    temp_audio = None
    if capabilities["providers"]["asr"]["available"]:
        try:
            from src.infrastructure.video.ffmpeg import extract_audio
            from src.interfaces.composition.providers import build_asr_provider, build_translation_service

            temp_audio = OUTPUT_ROOT / f".plan_audio_{uuid.uuid4().hex[:8]}.wav"
            temp_audio.parent.mkdir(parents=True, exist_ok=True)
            extract_audio(str(input_path), str(temp_audio))

            asr = build_asr_provider("auto")
            raw_segments = asr.transcribe(str(temp_audio), language=source_language)

            if raw_segments:
                translations: list[str] = []
                if capabilities["providers"]["translation"]["available"]:
                    try:
                        trans_svc = build_translation_service("spoken", "auto")
                        texts = [s.source_text for s in raw_segments]
                        translations = trans_svc.translate_texts(
                            texts,
                            source_language=source_language,
                            target_language=target_language,
                        )
                    except Exception as trans_err:
                        sys.stderr.write(f"[dubbing_plan] Primary translation failed: {trans_err}; trying Gemini direct...\n")
                        try:
                            from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter
                            g_trans = GeminiTranslationAdapter(model="gemini-3.5-flash-lite", purpose="spoken")
                            translations = g_trans.translate_texts(
                                texts,
                                source_language=source_language,
                                target_language=target_language,
                            )
                        except Exception as g_err:
                            sys.stderr.write(f"[dubbing_plan] Gemini translation failed: {g_err}\n")
                            translations = [s.source_text for s in raw_segments]
                else:
                    translations = [s.source_text for s in raw_segments]

                spk_map: dict[str, int] = {}
                for idx, s in enumerate(raw_segments):
                    t_text = translations[idx] if idx < len(translations) else s.source_text
                    spk_id = s.speaker_id or "S01"
                    spk_map[spk_id] = spk_map.get(spk_id, 0) + 1

                    segments.append({
                        "id": s.segment_id or f"seg_{idx + 1}",
                        "start": round(s.start_time, 2),
                        "end": round(s.end_time, 2),
                        "source_text": s.source_text,
                        "target_text": t_text,
                        "speaker_id": spk_id,
                        "speaker_label": f"Speaker {spk_id.replace('S', '')}",
                    })

                for s_id in sorted(spk_map.keys()):
                    s_idx = s_id.replace("S", "")
                    speakers.append({
                        "id": s_id,
                        "label": f"Speaker {s_idx}",
                        "short": f"S{s_idx}",
                        "count": spk_map[s_id],
                        "voice_id": voice_id,
                    })
        except Exception as asr_err:
            sys.stderr.write(f"[dubbing_plan] ASR analysis error: {asr_err}\n")
        finally:
            if temp_audio and temp_audio.exists():
                try:
                    temp_audio.unlink(missing_ok=True)
                except OSError:
                    pass

    # Phân tích vị trí phụ đề gốc bằng YOLO detector nếu có weights
    subtitle_detections = []
    default_subtitle_bbox = None
    yolo_weights = PROJECT_ROOT / "models" / "yolo" / "best.pt"
    if not yolo_weights.is_file():
        alt = PROJECT_ROOT / "models" / "yolo11x_best.pt"
        if alt.is_file():
            yolo_weights = alt

    if yolo_weights.is_file():
        try:
            import torch
            from src.infrastructure.detection.yolo import YoloDetectorAdapter
            device = "cuda" if torch.cuda.is_available() else "cpu"
            detector = YoloDetectorAdapter(weights_path=str(yolo_weights), device=device)
            raw_dets = detector.detect_video(str(input_path), stride=5)
            subtitle_detections = [d for d in raw_dets if getattr(d, "class_name", "") == "subtitle"]
            if subtitle_detections:
                best_overall = max(subtitle_detections, key=lambda d: getattr(d, "confidence", 0.0))
                default_subtitle_bbox = {
                    "x1": round(best_overall.bbox.x1, 1),
                    "y1": round(best_overall.bbox.y1, 1),
                    "x2": round(best_overall.bbox.x2, 1),
                    "y2": round(best_overall.bbox.y2, 1),
                    "frame_width": best_overall.bbox.frame_width,
                    "frame_height": best_overall.bbox.frame_height,
                }
        except Exception as det_err:
            sys.stderr.write(f"[dubbing_plan] Subtitle YOLO detection error: {det_err}\n")

    # Ánh xạ bbox vào từng segment: CHỈ gán bbox khi thực sự có detection khớp thời gian
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        matched = [
            d for d in subtitle_detections
            if (seg_start - 0.5) <= getattr(d, "timestamp_seconds", 0.0) <= (seg_end + 0.5)
        ]
        if matched:
            best_det = max(matched, key=lambda d: getattr(d, "confidence", 0.0))
            seg["bbox"] = {
                "x1": round(best_det.bbox.x1, 1),
                "y1": round(best_det.bbox.y1, 1),
                "x2": round(best_det.bbox.x2, 1),
                "y2": round(best_det.bbox.y2, 1),
                "frame_width": best_det.bbox.frame_width,
                "frame_height": best_det.bbox.frame_height,
            }
        else:
            seg["bbox"] = None

    if not speakers:
        speakers = [{
            "id": "S01",
            "label": "Speaker 01",
            "short": "S1",
            "count": len(segments) or 1,
            "voice_id": voice_id,
        }]

    message = (
        f"Đã phân tích xong: {len(segments)} câu thoại, {len(speakers)} nhân vật sẵn sàng."
        if ready and segments
        else ("Pipeline dubbing single-speaker đã sẵn sàng." if ready else "Cấu hình dubbing còn thiếu provider bắt buộc.")
    )

    return {
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "message": message,
        "input_path": str(input_path),
        "input_relative": relative_project_path(input_path),
        "source_language": source_language,
        "target_language": target_language,
        "speaker_mode": speaker_mode,
        "fit_policy": fit_policy,
        "preserve_ambience": bool(payload.get("preserve_ambience", True)),
        "voice": {"id": voice_id, "label": DUBBING_VOICES[voice_id]},
        "stages": stages,
        "blockers": blockers,
        "capabilities": capabilities,
        "segments": segments,
        "speakers": speakers,
        "duration": duration,
        "default_subtitle_bbox": default_subtitle_bbox,
        "detected_subtitles_count": len(subtitle_detections),
    }


def build_job_options(job: dict[str, Any]) -> JobOptions:
    '''Map the validated web job to the shared Python execution contract.'''
    return JobOptions(
        input_path=job["input_path"],
        output_path=job["output_path"],
        preset=job["preset"],
        device=job["device"],
        job_id=job["id"],
        overwrite=job["overwrite"],
        profile=str(job.get("profile", "balanced")),
        source_language=str(job.get("source_language", "auto")),
        target_language=str(job.get("target_language", "vi")),
        voice=str(job.get("voice_id", "vi-VN-HoaiMyNeural")),
        asr_provider=str(job.get("asr_provider", "auto")),
        translation_provider=str(job.get("translation_provider", "auto")),
        tts_provider=str(job.get("tts_provider", "auto")),
        subtitle_layout=str(job.get("subtitle_layout", "temporal")),
        hide_source_subtitle=bool(job.get("hide_source_subtitle", True)),
        segments=job.get("segments"),
        speaker_voices=job.get("speaker_voices"),
        blur_extra=bool(job.get("blur_extra", False)),
        speaker_mode=str(job.get("speaker_mode", "single")),
        preserve_ambience=bool(job.get("preserve_ambience", True)),
        mix_original_volume=float(job.get("mix_original_volume", 0.20)),
    )


def _update_job(job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(updates)


def _handle_job_event(job_id: str, event: JobEvent) -> None:
    details = event.details or {}
    updates: dict[str, Any] = {
        "progress": event.progress,
        "message": event.message,
        "stage": event.stage,
    }
    if event.type == "started":
        updates.update(status="running", started_at=utc_now())
    elif event.type == "completed":
        result = details.get("result", {})
        updates.update(
            status="succeeded",
            detections=result.get("detections_summary", {}),
        )
    elif event.type == "failed":
        updates.update(status="failed", error=details.get("error"))
    with _JOBS_LOCK:
        current = _JOBS.get(job_id)
        if current is None:
            return
        current.update(updates)
        events = current.setdefault("events", [])
        events.append(event.to_dict())
        del events[:-120]
        logs = current.setdefault("logs", [])
        time_str = datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{time_str}] [{event.stage.upper()}] {event.message} ({event.progress}%)")
        del logs[:-200]


def _run_job(job_id: str) -> None:
    with _JOBS_LOCK:
        job = dict(_JOBS[job_id])
    started = time.monotonic()
    try:
        result = JobRunner().run(
            build_job_options(job),
            lambda event: _handle_job_event(job_id, event),
        )
        output = Path(result.output_video_path)
        _update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Video processing completed.",
            finished_at=utc_now(),
            duration_seconds=round(time.monotonic() - started, 2),
            output_size=output.stat().st_size if output.is_file() else None,
            pid=None,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            progress=100,
            message=str(exc),
            finished_at=utc_now(),
            duration_seconds=round(time.monotonic() - started, 2),
            pid=None,
        )

def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    raw_input = payload.get("input_path")
    if not raw_input:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chưa chọn video đầu vào.")
    input_path = resolve_project_video(str(raw_input))

    subtitle_layout = str(payload.get("subtitle_layout", "temporal"))
    if subtitle_layout in {"in-place", "temporal"}:
        subtitle_layout = "temporal"
    elif subtitle_layout in {"bottom", "legacy_centered"}:
        subtitle_layout = "bottom"
    else:
        subtitle_layout = "temporal"
    hide_source_subtitle = bool(payload.get("hide_source_subtitle", True))

    raw_presets = payload.get("presets")
    if "preset" in payload and str(payload.get("preset")) in PRESETS:
        preset = str(payload.get("preset"))
    elif isinstance(raw_presets, list) and raw_presets:
        has_blur = "blur-all" in raw_presets
        has_trans = "translate-subtitle" in raw_presets
        has_dub = "dub-audio" in raw_presets
        has_preserve = "preserve-all" in raw_presets
        if has_preserve and not (has_blur or has_trans or has_dub):
            preset = "preserve-all"
        elif (has_trans or has_blur) and has_dub:
            preset = "translate-video" if has_trans else "blur-and-dub"
        elif has_dub:
            preset = "dub-audio"
        elif has_trans:
            preset = "translate-subtitle"
        elif has_blur:
            preset = "blur-all"
        else:
            preset = "preserve-all"
    else:
        preset = str(payload.get("preset", "translate-video"))

    blur_extra = bool(
        payload.get("blur_extra", False)
        or (isinstance(raw_presets, list) and "blur-all" in raw_presets)
    )

    device = str(payload.get("device", "auto"))
    overwrite = bool(payload.get("overwrite", False))
    source_language = str(payload.get("source_language", "auto"))
    target_language = str(payload.get("target_language", "vi"))
    voice_id = str(payload.get("voice_id", "elevenlabs:EXAVITQu4vr4xnSDxMaL"))
    asr_provider = str(payload.get("asr_provider", "auto"))
    translation_provider = str(payload.get("translation_provider", "auto"))
    tts_provider = str(payload.get("tts_provider", "auto"))
    profile = str(payload.get("profile", "quality" if payload.get("dubbing_engine") == "quality-auto" else "balanced"))

    if preset not in PRESETS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Preset không hợp lệ.")
    if profile not in {"quality", "balanced", "fast"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Runtime profile không hợp lệ.")
    if device not in DEVICES:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Thiết bị xử lý không hợp lệ.")
    if preset == "translate-subtitle":
        if target_language not in DUBBING_LANGUAGES or target_language == "auto":
            raise RequestError(HTTPStatus.BAD_REQUEST, "Ngôn ngữ đích không hợp lệ.")
        if translation_provider not in {"auto", "openrouter", "gemini"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Translation provider không hợp lệ.")

    if preset in {"dub-audio", "translate-video", "blur-and-dub"}:
        if source_language not in DUBBING_LANGUAGES:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Ngôn ngữ nguồn không hợp lệ.")
        if target_language not in DUBBING_LANGUAGES or target_language == "auto":
            raise RequestError(HTTPStatus.BAD_REQUEST, "Ngôn ngữ đích không hợp lệ.")
        if voice_id not in DUBBING_VOICES:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Voice không được hỗ trợ.")
        if asr_provider not in {"auto", "groq", "gemini"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "ASR provider không hợp lệ.")
        if translation_provider not in {"auto", "openrouter", "gemini"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Translation provider không hợp lệ.")
        if tts_provider not in {"auto", "elevenlabs", "fish", "gemini-live", "edge"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "TTS provider không hợp lệ.")
        if not dubbing_capabilities()["ready"]:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Dubbing provider chưa sẵn sàng.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_name = safe_output_name(str(payload.get("output_name", "")))
    output_path = (OUTPUT_ROOT / output_name).resolve()
    if not is_within(output_path, OUTPUT_ROOT.resolve()):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Đường dẫn output không hợp lệ.")
    if output_path.exists() and not overwrite:
        raise RequestError(
            HTTPStatus.CONFLICT,
            "Output đã tồn tại. Bật ghi đè hoặc đổi tên file.",
        )

    job_id = f"web_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Đã xếp hàng.",
        "input_path": str(input_path),
        "input_relative": relative_project_path(input_path),
        "output_path": str(output_path),
        "output_relative": relative_project_path(output_path),
        "output_name": output_name,
        "preset": preset,
        "profile": profile,
        "device": device,
        "overwrite": overwrite,
        "source_language": source_language,
        "target_language": target_language,
        "voice_id": voice_id,
        "asr_provider": asr_provider,
        "translation_provider": translation_provider,
        "tts_provider": tts_provider,
        "subtitle_layout": subtitle_layout,
        "hide_source_subtitle": hide_source_subtitle,
        "segments": payload.get("segments") if bool(payload.get("use_custom_segments", False)) else None,
        "speaker_voices": payload.get("speaker_voices") if isinstance(payload.get("speaker_voices"), dict) else {},
        "blur_extra": blur_extra,
        "speaker_mode": str(payload.get("speaker_mode", "single")),
        "preserve_ambience": bool(payload.get("preserve_ambience", True)),
        "mix_original_volume": float(payload.get("mix_original_volume", 0.20)),
        "dubbing_engine": str(payload.get("dubbing_engine", "quality-auto")),
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "output_size": None,
        "detections": {},
        "events": [],
        "logs": [],
        "pid": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return dict(job)


def list_project_videos() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    roots = [PROJECT_ROOT / "input", PROJECT_ROOT / "test_videos", OUTPUT_ROOT]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append(
                {
                    "name": path.name,
                    "path": relative_project_path(path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "source": relative_project_path(root),
                }
            )
            if len(items) >= 150:
                return items
    return items


class MediaFlowHandler(BaseHTTPRequestHandler):
    server_version = "MediaFlowLocal/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[web] %s\n" % (format % args))

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length không hợp lệ.") from exc
        if length <= 0 or length > MAX_JSON_BYTES:
            raise RequestError(HTTPStatus.BAD_REQUEST, "JSON request không hợp lệ.")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "JSON request không hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "JSON request phải là object.")
        return payload

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        total = resolved.stat().st_size
        start = 0
        end = max(0, total - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")

        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
            if not match or not (match.group(1) or match.group(2)):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
            else:
                # Suffix form "bytes=-N": serve the last N bytes.
                start = max(0, total - int(match.group(2)))
            if match.group(1) and match.group(2):
                end = min(int(match.group(2)), end)
            if start > end or start >= total:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT

        mime = content_type or mimetypes.guess_type(resolved.name)[0]
        mime = mime or "application/octet-stream"
        length = end - start + 1

        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if resolved.suffix in {".html", ".js", ".css", ".json"}:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()

        with resolved.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ConnectionError):
                    return
                remaining -= len(chunk)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/dubbing/capabilities":
                self._send_json(dubbing_capabilities())
                return

            if path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "project": "MediaFlow",
                        "presets": sorted(PRESETS),
                        "devices": ["auto", "cuda", "cpu"],
                    }
                )
                return

            if path == "/api/files":
                self._send_json({"files": list_project_videos()})
                return

            if path == "/api/jobs":
                with _JOBS_LOCK:
                    jobs = [dict(job) for job in _JOBS.values()]
                jobs.sort(key=lambda item: item["created_at"], reverse=True)
                self._send_json({"jobs": jobs[:30]})
                return

            if path.startswith("/api/jobs/"):
                job_id = path.split("/")[3] if len(path.split("/")) > 3 else ""
                with _JOBS_LOCK:
                    job = dict(_JOBS.get(job_id, {}))
                if not job:
                    raise RequestError(HTTPStatus.NOT_FOUND, "Không tìm thấy job.")
                self._send_json(job)
                return

            if path == "/api/media":
                raw = parse_qs(parsed.query).get("path", [""])[0]
                video_path = resolve_project_video(raw)
                self._serve_file(video_path)
                return

            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            # MediaFlow CLI Studio is the primary landing surface; /index.html and /dubbing.html route to studio.html.
            if path in {"/", "/index.html", "/dubbing.html"}:
                static_path = "studio.html"
            else:
                static_path = path.lstrip("/")
            candidate = (FRONTEND_ROOT / static_path).resolve()
            if not is_within(candidate, FRONTEND_ROOT.resolve()) or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(candidate)
        except RequestError as exc:
            self._send_error_json(exc.status, exc.message)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ConnectionError):
            return
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/dubbing/voice-preview":
                self._handle_voice_preview()
                return

            if parsed.path == "/api/dubbing/plan":
                plan = build_dubbing_plan(self._read_json())
                self._send_json(plan)
                return

            if parsed.path == "/api/jobs":
                job = create_job(self._read_json())
                self._send_json(job, HTTPStatus.ACCEPTED)
                return

            if parsed.path == "/api/uploads":
                self._handle_upload()
                return

            self._send_error_json(HTTPStatus.NOT_FOUND, "Endpoint không tồn tại.")
        except RequestError as exc:
            self._send_error_json(exc.status, exc.message)
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/files":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                if raw_path:
                    # Xóa 1 file video cụ thể
                    target = resolve_project_video(raw_path, must_exist=True)
                    input_dir = (PROJECT_ROOT / "input").resolve()
                    output_dir = OUTPUT_ROOT.resolve()
                    if not (is_within(target, input_dir) or is_within(target, output_dir)):
                        raise RequestError(HTTPStatus.FORBIDDEN, "Không thể xóa file hệ thống hoặc test_videos mẫu.")
                    try:
                        target.unlink()
                    except OSError as exc:
                        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, f"Không thể xóa file: {exc}") from exc
                    self._send_json({"deleted": relative_project_path(target), "success": True})
                    return
                else:
                    # Xóa toàn bộ lịch sử video tải lên và video kết quả (giữ input gốc và test_videos mẫu)
                    deleted_files: list[str] = []
                    for root_dir in [UPLOAD_ROOT, OUTPUT_ROOT]:
                        if root_dir.is_dir():
                            for f in list(root_dir.rglob("*")):
                                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                                    try:
                                        rel = relative_project_path(f)
                                        f.unlink()
                                        deleted_files.append(rel)
                                    except OSError:
                                        pass
                    self._send_json({"cleared": True, "deleted_count": len(deleted_files), "files": deleted_files})
                    return

            self._send_error_json(HTTPStatus.NOT_FOUND, "Endpoint không tồn tại.")
        except RequestError as exc:
            self._send_error_json(exc.status, exc.message)
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _handle_voice_preview(self) -> None:
        payload = self._read_json()
        voice_id = str(payload.get("voice_id") or "elevenlabs:EXAVITQu4vr4xnSDxMaL").strip()
        if voice_id not in DUBBING_VOICES:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Voice preview không hợp lệ.")
        raw_text = payload.get("text")
        if raw_text is None:
            text = "Cảm ơn anh. Không có gì. Xin lỗi, tôi giao nhầm đơn rồi."
        else:
            text = str(raw_text).strip()
        if not text or len(text) > 300:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Preview text phải có 1–300 ký tự.")

        if voice_id.startswith("elevenlabs:"):
            from src.infrastructure.tts.elevenlabs_tts import ElevenLabsTTSAdapter
            provider = ElevenLabsTTSAdapter()
        elif voice_id.startswith("fish:"):
            from src.infrastructure.tts.openrouter_tts import OpenRouterTTSAdapter
            provider = OpenRouterTTSAdapter()
        elif voice_id.startswith("gemini-live:"):
            from src.infrastructure.tts.gemini_flash_live import GeminiFlashLiveTTSAdapter
            provider = GeminiFlashLiveTTSAdapter()
        else:
            from src.infrastructure.tts.edge_tts import EdgeTTSAdapter
            provider = EdgeTTSAdapter()

        preview_dir = OUTPUT_ROOT / "voice_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        output = preview_dir / f"preview-{uuid.uuid4().hex[:12]}.mp3"
        try:
            provider.synthesize(text=text, output_path=str(output), voice=voice_id)
            self._serve_file(output, content_type="audio/mpeg")
        finally:
            output.unlink(missing_ok=True)

    def _handle_upload(self) -> None:
        filename = safe_upload_name(self.headers.get("X-Filename", ""))
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length không hợp lệ.") from exc
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Video rỗng hoặc vượt giới hạn 4 GB.",
            )

        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        destination = (UPLOAD_ROOT / filename).resolve()
        if destination.exists():
            destination = destination.with_name(
                f"{destination.stem}-{uuid.uuid4().hex[:6]}{destination.suffix}"
            )

        remaining = length
        try:
            with destination.open("wb") as output:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RequestError(
                            HTTPStatus.BAD_REQUEST,
                            "Upload bị gián đoạn trước khi hoàn tất.",
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        self._send_json(
            {
                "name": destination.name,
                "path": relative_project_path(destination),
                "size": destination.stat().st_size,
            },
            HTTPStatus.CREATED,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MediaFlow web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not FRONTEND_ROOT.is_dir():
        raise SystemExit(f"Frontend directory not found: {FRONTEND_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), MediaFlowHandler)
    server.daemon_threads = True
    print(f"MediaFlow web: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
