"""Local MediaFlow web interface.

Serves the static frontend and runs the shared Python job service on background
threads. The server binds to localhost by default and never invokes a shell.
"""

from __future__ import annotations

import argparse
import json
import logging
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

from src.interfaces.jobs.runner import (
    LEGACY_PRESET_FEATURES,
    JobEvent,
    JobOptions,
    JobRunner,
    default_jobs_dir,
    resolve_preset,
)
from src.interfaces.jobs.store import JobStore


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
PRESETS = {"yolo-4class", "yolo-text"}
SELECTABLE_PRESETS = PRESETS | {"reup"}
SUBTITLE_ACTIONS = {"translate", "blur", "preserve"}
EXTRA_TEXT_ACTIONS = {"blur", "preserve"}
DEVICES = {"auto", "cuda", "cpu"}
DUBBING_LANGUAGES = {"auto", "zh", "en", "ja", "ko", "vi"}
DUBBING_SPEAKER_MODES = {"single", "auto"}
DUBBING_FIT_POLICIES = {"fit"}
DEFAULT_WEB_VOICE = "elevenlabs:nPczCjzI2devNBz1zQrb"
DUBBING_VOICES = {
    DEFAULT_WEB_VOICE: "ElevenLabs v3 · Brian · Nam · Free API",
    "elevenlabs:cKxI7IOUsPeiDfNgV1cE": "ElevenLabs · Tung Anh · Nam · tiếng Việt (cần gói trả phí)",
    "elevenlabs:EXAVITQu4vr4xnSDxMaL": "ElevenLabs v3 · Sarah · Nữ · Free API shortlist #1",
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

#: Durable job state lives here; JobRunner writes the same directory, so the web
#: UI can list jobs that finished before the current process started.
JOB_STORE = JobStore(default_jobs_dir())

#: Job statuses that no longer change. Mirrors JobStore's terminal set.
_TERMINAL_STATUSES = {"SUCCESS", "PARTIAL", "FAILED", "BLOCKED"}
#: Web-level statuses that mean the background thread is done.
_TERMINAL_WEB_STATUSES = {"succeeded", "failed"}


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
    voice_id = str(payload.get("voice_id", DEFAULT_WEB_VOICE))

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

    resolved_preset, subtitle_action, dub_audio, _extra_text = resolve_preset(
        str(payload.get("preset", "translate-video")),
        str(payload.get("subtitle_action", "translate")),
        bool(payload.get("dub_audio", False)),
        str(payload.get("extra_text_action", "preserve")),
        blur_extra=bool(payload.get("blur_extra", False)),
    )
    capabilities = dubbing_capabilities()
    if dub_audio:
        required_provider_ids = ("asr", "translation", "tts", "composer")
    elif subtitle_action == "translate":
        required_provider_ids = ("asr", "translation", "composer")
    else:
        required_provider_ids = ("composer",)

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
    # Resolve here so persisted jobs created before the flow-preset refactor
    # (and hand-built job dicts) normalize exactly like fresh API calls.
    requested = str(job.get("preset", "yolo-4class"))
    reup_contract = requested.removesuffix("-generic") == "reup" or bool(job.get("reup_contract"))
    try:
        preset, subtitle_action, dub_audio, extra_text_action = resolve_preset(
            str(job.get("preset", "yolo-4class")),
            str(job.get("subtitle_action", "translate")),
            bool(job.get("dub_audio", False)),
            str(job.get("extra_text_action", "preserve")),
            blur_extra=bool(job.get("blur_extra", False)),
        )
    except ValueError:
        # Unknown preset: pass through verbatim and let the runner raise the
        # canonical validation error.
        preset = str(job.get("preset", "yolo-4class"))
        subtitle_action = str(job.get("subtitle_action", "translate"))
        dub_audio = bool(job.get("dub_audio", False))
        extra_text_action = str(job.get("extra_text_action", "preserve"))
    return JobOptions(
        input_path=job["input_path"],
        output_path=job["output_path"],
        preset=preset,
        subtitle_action=subtitle_action,
        dub_audio=dub_audio,
        extra_text_action=extra_text_action,
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
        max_tts_speedup=float(job.get("max_tts_speedup", 1.30)),
        reup_contract=reup_contract,
        variation_level=str(job.get("variation_level") or "balanced"),
        create_missing_subtitles=bool(job.get("create_missing_subtitles", True)),
    )


def _update_job(job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(updates)


#: Map the durable runner status onto the web UI vocabulary. The store keeps
#: the pipeline's own SUCCESS/PARTIAL/FAILED/BLOCKED vocabulary; the UI only
#: distinguishes "still running" from "done, look at the result".
_STORE_STATUS_TO_WEB = {
    "SUCCESS": "succeeded",
    "PARTIAL": "succeeded",
    "FAILED": "failed",
    "BLOCKED": "failed",
    "RUNNING": "running",
}


def _store_job_message(state: dict[str, Any]) -> str:
    """Text the dock should show after a refresh.

    A failed run leaves ``stage`` on the last progress checkpoint, so the
    failure sentence lives on ``error`` and the last event, not on ``stage``.
    """
    error = state.get("error")
    if isinstance(error, dict) and str(error.get("message") or "").strip():
        return str(error["message"]).strip()
    explicit = str(state.get("message") or "").strip()
    if explicit:
        return explicit
    events = state.get("events") or []
    if events and isinstance(events[-1], dict):
        last = str(events[-1].get("message") or "").strip()
        if last:
            return last
    return str(state.get("stage") or "")


def _store_job_logs(state: dict[str, Any]) -> list[str]:
    logs: list[str] = []
    for event in (state.get("events") or [])[-8:]:
        if not isinstance(event, dict):
            continue
        text = str(event.get("message") or "").strip()
        if text:
            logs.append(text)
    return logs


def _web_job_from_store(state: dict[str, Any]) -> dict[str, Any]:
    """Project a durable ``JobStore`` state document onto the web job shape.

    The store is written by ``JobRunner`` for both CLI and web jobs, so this is
    what lets the UI list work that finished before this process started.
    """
    request = state.get("request") or {}
    output_path = str(request.get("output_video") or "")
    input_path = str(request.get("input_video") or "")
    metrics = state.get("metrics") or {}
    output_size = None
    if output_path:
        try:
            candidate = Path(output_path)
            if candidate.is_file():
                output_size = candidate.stat().st_size
        except OSError:
            output_size = None
    return {
        "id": str(state.get("id", "")),
        "status": _STORE_STATUS_TO_WEB.get(str(state.get("status", "")), "failed"),
        "progress": int(state.get("progress") or 0),
        "message": _store_job_message(state),
        "stage": str(state.get("stage") or ""),
        "input_path": input_path,
        "input_relative": relative_project_path(Path(input_path)) if input_path else "",
        "output_path": output_path,
        "output_relative": relative_project_path(Path(output_path)) if output_path else "",
        "output_name": Path(output_path).name if output_path else "",
        "output_size": output_size,
        "detections": metrics.get("detections", {}),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "duration_seconds": metrics.get("duration_seconds"),
        "error": state.get("error"),
        "events": state.get("events", []),
        "logs": _store_job_logs(state),
        "persisted": True,
    }


def list_jobs(limit: int = 30) -> list[dict[str, Any]]:
    """Live in-memory jobs merged with durable history, newest first.

    In-memory entries win: they carry live log lines and progress that the
    store only receives on stage boundaries.
    """
    with _JOBS_LOCK:
        live = {job_id: dict(job) for job_id, job in _JOBS.items()}
    merged = dict(live)
    for state in JOB_STORE.list():
        job_id = str(state.get("id", ""))
        if not job_id or job_id in merged:
            continue
        merged[job_id] = _web_job_from_store(state)

    def _sort_key(item: dict[str, Any]) -> float:
        for field in ("updated_at", "created_at"):
            value = item.get(field)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
        return 0.0

    jobs = sorted(merged.values(), key=_sort_key, reverse=True)
    # In-memory jobs carry the live log. A full history page must not push
    # them off the list: their timestamps are ISO strings, which used to sort as 0.
    live_jobs = [job for job in jobs if not job.get("persisted")]
    history = [job for job in jobs if job.get("persisted")]
    return (live_jobs + history)[:limit]


def get_job(job_id: str) -> dict[str, Any]:
    """Return one job from memory, falling back to durable history."""
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id, {}))
    if job:
        return job
    try:
        state = JOB_STORE.get(job_id)
    except ValueError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Job id không hợp lệ.") from exc
    if not state:
        raise RequestError(HTTPStatus.NOT_FOUND, "Không tìm thấy job.")
    return _web_job_from_store(state)


def cancel_job(job_id: str) -> dict[str, Any]:
    """Request cooperative cancellation of a running job.

    The runner checks this marker between stages and aborts; a job that has
    already finished is reported as such rather than silently "cancelled".
    """
    job = get_job(job_id)
    if job.get("status") in _TERMINAL_WEB_STATUSES:
        return {"cancelled": False, "status": job.get("status"), "message": "Job đã kết thúc."}
    try:
        JOB_STORE.request_cancel(job_id)
    except ValueError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Job id không hợp lệ.") from exc
    _update_job(job_id, message="Đã yêu cầu hủy job.", cancel_requested=True)
    return {"cancelled": True, "status": job.get("status"), "message": "Đã gửi yêu cầu hủy."}


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


# Loggers whose records belong to the pipeline run and are useful in the
# web terminal: pipeline stage boundaries, render/TTS progress, provider
# warnings and errors.
_PIPELINE_LOGGERS = (
    "src.application.process_video",
    "src.features.visual.pipeline",
    "src.features.audio.pipeline",
    "src.infrastructure.ocr.gemini_ocr",
    "src.infrastructure.translation.gemini_translation",
    "src.infrastructure.translation.openrouter_translation",
    "src.infrastructure.translation.router",
)


class _JobLogHandler(logging.Handler):
    """Stream pipeline log records into a job's web terminal log.

    Records may carry ``mediaflow_pct`` (0-100 progress of the reporting
    pipeline) and ``mediaflow_stage`` (``visual`` / ``audio``) extras; the
    percentage is mapped onto the window between the runner's coarse stage
    events (40% -> 90%) with a monotonic guard.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self._job_id = job_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            time_str = datetime.now().strftime("%H:%M:%S")
            source = record.name.rsplit(".", 1)[-1].upper()
            line = f"[{time_str}] [{source}] {record.getMessage()}"
            pct = getattr(record, "mediaflow_pct", None)
            stage = getattr(record, "mediaflow_stage", None)
            with _JOBS_LOCK:
                current = _JOBS.get(self._job_id)
                if current is None:
                    return
                logs = current.setdefault("logs", [])
                logs.append(line)
                del logs[:-300]
                if pct is not None:
                    window_start = 82 if stage == "audio" else 40
                    window_span = 0.08 if stage == "audio" else 0.5
                    mapped = window_start + int(max(0, min(100, int(pct))) * window_span)
                    if mapped > int(current.get("progress") or 0):
                        current["progress"] = mapped
                        current["message"] = record.getMessage()
        except Exception:
            self.handleError(record)


def _attach_job_log_handler(job_id: str) -> _JobLogHandler:
    handler = _JobLogHandler(job_id)
    for name in _PIPELINE_LOGGERS:
        pipeline_logger = logging.getLogger(name)
        pipeline_logger.addHandler(handler)
        pipeline_logger.setLevel(logging.INFO)
    return handler


def _detach_job_log_handler(handler: _JobLogHandler) -> None:
    for name in _PIPELINE_LOGGERS:
        pipeline_logger = logging.getLogger(name)
        pipeline_logger.removeHandler(handler)
        if not pipeline_logger.handlers:
            pipeline_logger.setLevel(logging.NOTSET)


def _run_job(job_id: str) -> None:
    with _JOBS_LOCK:
        job = dict(_JOBS[job_id])
    started = time.monotonic()
    log_handler = _attach_job_log_handler(job_id)
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
    finally:
        _detach_job_log_handler(log_handler)

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
    requested_preset = str(payload.get("preset", ""))
    is_legacy_preset = (
        requested_preset not in PRESETS
        and requested_preset.removesuffix("-generic") in LEGACY_PRESET_FEATURES
    )
    if isinstance(raw_presets, list) and raw_presets and not is_legacy_preset:
        # Legacy clients describe the job as a list of feature presets.
        has_blur = "blur-all" in raw_presets
        has_trans = "translate-subtitle" in raw_presets
        has_dub = "dub-audio" in raw_presets
        has_preserve = "preserve-all" in raw_presets
        if has_preserve and not (has_blur or has_trans or has_dub):
            requested_preset = "preserve-all"
        elif (has_trans or has_blur) and has_dub:
            requested_preset = "translate-video" if has_trans else "blur-and-dub"
        elif has_dub:
            requested_preset = "dub-audio"
        elif has_trans:
            requested_preset = "translate-subtitle"
        elif has_blur:
            requested_preset = "blur-all"
        else:
            requested_preset = "preserve-all"
    elif not requested_preset:
        requested_preset = "translate-video"

    blur_extra = bool(
        payload.get("blur_extra", False)
        or (isinstance(raw_presets, list) and "blur-all" in raw_presets)
    )
    subtitle_action = str(payload.get("subtitle_action", ""))
    reup_requested = requested_preset.removesuffix("-generic") == "reup"
    if "create_missing_subtitles" in payload:
        create_missing_subtitles = bool(payload.get("create_missing_subtitles"))
    else:
        create_missing_subtitles = True
    if reup_requested and "dub_audio" not in payload:
        dub_audio = True
    else:
        dub_audio = bool(payload.get("dub_audio", False))
    extra_text_action = str(payload.get("extra_text_action", ""))
    try:
        preset, subtitle_action, dub_audio, extra_text_action = resolve_preset(
            requested_preset,
            subtitle_action or "translate",
            dub_audio,
            extra_text_action or ("blur" if blur_extra or reup_requested else "preserve"),
            blur_extra=blur_extra,
        )
    except ValueError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Preset không hợp lệ.") from exc

    device = str(payload.get("device", "auto"))
    overwrite = bool(payload.get("overwrite", False))
    source_language = str(payload.get("source_language", "auto"))
    target_language = str(payload.get("target_language", "vi"))
    voice_id = str(payload.get("voice_id", DEFAULT_WEB_VOICE))
    asr_provider = str(payload.get("asr_provider", "auto"))
    translation_provider = str(payload.get("translation_provider", "auto"))
    tts_provider = str(payload.get("tts_provider", "auto"))
    profile = str(payload.get("profile", "quality" if payload.get("dubbing_engine") == "quality-auto" else "balanced"))

    if preset not in PRESETS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Preset không hợp lệ.")
    if subtitle_action not in SUBTITLE_ACTIONS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Hành động phụ đề không hợp lệ.")
    if requested_preset.removesuffix("-generic") == "reup" and subtitle_action == "preserve" and not dub_audio:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chọn xử lý hình, tiếng, hoặc cả hai.")
    if extra_text_action not in EXTRA_TEXT_ACTIONS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Hành động text khác không hợp lệ.")
    if profile not in {"quality", "balanced", "fast"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Runtime profile không hợp lệ.")
    if device not in DEVICES:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Thiết bị xử lý không hợp lệ.")
    variation_level = str(payload.get("variation_level") or "balanced")
    if variation_level not in {"off", "light", "balanced", "aggressive"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Mức biến thể không hợp lệ.")
    if subtitle_action == "translate":
        if target_language not in DUBBING_LANGUAGES or target_language == "auto":
            raise RequestError(HTTPStatus.BAD_REQUEST, "Ngôn ngữ đích không hợp lệ.")
        if translation_provider not in {"auto", "openrouter", "gemini"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Translation provider không hợp lệ.")

    if dub_audio:
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
        "reup_contract": requested_preset.removesuffix("-generic") == "reup",
        "create_missing_subtitles": create_missing_subtitles,
        "subtitle_action": subtitle_action,
        "dub_audio": dub_audio,
        "extra_text_action": extra_text_action,
        "profile": profile,
        "variation_level": variation_level,
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
        "preserve_ambience": False if requested_preset.removesuffix("-generic") == "reup" and dub_audio else bool(payload.get("preserve_ambience", True)),
        "mix_original_volume": 0.0 if requested_preset.removesuffix("-generic") == "reup" and dub_audio else float(payload.get("mix_original_volume", 0.20)),
        "max_tts_speedup": 1.8 if requested_preset.removesuffix("-generic") == "reup" and dub_audio else float(payload.get("max_tts_speedup", 1.30)),
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


ZERNIO_STORE_PATH = PROJECT_ROOT / "secrets" / "zernio_accounts.json"
SOURCES_PATH = PROJECT_ROOT / "input" / "sources.txt"
_INGEST_LOCK = threading.Lock()
_INGEST_STATE: dict[str, Any] = {"status": "idle", "message": "", "exit_code": None, "log": []}


def zernio_store():
    from src.infrastructure.publishing.zernio_store import ZernioAccountStore

    return ZernioAccountStore(ZERNIO_STORE_PATH)


def ingest_status() -> dict[str, Any]:
    with _INGEST_LOCK:
        return dict(_INGEST_STATE)


def start_source_poll(text: str, limit: int, then_process: bool) -> dict[str, Any]:
    from src.features.ingest.watch import parse_watch_text

    body = (text or "").strip()
    if not body:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chưa có nguồn.")
    try:
        parse_watch_text(body)
    except ValueError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    if limit < 1 or limit > 50:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Giới hạn mỗi nguồn từ 1 đến 50.")
    with _INGEST_LOCK:
        if _INGEST_STATE["status"] == "running":
            raise RequestError(HTTPStatus.CONFLICT, "Đang tải nguồn.")
        _INGEST_STATE.update({
            "status": "running",
            "message": "Đang xem nguồn mới.",
            "exit_code": None,
            "log": [],
        })
    threading.Thread(
        target=_run_source_poll,
        args=(body, limit, then_process),
        daemon=True,
    ).start()
    return ingest_status()


def _run_source_poll(text: str, limit: int, then_process: bool) -> None:
    import io
    from argparse import Namespace
    from contextlib import redirect_stdout

    from src.features.ingest.service import IngestService
    from src.infrastructure.ingest.yt_dlp import YtDlpDownloader
    from src.interfaces.ingest import _poll_once

    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(text + "\n", encoding="utf-8")
    args = Namespace(
        url=[],
        urls_file=None,
        sources=str(SOURCES_PATH),
        out=str(PROJECT_ROOT / "input" / "downloads"),
        quality="1080",
        trim=0.0,
        subtitles=False,
        overwrite=False,
        limit=limit,
        interval=0,
        json=False,
        then_process=then_process,
        preset="reup",
        subtitle_action="translate",
        extra_text_action="blur",
        dub_audio=True,
        device="auto",
        max_attempts=3,
        backoff_base=5.0,
    )
    downloader = YtDlpDownloader()
    service = IngestService(downloader)
    buffer = io.StringIO()
    code = 1
    try:
        with redirect_stdout(buffer):
            code = _poll_once(args, downloader, service)
        message = "Đã tải nguồn mới." if code == 0 else "Tải nguồn có lỗi."
        status = "succeeded" if code == 0 else "failed"
    except Exception as exc:
        message = str(exc)
        status = "failed"
    with _INGEST_LOCK:
        _INGEST_STATE.update({
            "status": status,
            "message": message,
            "exit_code": code,
            "log": buffer.getvalue().splitlines()[-80:],
        })


def add_zernio_account(payload: dict[str, Any]) -> dict[str, str]:
    try:
        return zernio_store().add(
            label=str(payload.get("label") or ""),
            platform=str(payload.get("platform") or ""),
            api_key=str(payload.get("api_key") or ""),
        )
    except ValueError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc


def check_zernio_account(account_id: str) -> dict[str, Any]:
    from src.infrastructure.publishing.zernio import ZernioClient, ZernioError

    row = zernio_store().get(account_id)
    if row is None:
        raise RequestError(HTTPStatus.NOT_FOUND, "Không thấy API key.")
    try:
        connected = ZernioClient(row["api_key"]).list_accounts()
    except ZernioError as exc:
        raise RequestError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
    matched = [item for item in connected if item["platform"] == row["platform"]]
    return {
        "id": account_id,
        "platform": row["platform"],
        "connected": connected,
        "matched": matched,
    }


def spoken_lines_for_video(video: Path, jobs_root: Path | None = None) -> list[str]:
    """Collect spoken/translated lines from the job that produced ``video``."""
    root = jobs_root or default_jobs_dir()
    if not root.is_dir():
        return []
    target = str(video.resolve())
    for job_dir in sorted(root.iterdir(), reverse=True):
        result_path = job_dir / "result.json"
        segments_path = job_dir / "audio" / "segments.json"
        if not result_path.is_file() or not segments_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        output = str(result.get("output_video_path") or "")
        if output and Path(output).resolve() != Path(target):
            continue
        try:
            rows = json.loads(segments_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        lines: list[str] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("spoken_text") or row.get("translated_text") or row.get("source_text") or "").strip()
                if text:
                    lines.append(text)
        return lines
    return []


def suggest_publish_metadata(video_path: str) -> dict[str, Any]:
    from src.infrastructure.publishing.gemini_metadata import GeminiMetadataGenerator

    video = resolve_project_video(video_path)
    lines = spoken_lines_for_video(video)
    transcript = " | ".join(lines[:40])
    metadata = GeminiMetadataGenerator().generate(
        str(video),
        {"transcript": transcript} if transcript else {},
    )
    payload = {
        "title": metadata.title,
        "description": metadata.description,
        "tags": list(metadata.tags),
        "language": metadata.language,
    }
    sidecar = video.with_suffix(".publish.json")
    try:
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return payload


def publish_zernio(payload: dict[str, Any]) -> dict[str, Any]:
    from src.infrastructure.publishing.zernio import ZernioClient, ZernioError

    video = resolve_project_video(str(payload.get("video_path") or ""))
    account_ids = payload.get("account_ids")
    if not isinstance(account_ids, list) or not account_ids:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Chọn ít nhất một API key.")
    title = str(payload.get("title") or "")
    content = str(payload.get("content") or "")
    raw_tags = payload.get("tags")
    if isinstance(raw_tags, str):
        tags = [part.strip() for part in raw_tags.replace("#", " ").split(",") if part.strip()]
    elif isinstance(raw_tags, list):
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    else:
        tags = []
    if not title.strip() or not content.strip() or not tags:
        suggested = suggest_publish_metadata(str(video))
        title = title.strip() or str(suggested.get("title") or "")
        content = content.strip() or str(suggested.get("description") or "")
        if not tags:
            tags = [str(tag) for tag in suggested.get("tags") or []]
    results: list[dict[str, Any]] = []
    for account_id in account_ids:
        row = zernio_store().get(str(account_id))
        if row is None:
            results.append({"id": str(account_id), "ok": False, "error": "Không thấy API key."})
            continue
        try:
            published = ZernioClient(row["api_key"]).publish_video(
                video_path=str(video),
                platform=row["platform"],
                title=title,
                content=content,
                tags=tags,
            )
            results.append({"id": row["id"], "label": row["label"], "ok": True, **published})
        except ZernioError as exc:
            results.append({"id": row["id"], "label": row["label"], "ok": False, "error": str(exc)})
    return {"results": results, "title": title, "content": content, "tags": tags}


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
                        "presets": sorted(SELECTABLE_PRESETS),
                        "subtitle_actions": sorted(SUBTITLE_ACTIONS),
                        "extra_text_actions": sorted(EXTRA_TEXT_ACTIONS),
                        "devices": ["auto", "cuda", "cpu"],
                    }
                )
                return

            if path == "/api/files":
                self._send_json({"files": list_project_videos()})
                return

            if path == "/api/jobs":
                self._send_json({"jobs": list_jobs(limit=30)})
                return

            if path == "/api/ingest":
                self._send_json(ingest_status())
                return

            if path == "/api/zernio/accounts":
                self._send_json({"accounts": zernio_store().list_public()})
                return

            if path.startswith("/api/jobs/"):
                job_id = path.split("/")[3] if len(path.split("/")) > 3 else ""
                self._send_json(get_job(job_id))
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

            if parsed.path == "/api/ingest":
                payload = self._read_json()
                try:
                    limit = int(payload.get("limit") or 10)
                except (TypeError, ValueError) as exc:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Giới hạn không hợp lệ.") from exc
                self._send_json(start_source_poll(
                    str(payload.get("sources") or ""),
                    limit,
                    bool(payload.get("then_process", True)),
                ), HTTPStatus.ACCEPTED)
                return

            if parsed.path == "/api/zernio/accounts":
                self._send_json(add_zernio_account(self._read_json()), HTTPStatus.CREATED)
                return

            if parsed.path == "/api/zernio/publish":
                self._send_json(publish_zernio(self._read_json()))
                return

            if parsed.path == "/api/publish/suggest":
                self._send_json(suggest_publish_metadata(str(self._read_json().get("video_path") or "")))
                return

            if parsed.path.startswith("/api/zernio/accounts/") and parsed.path.endswith("/check"):
                account_id = parsed.path.split("/")[4]
                self._send_json(check_zernio_account(account_id))
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
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[3] if len(parsed.path.split("/")) > 3 else ""
                if not job_id:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Thiếu job id.")
                self._send_json(cancel_job(job_id), HTTPStatus.ACCEPTED)
                return

            if parsed.path.startswith("/api/zernio/accounts/"):
                account_id = parsed.path.split("/")[4] if len(parsed.path.split("/")) > 4 else ""
                if not account_id or not zernio_store().remove(account_id):
                    raise RequestError(HTTPStatus.NOT_FOUND, "Không thấy API key.")
                self._send_json({"deleted": account_id})
                return

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
        voice_id = str(payload.get("voice_id") or DEFAULT_WEB_VOICE).strip()
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
    parser.add_argument(
        "--keep-jobs",
        type=int,
        default=100,
        help="Keep at most this many finished job workspaces (default: 100, 0 disables cleanup)",
    )
    args = parser.parse_args()

    if not FRONTEND_ROOT.is_dir():
        raise SystemExit(f"Frontend directory not found: {FRONTEND_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # A crashed run leaves state RUNNING forever. Reconcile before serving so
    # the UI can offer a clean rerun instead of showing a stuck "đang xử lý".
    try:
        interrupted = JOB_STORE.reconcile_interrupted()
    except OSError as exc:
        print(f"MediaFlow web: job reconcile skipped ({exc})")
    else:
        if interrupted:
            print(f"MediaFlow web: marked {len(interrupted)} interrupted job(s) as failed")

    # Job workspaces accumulate one folder per run; without this the directory
    # only ever grows. Only terminal, unclaimed jobs are removed.
    if args.keep_jobs > 0:
        try:
            removed = JOB_STORE.cleanup(max_jobs=args.keep_jobs)
        except OSError as exc:
            print(f"MediaFlow web: job cleanup skipped ({exc})")
        else:
            if removed:
                print(f"MediaFlow web: removed {len(removed)} old job workspace(s)")

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
