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
from src.interfaces.jobs.store import JobStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
UPLOAD_ROOT = PROJECT_ROOT / "input" / "uploads"
E2E_SCRIPT = PROJECT_ROOT / "scripts" / "run_e2e.py"
FINAL_DUBBING_SCRIPT = PROJECT_ROOT / "scripts" / "run_final_dubbing.py"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PRESETS = {"blur-all", "translate-subtitle", "preserve-all", "dub-audio", "translate-video"}
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
_JOB_STORE = JobStore(PROJECT_ROOT / "outputs" / "jobs")


def load_project_env() -> None:
    """Load project-local .env without overwriting explicit process variables."""
    env_path = PROJECT_ROOT / ".env"
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
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


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

    capabilities = dubbing_capabilities()
    required_provider_ids = ("asr", "translation", "tts", "composer")
    blockers = [
        capabilities["providers"][provider_id]["reason"]
        for provider_id in required_provider_ids
        if not capabilities["providers"][provider_id]["available"]
    ]
    if speaker_mode == "auto" and not capabilities["providers"]["diarization"]["available"]:
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
    return {
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "message": "Pipeline dubbing single-speaker đã sẵn sàng." if ready else "Cấu hình dubbing còn thiếu provider bắt buộc.",
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
    input_path = resolve_project_video(str(payload.get("input_path", "")))
    preset = str(payload.get("preset", "blur-all"))
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

    if preset in {"dub-audio", "translate-video"}:
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
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
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
                self.wfile.write(chunk)
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

            # AI Studio is the default landing surface; the original cleanup UI remains at /index.html.
            static_path = "studio.html" if path == "/" else path.lstrip("/")
            candidate = (FRONTEND_ROOT / static_path).resolve()
            if not is_within(candidate, FRONTEND_ROOT.resolve()):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(candidate)
        except RequestError as exc:
            self._send_error_json(exc.status, exc.message)
        except (BrokenPipeError, ConnectionResetError):
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

    def _handle_voice_preview(self) -> None:
        payload = self._read_json()
        voice_id = str(payload.get("voice_id") or "elevenlabs:EXAVITQu4vr4xnSDxMaL").strip()
        if voice_id not in DUBBING_VOICES:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Voice preview không hợp lệ.")
        text = str(payload.get("text") or "Cảm ơn anh. Không có gì. Xin lỗi, tôi giao nhầm đơn rồi.").strip()
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
    load_project_env()
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
