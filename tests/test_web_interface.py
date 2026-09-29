from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from src.interfaces.web.server import (
    PROJECT_ROOT,
    MediaFlowHandler,
    RequestError,
    build_dubbing_plan,
    build_job_options,
    dubbing_capabilities,
    resolve_project_video,
    safe_output_name,
    safe_upload_name,
)


def test_safe_output_name_removes_directories_and_forces_mp4() -> None:
    assert safe_output_name("../unsafe name.mkv") == "unsafe-name.mp4"
    assert safe_output_name("") == "mediaflow-output.mp4"


def test_safe_upload_name_rejects_unsupported_extension() -> None:
    with pytest.raises(RequestError, match="không hợp lệ"):
        safe_upload_name("payload.exe")


def test_resolve_project_video_rejects_project_escape() -> None:
    with pytest.raises(RequestError, match="thư mục dự án"):
        resolve_project_video("../outside.mp4", must_exist=False)


def test_dubbing_capabilities_report_partial_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    for name in list(__import__("os").environ):
        if name.startswith("OPENROUTER_API_KEY_"):
            monkeypatch.delenv(name, raising=False)
    capabilities = dubbing_capabilities()
    assert capabilities["ready"] is False
    assert capabilities["status"] == "partial"
    assert capabilities["providers"]["asr"]["available"] is False
    assert capabilities["providers"]["translation"]["available"] is False


def test_build_dubbing_plan_single_speaker_can_be_ready(monkeypatch) -> None:
    pytest.importorskip("edge_tts")
    sample = PROJECT_ROOT / "input" / "sample5.mp4"
    if not sample.is_file():
        candidates = sorted((PROJECT_ROOT / "input").glob("*.mp4"))
        if candidates:
            sample = candidates[0]
        else:
            pytest.skip("sample video is not present in this checkout")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("src.interfaces.web.server.shutil.which", lambda name: name)
    plan = build_dubbing_plan(
        {
            "input_path": str(sample),
            "source_language": "zh",
            "target_language": "vi",
            "voice_id": "vi-VN-HoaiMyNeural",
            "speaker_mode": "single",
            "fit_policy": "fit",
            "preserve_ambience": False,
        }
    )
    assert plan["ready"] is True
    assert plan["status"] == "ready"
    assert plan["input_relative"] == sample.relative_to(PROJECT_ROOT).as_posix()
    assert all(stage["status"] == "ready" for stage in plan["stages"])
    assert plan["blockers"] == []


def test_web_job_maps_to_shared_runner_contract() -> None:
    job = {
        "id": "web_quality",
        "input_path": str(PROJECT_ROOT / "input" / "source video.mp4"),
        "output_path": str(PROJECT_ROOT / "outputs" / "final video.mp4"),
        "preset": "dub-audio",
        "profile": "quality",
        "device": "auto",
        "overwrite": True,
        "source_language": "zh",
        "target_language": "vi",
        "subtitle_layout": "bottom",
        "hide_source_subtitle": False,
    }

    options = build_job_options(job)

    assert options.input_path == job["input_path"]
    assert options.output_path == job["output_path"]
    assert options.preset == "dub-audio"
    assert options.profile == "quality"
    assert options.subtitle_layout == "bottom"
    assert options.hide_source_subtitle is False


def test_create_job_requires_input_path() -> None:
    from src.interfaces.web.server import create_job

    with pytest.raises(RequestError, match="Chưa chọn video đầu vào"):
        create_job({"presets": ["blur-all"]})


def test_create_job_aligns_with_cli_pipeline_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    from src.interfaces.web.server import create_job
    import src.interfaces.web.server as web_server

    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_bytes(b"dummy")
    monkeypatch.setattr(web_server, "resolve_project_video", lambda p: dummy_video)
    monkeypatch.setattr(web_server, "dubbing_capabilities", lambda: {"ready": True})
    monkeypatch.setattr(web_server.threading, "Thread", MagicMock())

    payload = {
        "input_path": str(dummy_video),
        "preset": "translate-video",
        "blur_extra": True,
        "subtitle_layout": "temporal",
        "hide_source_subtitle": True,
    }

    job = create_job(payload)
    assert job["preset"] == "translate-video"
    assert job["blur_extra"] is True
    assert job["subtitle_layout"] == "temporal"
    assert job["hide_source_subtitle"] is True
    assert job["segments"] is None  # CLI flow: segments is None!


def test_local_server_serves_health_and_frontend() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MediaFlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with urlopen(f"{base_url}/api/health", timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{base_url}/", timeout=2) as response:
            html = response.read().decode("utf-8")
        with urlopen(f"{base_url}/index.html", timeout=2) as response:
            cleanup_html = response.read().decode("utf-8")
        with urlopen(f"{base_url}/dubbing.html", timeout=2) as response:
            dubbing_html = response.read().decode("utf-8")
        with urlopen(f"{base_url}/studio.html", timeout=2) as response:
            studio_html = response.read().decode("utf-8")
        with urlopen(f"{base_url}/studio.js", timeout=2) as response:
            studio_js = response.read().decode("utf-8")
        with urlopen(f"{base_url}/api/dubbing/capabilities", timeout=2) as response:
            capabilities = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health["status"] == "ok"
    assert "blur-all" in health["presets"]
    assert "translate-video-generic" in health["presets"]
    assert "translate-subtitle-generic" in health["presets"]
    assert "MediaFlow" in html
    assert "Xử lý" in cleanup_html
    # Legacy routes (/index.html, /dubbing.html) intentionally serve the unified studio page.
    assert dubbing_html == studio_html == cleanup_html
    assert "MediaFlow CLI Studio" in studio_html
    # studio.js is the CLI-runner GUI; it probes runtime status via /api/health.
    assert "/api/health" in studio_js
    assert capabilities["status"] in {"partial", "ready"}


def test_local_server_delete_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from urllib.request import Request
    from urllib.error import HTTPError
    import src.interfaces.web.server as web_server

    test_upload_dir = tmp_path / "uploads"
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir = tmp_path / "outputs"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch to avoid wiping real outputs or uploads
    monkeypatch.setattr(web_server, "UPLOAD_ROOT", test_upload_dir)
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", test_output_dir)

    server = ThreadingHTTPServer(("127.0.0.1", 0), MediaFlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    # 1. Delete a specific file in input/uploads (real path in project)
    test_input_dir = PROJECT_ROOT / "input" / "uploads"
    test_input_dir.mkdir(parents=True, exist_ok=True)
    test_input_file = test_input_dir / "test_delete_sample.mp4"
    test_input_file.write_bytes(b"dummy video content")
    protected_file: Path | None = None

    try:
        req = Request(f"{base_url}/api/files?path=input/uploads/test_delete_sample.mp4", method="DELETE")
        with urlopen(req, timeout=2) as response:
            res = json.loads(response.read().decode("utf-8"))
            assert res["success"] is True
            assert res["deleted"] == "input/uploads/test_delete_sample.mp4"
            assert not test_input_file.exists()

        # 2. Prevent deleting protected test_videos (raises 403 Forbidden)
        protected_dir = PROJECT_ROOT / "test_videos"
        protected_dir.mkdir(parents=True, exist_ok=True)
        protected_file = protected_dir / "sample_protected.mp4"
        protected_file.write_bytes(b"sample protected video")

        req_protected = Request(f"{base_url}/api/files?path=test_videos/sample_protected.mp4", method="DELETE")
        with pytest.raises(HTTPError) as excinfo:
            urlopen(req_protected, timeout=2)
        assert excinfo.value.code == 403
        assert protected_file.exists()

        # 3. Clear all files (isolated in tmp_path via monkeypatch)
        dummy_output = test_output_dir / "temp_output_to_delete.mp4"
        dummy_output.write_bytes(b"dummy output")
        req_clear = Request(f"{base_url}/api/files", method="DELETE")
        with urlopen(req_clear, timeout=2) as response:
            res_clear = json.loads(response.read().decode("utf-8"))
            assert res_clear["cleared"] is True
            assert res_clear["deleted_count"] >= 1
            assert not dummy_output.exists()
    finally:
        if test_input_file.exists():
            test_input_file.unlink(missing_ok=True)
        if protected_file and protected_file.exists():
            protected_file.unlink(missing_ok=True)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)