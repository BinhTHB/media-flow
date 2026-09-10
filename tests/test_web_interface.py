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
    assert plan["input_relative"] == "input/sample5.mp4"
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
    }

    options = build_job_options(job)

    assert options.input_path == job["input_path"]
    assert options.output_path == job["output_path"]
    assert options.preset == "dub-audio"
    assert options.profile == "quality"

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
    assert "LOCAL MEDIA WORKSPACE" in html
    assert "Xử lý video" in cleanup_html
    assert "Dubbing Studio" in dubbing_html
    assert "MediaFlow" in studio_html
    assert "Video Dubbing" in studio_html
    assert "renderCapabilities" in studio_js
    assert capabilities["status"] in {"partial", "ready"}
