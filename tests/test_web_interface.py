from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
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
    # The legacy dub-audio preset resolves onto the flow + feature fields.
    assert options.preset == "yolo-4class"
    assert options.subtitle_action == "preserve"
    assert options.dub_audio is True
    assert options.extra_text_action == "preserve"
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
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_server.threading, "Thread", MagicMock())

    payload = {
        "input_path": str(dummy_video),
        "preset": "translate-video",
        "blur_extra": True,
        "subtitle_layout": "temporal",
        "hide_source_subtitle": True,
    }

    job = create_job(payload)
    # The legacy feature preset resolves onto flow + feature fields.
    assert job["preset"] == "yolo-4class"
    assert job["subtitle_action"] == "translate"
    assert job["dub_audio"] is True
    assert job["extra_text_action"] == "blur"
    assert job["blur_extra"] is True
    assert job["subtitle_layout"] == "temporal"
    assert job["hide_source_subtitle"] is True
    assert job["segments"] is None  # CLI flow: segments is None!


def test_create_job_resolves_legacy_generic_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    from src.interfaces.web.server import create_job
    import src.interfaces.web.server as web_server

    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_bytes(b"dummy")
    monkeypatch.setattr(web_server, "resolve_project_video", lambda p: dummy_video)
    monkeypatch.setattr(web_server, "dubbing_capabilities", lambda: {"ready": True})
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_server.threading, "Thread", MagicMock())

    job = create_job({
        "input_path": str(dummy_video),
        "preset": "translate-subtitle-generic",
        "subtitle_layout": "temporal",
    })
    assert job["preset"] == "yolo-text"
    assert job["subtitle_action"] == "translate"
    assert job["dub_audio"] is False
    assert job["extra_text_action"] == "preserve"


def test_create_job_accepts_flow_preset_with_feature_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    from src.interfaces.web.server import create_job
    import src.interfaces.web.server as web_server

    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_bytes(b"dummy")
    monkeypatch.setattr(web_server, "resolve_project_video", lambda p: dummy_video)
    monkeypatch.setattr(web_server, "dubbing_capabilities", lambda: {"ready": True})
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_server.threading, "Thread", MagicMock())

    job = create_job({
        "input_path": str(dummy_video),
        "preset": "yolo-text",
        "subtitle_action": "blur",
        "dub_audio": False,
        "extra_text_action": "blur",
        "subtitle_layout": "temporal",
    })
    assert job["preset"] == "yolo-text"
    assert job["subtitle_action"] == "blur"
    assert job["dub_audio"] is False
    assert job["extra_text_action"] == "blur"


def test_create_job_reup_keeps_the_ticked_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    from src.interfaces.web.server import create_job
    import src.interfaces.web.server as web_server

    dummy_video = tmp_path / "test.mp4"
    dummy_video.write_bytes(b"dummy")
    monkeypatch.setattr(web_server, "resolve_project_video", lambda p: dummy_video)
    monkeypatch.setattr(web_server, "dubbing_capabilities", lambda: {"ready": True})
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_server.threading, "Thread", MagicMock())

    visual = create_job({
        "input_path": str(dummy_video),
        "preset": "reup",
        "subtitle_action": "translate",
        "dub_audio": False,
        "extra_text_action": "blur",
        "target_language": "vi",
        "variation_level": "off",
    })
    assert visual["preset"] == "yolo-4class"
    assert visual["reup_contract"] is True
    assert visual["subtitle_action"] == "translate"
    assert visual["dub_audio"] is False
    assert visual["extra_text_action"] == "blur"
    assert visual["preserve_ambience"] is True
    options = build_job_options(visual)
    assert options.reup_contract is True
    assert options.dub_audio is False
    assert options.variation_level == "off"

    audio = create_job({
        "input_path": str(dummy_video),
        "output_name": "audio-only.mp4",
        "preset": "reup",
        "subtitle_action": "preserve",
        "dub_audio": True,
        "extra_text_action": "preserve",
        "source_language": "ja",
        "target_language": "vi",
        "voice_id": "vi-VN-HoaiMyNeural",
    })
    assert audio["reup_contract"] is True
    assert audio["subtitle_action"] == "preserve"
    assert audio["dub_audio"] is True
    assert audio["preserve_ambience"] is False
    assert audio["mix_original_volume"] == 0.0
    assert audio["max_tts_speedup"] == 1.8

    with pytest.raises(RequestError, match="hình, tiếng"):
        create_job({
            "input_path": str(dummy_video),
            "output_name": "empty.mp4",
            "preset": "reup",
            "subtitle_action": "preserve",
            "dub_audio": False,
            "extra_text_action": "preserve",
        })


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
    assert set(health["presets"]) == {"yolo-4class", "yolo-text", "reup"}
    assert set(health["subtitle_actions"]) == {"translate", "blur", "preserve"}
    assert set(health["extra_text_actions"]) == {"blur", "preserve"}
    assert "MediaFlow" in html
    assert "Xử lý" in cleanup_html
    # Legacy routes (/index.html, /dubbing.html) intentionally serve the unified studio page.
    assert dubbing_html == studio_html == cleanup_html
    assert "MediaFlow CLI Studio" in studio_html
    assert 'data-preset="reup"' in studio_html
    assert 'id="reupVisualCheck"' in studio_html
    assert 'id="reupAudioCheck" checked' in studio_html
    assert 'id="sourceSection"' in studio_html
    assert 'id="publishSection"' in studio_html
    assert 'id="variationSelect"' in studio_html
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
        # Do not leave empty scratch dirs behind in the real working tree.
        for scratch in (
            PROJECT_ROOT / "input" / "uploads",
            PROJECT_ROOT / "test_videos",
        ):
            try:
                scratch.rmdir()
            except OSError:
                pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

def _make_registered_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    from unittest.mock import MagicMock
    import src.interfaces.web.server as web_server

    dummy_video = tmp_path / "log-handler-video.mp4"
    dummy_video.write_bytes(b"dummy")
    monkeypatch.setattr(web_server, "resolve_project_video", lambda p: dummy_video)
    monkeypatch.setattr(web_server, "dubbing_capabilities", lambda: {"ready": True})
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_server.threading, "Thread", MagicMock())
    job = web_server.create_job({"input_path": str(dummy_video), "overwrite": True})
    return job


def test_job_log_handler_streams_pipeline_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import logging
    import src.interfaces.web.server as web_server

    job = _make_registered_job(monkeypatch, tmp_path)
    job_id = job["id"]
    try:
        handler = web_server._attach_job_log_handler(job_id)
        try:
            pipeline_logger = logging.getLogger("src.features.visual.pipeline")
            pipeline_logger.info("Visual: rendering 500/1000 frames (50%)", extra={"mediaflow_pct": 50, "mediaflow_stage": "visual"})
            pipeline_logger.warning("Visual: provider slow, retrying")
        finally:
            web_server._detach_job_log_handler(handler)

        logs = web_server._JOBS[job_id]["logs"]
        assert any("[PIPELINE] Visual: rendering 500/1000 frames (50%)" in line for line in logs)
        assert any("provider slow, retrying" in line for line in logs)
        # Pipeline-internal 50% maps into the runner's 40-90 window.
        assert web_server._JOBS[job_id]["progress"] == 65
        assert web_server._JOBS[job_id]["message"] == "Visual: rendering 500/1000 frames (50%)"
    finally:
        web_server._JOBS.pop(job_id, None)


def test_job_log_handler_progress_is_monotonic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import logging
    import src.interfaces.web.server as web_server

    job = _make_registered_job(monkeypatch, tmp_path)
    job_id = job["id"]
    try:
        handler = web_server._attach_job_log_handler(job_id)
        try:
            visual_logger = logging.getLogger("src.features.visual.pipeline")
            audio_logger = logging.getLogger("src.features.audio.pipeline")
            visual_logger.info("Visual: done", extra={"mediaflow_pct": 100, "mediaflow_stage": "visual"})
            visual_progress = web_server._JOBS[job_id]["progress"]
            # Audio starts its own 0-100 scale; job progress must never regress.
            audio_logger.info("Audio: starting", extra={"mediaflow_pct": 0, "mediaflow_stage": "audio"})
            audio_logger.info("Audio: TTS 5/10 segments (50%)", extra={"mediaflow_pct": 50, "mediaflow_stage": "audio"})
            final_progress = web_server._JOBS[job_id]["progress"]
        finally:
            web_server._detach_job_log_handler(handler)

        assert visual_progress == 90
        # Audio reports its own 0-100 scale; the monotonic guard keeps the
        # job progress from regressing below the visual pipeline's peak.
        assert final_progress == 90
        assert web_server._JOBS[job_id]["message"] == "Visual: done"
    finally:
        web_server._JOBS.pop(job_id, None)


def test_job_log_handler_ignores_unknown_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import logging
    import src.interfaces.web.server as web_server

    handler = web_server._JobLogHandler("web_missing_job")
    record = logging.LogRecord(
        "src.features.visual.pipeline", logging.INFO, __file__, 1,
        "Visual: harmless", None, None,
    )
    handler.emit(record)  # Must not raise.


# --------------------------------------------------------------------------
# Durable job history, cancellation, cleanup
# --------------------------------------------------------------------------

def _isolated_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the module-level store at a throwaway directory."""
    import src.interfaces.web.server as web_server
    from src.interfaces.jobs.store import JobStore

    store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(web_server, "JOB_STORE", store)
    return web_server, store


def test_list_jobs_survives_process_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A job written by JobRunner is listed even with no in-memory state.

    This is the regression the web UI used to have: jobs lived only in the
    ``_JOBS`` dict, so every restart erased the history.
    """
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"video-bytes")
    store.save_state("job_persisted", {
        "id": "job_persisted",
        "status": "SUCCESS",
        "progress": 100,
        "stage": "completed",
        "created_at": 100.0,
        "updated_at": 200.0,
        "request": {"input_video": str(tmp_path / "in.mp4"), "output_video": str(output)},
        "metrics": {"duration_seconds": 3.5, "detections": {"subtitle": 4}},
    })
    monkeypatch.setattr(web_server, "_JOBS", {})

    jobs = web_server.list_jobs()

    assert [job["id"] for job in jobs] == ["job_persisted"]
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["output_size"] == len(b"video-bytes")
    assert jobs[0]["detections"] == {"subtitle": 4}
    assert jobs[0]["persisted"] is True


def test_list_jobs_prefers_live_state_over_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Live progress and logs win over the coarser persisted snapshot."""
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    store.save_state("job_live", {"id": "job_live", "status": "RUNNING", "progress": 40, "created_at": 1.0})
    monkeypatch.setattr(web_server, "_JOBS", {
        "job_live": {"id": "job_live", "status": "running", "progress": 65, "logs": ["live line"], "created_at": 1.0},
    })

    jobs = web_server.list_jobs()

    assert len(jobs) == 1
    assert jobs[0]["progress"] == 65
    assert jobs[0]["logs"] == ["live line"]


def test_list_jobs_keeps_the_live_job_when_history_fills_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A running studio job stays listed after thirty older jobs are on disk.

    Web jobs stamp ``created_at`` as an ISO string. Those used to sort as zero
    and fall off the thirty-item page, so the progress dock attached to a stale
    store row instead of the job actually rendering.
    """
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    for index in range(30):
        store.save_state(
            f"old_{index}",
            {
                "id": f"old_{index}",
                "status": "SUCCESS",
                "created_at": float(index + 1),
                "updated_at": float(index + 1),
            },
        )
    monkeypatch.setattr(web_server, "_JOBS", {
        "web_live": {
            "id": "web_live",
            "status": "running",
            "progress": 72,
            "message": "Visual: rendering 1800/2491 frames (72%)",
            "created_at": web_server.utc_now(),
            "logs": ["[18:07:25] [PIPELINE] Visual: rendering 1800/2491 frames (72%)"],
        },
    })

    jobs = web_server.list_jobs()

    assert jobs[0]["id"] == "web_live"
    assert jobs[0]["logs"]


def test_failed_job_keeps_its_error_after_the_process_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refresh must still show why the run died.

    The store leaves ``stage`` on the last checkpoint (``processing``). The
    failure sentence is on ``error``. Projecting the stage made the dock say
    the job was still processing.
    """
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    failure = (
        "All translation providers failed. gemini: "
        "provider returned an empty translation batch"
    )
    store.save_state("web_failed", {
        "id": "web_failed",
        "status": "FAILED",
        "progress": 100,
        "stage": "processing",
        "created_at": 10.0,
        "updated_at": 20.0,
        "error": {"code": "internal_error", "message": failure},
        "events": [
            {"type": "progress", "stage": "processing", "progress": 40, "message": "Processing media"},
            {"type": "failed", "stage": "failed", "progress": 100, "message": failure},
        ],
        "request": {},
    })
    monkeypatch.setattr(web_server, "_JOBS", {})

    job = next(item for item in web_server.list_jobs() if item["id"] == "web_failed")

    assert job["status"] == "failed"
    assert job["message"] == failure
    assert job["logs"][-1] == failure


def test_list_jobs_orders_newest_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    store.save_state("old", {"id": "old", "status": "SUCCESS", "created_at": 1.0, "updated_at": 1.0})
    store.save_state("new", {"id": "new", "status": "SUCCESS", "created_at": 2.0, "updated_at": 2.0})
    monkeypatch.setattr(web_server, "_JOBS", {})

    assert [job["id"] for job in web_server.list_jobs()] == ["new", "old"]


def test_get_job_falls_back_to_store_and_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    store.save_state("job_x", {"id": "job_x", "status": "PARTIAL", "created_at": 1.0})
    monkeypatch.setattr(web_server, "_JOBS", {})

    assert web_server.get_job("job_x")["status"] == "succeeded"  # PARTIAL is still a usable output
    with pytest.raises(web_server.RequestError):
        web_server.get_job("nope")


def test_cancel_job_writes_marker_for_running_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    store.save_state("job_run", {"id": "job_run", "status": "RUNNING", "created_at": 1.0})
    monkeypatch.setattr(web_server, "_JOBS", {
        "job_run": {"id": "job_run", "status": "running", "progress": 40, "created_at": 1.0},
    })

    response = web_server.cancel_job("job_run")

    assert response["cancelled"] is True
    assert store.cancel_requested("job_run") is True
    assert web_server._JOBS["job_run"]["cancel_requested"] is True


def test_cancel_job_is_noop_for_finished_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "_JOBS", {
        "job_done": {"id": "job_done", "status": "succeeded", "progress": 100, "created_at": 1.0},
    })

    response = web_server.cancel_job("job_done")

    assert response["cancelled"] is False
    assert store.cancel_requested("job_done") is False


def test_cancel_job_rejects_traversal_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    web_server, _store = _isolated_store(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "_JOBS", {})

    with pytest.raises(web_server.RequestError):
        web_server.cancel_job("../../etc/passwd")


def test_delete_job_endpoint_requests_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DELETE /api/jobs/<id> is the cancel route the UI calls."""
    web_server, store = _isolated_store(monkeypatch, tmp_path)
    store.save_state("job_http", {"id": "job_http", "status": "RUNNING", "created_at": 1.0})
    monkeypatch.setattr(web_server, "_JOBS", {
        "job_http": {"id": "job_http", "status": "running", "progress": 10, "created_at": 1.0},
    })

    server = ThreadingHTTPServer(("127.0.0.1", 0), MediaFlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        from urllib.request import Request

        request = Request(f"{base_url}/api/jobs/job_http", method="DELETE")
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["cancelled"] is True
    assert store.cancel_requested("job_http") is True


def test_jobs_dir_env_override_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.interfaces.jobs.runner import default_jobs_dir

    monkeypatch.setenv("MEDIA_FLOW_JOBS_DIR", str(tmp_path / "custom"))
    assert default_jobs_dir() == tmp_path / "custom"

    monkeypatch.delenv("MEDIA_FLOW_JOBS_DIR", raising=False)
    assert default_jobs_dir().name == "jobs"
