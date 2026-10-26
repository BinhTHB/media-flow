import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("GEMINI_API_KEY", "dummy_testing_api_key")

# Keep durable job state out of the repository: the runner and the web server
# both write here, so a plain pytest run must not litter outputs/jobs.
if not os.environ.get("MEDIA_FLOW_JOBS_DIR"):
    os.environ["MEDIA_FLOW_JOBS_DIR"] = tempfile.mkdtemp(prefix="mediaflow-test-jobs-")


@pytest.fixture(autouse=True)
def _isolate_web_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never let a test create real ``outputs/`` or ``input/uploads/`` dirs.

    Several tests exercise ``create_job``/``build_dubbing_plan``, which call
    ``mkdir`` on these module-level roots. Redirecting them keeps the working
    tree clean even when an individual test forgets to patch.
    """
    import src.interfaces.web.server as web_server

    outputs = tmp_path / "_web_outputs"
    uploads = tmp_path / "_web_uploads"
    monkeypatch.setattr(web_server, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(web_server, "UPLOAD_ROOT", uploads)
    yield
