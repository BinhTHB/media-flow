"""Integration tests for the review server's frame serving and the UI wiring.

These tests exercise the real HTTP endpoints (localhost only, ephemeral port)
and the real evaluation dataset. They are skipped when the evaluation manifest
is not present (e.g. a fresh clone), so the unit suite stays green anywhere.
"""

from __future__ import annotations

import csv
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import pytest

import scripts.serve_evaluation_review as srv
from src.infrastructure.detection.evaluation.review_store import resolve_frame_path

MANIFEST = srv.MANIFEST_PATH
FRAMES = srv.FRAMES_DIR

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="evaluation dataset not present"
)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.ReviewHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _manifest_rows():
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _first_frame_id() -> str:
    return _manifest_rows()[0]["frame_id"]


def _get_json(url: str):
    import urllib.request

    with urllib.request.urlopen(url, timeout=15) as r:
        assert r.status == 200
        return json.loads(r.read().decode("utf-8"))


def _get_bytes(url: str, expect_status: int):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            assert r.status == expect_status
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        assert e.code == expect_status
        return e.code, e.headers.get("Content-Type", ""), b""


# ---------------------------------------------------------------------------
# Frame storage
# ---------------------------------------------------------------------------
def test_all_manifest_frames_resolve_and_decode():
    rows = _manifest_rows()
    missing = []
    corrupt = []
    for r in rows:
        p = resolve_frame_path(FRAMES, str(r["frame_id"]))
        if p is None:
            missing.append(r["frame_id"])
            continue
        img = cv2.imread(str(p))
        if img is None:
            corrupt.append(r["frame_id"])
    assert not missing, f"missing frames: {missing[:10]}"
    assert not corrupt, f"corrupt frames: {corrupt[:10]}"


def test_manifest_total_count():
    assert len(_manifest_rows()) == 200


# ---------------------------------------------------------------------------
# HTTP frame endpoint
# ---------------------------------------------------------------------------
def test_frame_endpoint_returns_200(server):
    status, _ctype, body = _get_bytes(
        f"{server}/api/frame/{_first_frame_id()}/image", 200
    )
    assert status == 200
    assert len(body) > 0


def test_frame_endpoint_correct_mimetype(server):
    _, ctype, _ = _get_bytes(f"{server}/api/frame/{_first_frame_id()}/image", 200)
    assert ctype == "image/jpeg"


def test_frame_image_decodes(server):
    _, _, body = _get_bytes(f"{server}/api/frame/{_first_frame_id()}/image", 200)
    import numpy as np

    img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    row = _manifest_rows()[0]
    assert img.shape[0] == int(row["height"])
    assert img.shape[1] == int(row["width"])


def test_missing_frame_returns_404(server):
    status, _, _ = _get_bytes(f"{server}/api/frame/qa_zzz_not_there/image", 404)
    assert status == 404


def test_path_traversal_rejected(server):
    # Any of 400/403/404 is a correct rejection; what matters is no 200.
    status, _, _ = _get_bytes(f"{server}/api/frame/..%2F..%2FREADME/image", 400)
    assert status in (400, 403, 404)


def test_debug_endpoint_reports_size_and_decode(server):
    info = _get_json(f"{server}/api/debug/frame/{_first_frame_id()}")
    assert info["exists"] is True
    assert info["size_bytes"] > 0
    assert info["decode_ok"] is True
    assert info["mime"] == "image/jpeg"


# ---------------------------------------------------------------------------
# UI wiring (static checks against the served HTML)
# ---------------------------------------------------------------------------
def test_ui_serves_html(server):
    import urllib.request

    with urllib.request.urlopen(f"{server}/", timeout=15) as r:
        html = r.read().decode("utf-8")
    assert 'id="frameImg"' in html  # <img> baseline always renders the frame
    assert 'id="cv"' in html
    assert 'id="frameError"' in html


def test_ui_frame_url_generated_from_frame_id(server):
    import urllib.request

    with urllib.request.urlopen(f"{server}/", timeout=15) as r:
        html = r.read().decode("utf-8")
    # The image element must use an HTTP URL built from frame_id, never a
    # Windows filesystem path.
    assert '"/api/frame/" + encodeURIComponent(d.frame_id) + "/image"' in html
    assert "E:\\projects\\" not in html
