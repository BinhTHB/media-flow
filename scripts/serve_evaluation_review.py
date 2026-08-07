"""Local human review server for YOLO ground-truth labels.

Starts a small HTTP server bound to ``127.0.0.1`` (localhost only, no auth, no
uploads) that lets a human review and correct the evaluation labels:

* browse frames with filters / sort / hard-case priority;
* draw, move, resize and delete bounding boxes on a canvas;
* switch classes (subtitle / watermark / logo / other_burn_in_text);
* mark frames ``verified`` / ``needs_review`` / ``unverified`` and set
  ``evaluation_role`` (dev / held_out / excluded / unknown) and leakage flag;
* autosave every edit atomically to ``evaluation/labels/<frame_id>.json`` and
  ``evaluation/manifests/manifest.csv``;
* view nearby context frames extracted from the source video (read-only);
* create one dated backup of manifest + labels before the first edit of a day.

Machine pre-labels are shown for speed but are **never** treated as ground
truth: nothing is marked verified unless the human presses "Verify".

Usage:
    .venv\\Scripts\\python.exe scripts\\serve_evaluation_review.py [--port 8765]
Then open http://127.0.0.1:8765 in a browser.

Endpoints (all JSON except static files):
    GET  /                              review UI (HTML)
    GET  /api/summary                   progress dashboard
    GET  /api/frames                    frame list (with filters/sort)
    GET  /api/frame/<id>                full label for one frame
    POST /api/frame/<id>                save label (objects + state)
    GET  /api/frame/<id>/image          frame JPEG
    GET  /api/frame/<id>/context        list of context offsets
    GET  /api/frame/<id>/context/<off>  context frame JPEG
    GET  /api/validate                  run label validator
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.detection.evaluation.review_store import (
    MANIFEST_COLUMNS,
    build_summary,
    create_backup_once,
    get_row,
    load_label,
    load_manifest,
    migrate_manifest,
    save_label,
    update_manifest_row,
    validate_objects,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
FRAMES_DIR = EVAL_DIR / "frames"
LABELS_DIR = EVAL_DIR / "labels"
MANIFEST_PATH = EVAL_DIR / "manifests" / "manifest.csv"
UI_PATH = PROJECT_ROOT / "scripts" / "evaluation_review_ui.html"

# Candidate source-video locations used for context frames.
SOURCE_DIRS = [
    PROJECT_ROOT / "qa_inputs",
    PROJECT_ROOT / "input",
    PROJECT_ROOT / "test_videos",
]

CONTEXT_OFFSETS = [-30, -15, -5, 5, 15, 30]  # frame offsets around current

_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_manifest_locked() -> List[Dict[str, Any]]:
    with _lock:
        return migrate_manifest(MANIFEST_PATH)


def _find_source_video(video_id: str) -> Optional[Path]:
    for d in SOURCE_DIRS:
        for ext in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
            cand = d / f"{video_id}{ext}"
            if cand.is_file():
                return cand
    return None


def _probe_video(path: Path) -> Optional[Dict[str, Any]]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout or "{}")
        s = data["streams"][0]
        num, den = s.get("avg_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
        return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps}
    except Exception:
        return None


def _extract_frame(path: Path, frame_index: int) -> Optional[bytes]:
    """Return JPEG bytes for one frame of the source video (read-only)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or (total and frame_index >= total):
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None
    finally:
        cap.release()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _frame_detail(row: Dict[str, Any]) -> Dict[str, Any]:
    frame_id = str(row["frame_id"])
    label = load_label(LABELS_DIR, frame_id)
    img_path = FRAMES_DIR / f"{frame_id}.jpg"
    has_image = img_path.is_file()
    return {
        "frame_id": frame_id,
        "video_id": str(row.get("video_id", "")),
        "frame_index": int(row.get("frame_index", 0)),
        "timestamp_seconds": float(row.get("timestamp_seconds", 0.0)),
        "width": int(row.get("width", 0)),
        "height": int(row.get("height", 0)),
        "verified": _to_bool(row.get("verified", False)),
        "needs_review": _to_bool(row.get("needs_review", False)),
        "needs_second_review": _to_bool(row.get("needs_second_review", False)),
        "evaluation_role": str(row.get("evaluation_role", "unknown")),
        "leakage_risk": _to_bool(row.get("leakage_risk", False)),
        "label_source": str(row.get("label_source", "prelabel_unverified")),
        "has_image": has_image,
        "objects": label.get("objects", []),
    }


def _save_frame(frame_id: str, payload: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validate + persist one frame's label and manifest state.

    Returns ``(ok, error, detail)``. A 200 on success, 400 with ``error`` on
    invalid payload.
    """
    row = get_row(MANIFEST_PATH, frame_id)
    if row is None:
        return False, f"unknown frame_id {frame_id}", {}
    width = int(row.get("width", 0))
    height = int(row.get("height", 0))

    if "objects" not in payload:
        return False, "payload must contain 'objects'", {}

    objects = payload["objects"]
    if not isinstance(objects, list):
        return False, "'objects' must be a list", {}

    errors = validate_objects(objects, width, height, frame_id)
    if errors:
        return False, "; ".join(errors[:10]), {}

    verified = bool(payload.get("verified", _to_bool(row.get("verified", False))))
    needs_review = bool(payload.get("needs_review", _to_bool(row.get("needs_review", False))))
    needs_second_review = bool(payload.get("needs_second_review", _to_bool(row.get("needs_second_review", False))))
    role = str(payload.get("evaluation_role", row.get("evaluation_role", "unknown")))
    if role not in {"unknown", "dev", "held_out", "excluded"}:
        return False, f"invalid evaluation_role {role!r}", {}
    leakage = bool(payload.get("leakage_risk", _to_bool(row.get("leakage_risk", False))))

    label = load_label(LABELS_DIR, frame_id)
    label["verified"] = verified
    label["needs_review"] = needs_review
    label["needs_second_review"] = needs_second_review
    label["evaluation_role"] = role
    label["leakage_risk"] = leakage
    label["objects"] = objects
    label["reviewed_at"] = payload.get("reviewed_at")
    label["label_source"] = "human_reviewed" if verified else label.get("label_source", "prelabel_unverified")

    with _lock:
        create_backup_once(EVAL_DIR)
        save_label(LABELS_DIR, frame_id, label)
        row = update_manifest_row(
            MANIFEST_PATH,
            frame_id,
            {
                "verified": "true" if verified else "false",
                "needs_review": "true" if needs_review else "false",
                "needs_second_review": "true" if needs_second_review else "false",
                "evaluation_role": role,
                "leakage_risk": "true" if leakage else "false",
                "label_source": "human_reviewed" if verified else label.get("label_source", "prelabel_unverified"),
            },
        )
    return True, None, _frame_detail(row)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, content: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_static(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            self._send_bytes(404, b"not found", "text/plain")
            return
        data = path.read_bytes()
        self._send_bytes(200, data, ctype)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            self._serve_static(UI_PATH, "text/html; charset=utf-8")
            return

        if path.startswith("/api/summary"):
            with _lock:
                manifest = _load_manifest_locked()
            self._send_json(200, build_summary(manifest, LABELS_DIR, FRAMES_DIR))
            return

        if path.startswith("/api/frames"):
            with _lock:
                manifest = _load_manifest_locked()
            rows = [r for r in manifest]
            rows = self._apply_filters(rows, query)
            details = [_frame_detail(r) for r in rows]
            self._send_json(200, details)
            return

        m = _match("/api/frame/<id>/context/<off>", path)
        if m:
            frame_id, off = m
            self._handle_context_frame(frame_id, off)
            return

        m = _match("/api/frame/<id>/context", path)
        if m:
            frame_id = m[0]
            self._handle_context_list(frame_id)
            return

        m = _match("/api/frame/<id>/image", path)
        if m:
            self._handle_image(m[0])
            return

        m = _match("/api/frame/<id>", path)
        if m:
            row = get_row(MANIFEST_PATH, m[0])
            if row is None:
                self._send_json(404, {"error": "unknown frame"})
                return
            self._send_json(200, _frame_detail(row))
            return

        if path == "/api/validate":
            import scripts.validate_evaluation_labels as validator  # type: ignore

            old_argv = sys.argv
            sys.argv = ["validate_evaluation_labels.py",
                        "--manifest", str(MANIFEST_PATH),
                        "--labels", str(LABELS_DIR),
                        "--frames", str(FRAMES_DIR)]
            try:
                code = validator.main()
            finally:
                sys.argv = old_argv
            self._send_json(200, {"exit_code": code})
            return

        self._send_bytes(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        m = _match("/api/frame/<id>", parsed.path)
        if not m:
            self._send_json(404, {"error": "not found"})
            return
        frame_id = m[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return
        ok, err, detail = _save_frame(frame_id, payload)
        if not ok:
            self._send_json(400, {"error": err})
            return
        self._send_json(200, detail)

    # -- helpers ---------------------------------------------------------
    def _apply_filters(self, rows: List[Dict[str, Any]], query: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        state = (query.get("state") or [""])[0]
        role = (query.get("role") or [""])[0]
        has = (query.get("has") or [""])[0]
        sort = (query.get("sort") or ["original"])[0]

        def keep(r: Dict[str, Any]) -> bool:
            if state == "verified" and not _to_bool(r.get("verified")):
                return False
            if state == "unverified" and _to_bool(r.get("verified")):
                return False
            if state == "needs_review" and not _to_bool(r.get("needs_review")):
                return False
            if state == "needs_second_review" and not _to_bool(r.get("needs_second_review")):
                return False
            if role and str(r.get("evaluation_role", "")) != role:
                return False
            if has:
                label = load_label(LABELS_DIR, str(r["frame_id"]))
                classes = {o.get("class_name") for o in label.get("objects", [])}
                if has == "empty" and classes:
                    return False
                if has != "empty" and has not in classes:
                    return False
            return True

        rows = [r for r in rows if keep(r)]
        if sort == "lowest_conf":
            def min_conf(r: Dict[str, Any]) -> float:
                label = load_label(LABELS_DIR, str(r["frame_id"]))
                confs = [float(o.get("confidence", 0.0)) for o in label.get("objects", [])]
                return min(confs) if confs else 1.0
            rows = sorted(rows, key=min_conf)
        elif sort == "most_boxes":
            def n_boxes(r: Dict[str, Any]) -> int:
                label = load_label(LABELS_DIR, str(r["frame_id"]))
                return len(label.get("objects", []))
            rows = sorted(rows, key=n_boxes, reverse=True)
        elif sort == "ambiguous":
            def amb_score(r: Dict[str, Any]) -> Tuple[int, int]:
                label = load_label(LABELS_DIR, str(r["frame_id"]))
                classes = [o.get("class_name") for o in label.get("objects", [])]
                score = int("watermark" in classes) + int("logo" in classes)
                score += int("subtitle" in classes and "other_burn_in_text" in classes)
                return (-score, len(classes))
            rows = sorted(rows, key=amb_score)
        return rows

    def _handle_image(self, frame_id: str) -> None:
        for ext in (".jpg", ".png", ".jpeg"):
            p = FRAMES_DIR / f"{frame_id}{ext}"
            if p.is_file():
                self._serve_static(p, "image/jpeg" if ext != ".png" else "image/png")
                return
        self._send_bytes(404, b"no image", "text/plain")

    def _handle_context_list(self, frame_id: str) -> None:
        row = get_row(MANIFEST_PATH, frame_id)
        if row is None:
            self._send_json(404, {"error": "unknown frame"})
            return
        video_path = _find_source_video(str(row.get("video_id", "")))
        if video_path is None:
            self._send_json(200, {"available": False, "offsets": []})
            return
        info = _probe_video(video_path)
        fps = info["fps"] if info else 30.0
        current = int(row.get("frame_index", 0))
        self._send_json(200, {
            "available": True,
            "fps": fps,
            "current": current,
            "offsets": CONTEXT_OFFSETS,
        })

    def _handle_context_frame(self, frame_id: str, off_raw: str) -> None:
        try:
            off = int(off_raw)
        except ValueError:
            self._send_json(400, {"error": "bad offset"})
            return
        row = get_row(MANIFEST_PATH, frame_id)
        if row is None:
            self._send_json(404, {"error": "unknown frame"})
            return
        video_path = _find_source_video(str(row.get("video_id", "")))
        if video_path is None:
            self._send_bytes(404, b"no source video", "text/plain")
            return
        target = int(row.get("frame_index", 0)) + off
        data = _extract_frame(video_path, target)
        if data is None:
            self._send_bytes(404, b"no frame", "text/plain")
            return
        self._send_bytes(200, data, "image/jpeg")


def _match(pattern: str, path: str) -> Optional[List[str]]:
    """Very small route matcher for ``/api/frame/<id>/...`` style patterns."""
    parts_p = pattern.strip("/").split("/")
    parts = path.strip("/").split("/")
    if len(parts_p) != len(parts):
        return None
    out: List[str] = []
    for pp, p in zip(parts_p, parts):
        if pp.startswith("<") and pp.endswith(">"):
            out.append(urllib.parse.unquote(p))
        elif pp != p:
            return None
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Local evaluation review server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not MANIFEST_PATH.is_file():
        print(f"ERROR: manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    with _lock:
        migrate_manifest(MANIFEST_PATH)
    summary = build_summary(load_manifest(MANIFEST_PATH), LABELS_DIR, FRAMES_DIR)
    print(f"Evaluation review server")
    print(f"  Frames : {summary['total']} (verified {summary['verified']}, "
          f"needs_review {summary['needs_review']})")
    print(f"  UI     : http://{args.host}:{args.port}/")
    print("  Ctrl+C to stop.")

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
