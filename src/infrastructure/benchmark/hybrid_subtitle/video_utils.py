from __future__ import annotations
import json, logging, os, re, subprocess, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)

YELLOW_BGR    = ( 66, 204, 255)
GREEN_BGR     = ( 89, 178,  85)
RED_BGR       = ( 77,  55, 216)
ORANGE_BGR    = ( 80, 165, 255)
BLUE_BGR      = (255, 172,  27)
BLACK         = (  0,   0,   0)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ffprobe_metadata(path: str) -> Dict[str, Any]:
    try:
        r = subprocess.run([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, check=True)
        payload = json.loads(r.stdout)
    except Exception as exc:
        logger.warning("ffprobe failed %s: %s", path, exc)
        return {}
    streams = payload.get("streams")
    if not streams:
        return {}
    strm = streams[0]
    try:
        num, den = strm.get("r_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    dur = 0.0
    if "format" in payload and "duration" in payload["format"]:
        try: dur = float(payload["format"]["duration"])
        except Exception: pass
    if dur == 0.0:
        try: dur = float(strm.get("duration", 0) or 0)
        except Exception: pass
    try: frames = int(strm.get("nb_frames", "0") or 0)
    except Exception: frames = 0
    return {
        "codec_name": strm.get("codec_name", "unknown"),
        "width":      int(strm.get("width", 0) or 0),
        "height":     int(strm.get("height", 0) or 0),
        "fps":        fps,
        "frame_count": frames,
        "duration":   dur,
    }

@dataclass
class VideoMetadata:
    path: str
    codec_name: str = "unknown"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    sha256: str = ""
    transcoded_path: Optional[str] = None
    transcoded_codec: Optional[str] = None

def transcode_h264(src: str, dst: str) -> str:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if Path(dst).exists():
        return dst
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", dst],
        capture_output=True, text=True, check=True)
    return dst

def probe_video(path: str, cache_dir: Optional[str] = None) -> VideoMetadata:
    meta = VideoMetadata(path=path)
    info = ffprobe_metadata(path)
    meta.codec_name       = info.get("codec_name", "unknown")
    meta.width            = int(info.get("width", 0) or 0)
    meta.height           = int(info.get("height", 0) or 0)
    meta.fps              = float(info.get("fps", 0) or 0)
    meta.frame_count      = int(info.get("frame_count", 0) or 0)
    meta.duration_seconds = float(info.get("duration", 0) or 0)
    meta.sha256           = _sha256(path)
    needs_tc = meta.codec_name.lower() not in {"h264", "hevc", "vp9", "mpeg4", "mjpeg"}
    if needs_tc and cache_dir:
        out = os.path.join(cache_dir, f"{Path(path).stem}_h264.mp4")
        try:
            transcode_h264(path, out)
            meta.transcoded_path  = out
            meta.transcoded_codec = "h264"
        except Exception as exc:
            logger.warning("Transcode failed for %s: %s", path, exc)
    return meta

def draw_bboxes(frame, detections: List[Dict], color_map: Dict[str, Tuple[int,int,int]],
                font: int = cv2.FONT_HERSHEY_SIMPLEX) -> Any:
    out = frame.copy()
    h, w = out.shape[:2]
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if not bbox: continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1,x2 = max(0,min(x1,w-1)), max(0,min(x2,w-1))
        y1,y2 = max(0,min(y1,h-1)), max(0,min(y2,h-1))
        key   = det.get("color_key", "yellow")
        color = color_map.get(key, YELLOW_BGR)
        thick = det.get("thickness", 2)
        cv2.rectangle(out, (x1,y1),(x2,y2), color, thick)
        label = det.get("label") or det.get("track_id") or ""
        if label:
            (tw,th),_ = cv2.getTextSize(label, font, 0.45, 1)
            bg_y1 = max(y1-th-4, 0)
            cv2.rectangle(out, (x1, bg_y1), (x1+tw+2, y1), color, -1)
            cv2.putText(out, label, (x1+1, y1-2), font, 0.45, BLACK, 1, cv2.LINE_AA)
    return out

COLOR_MAP = {
    "yellow": YELLOW_BGR,
    "raw":    YELLOW_BGR,
    "green":  GREEN_BGR,
    "accepted": GREEN_BGR,
    "red":    RED_BGR,
    "rejected": RED_BGR,
    "orange": ORANGE_BGR,
    "uncertain": ORANGE_BGR,
    "blue":   BLUE_BGR,
    "cover":  BLUE_BGR,
}

def redact_text(text: str) -> str:
    if not text: return text
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if api_key and len(api_key) > 5:
        text = text.replace(api_key, "[REDACTED]")
    return re.sub(r"(key|token|secret|password)[=:]\s*[A-Za-z0-9\-_]+", "[REDACTED]", text, flags=re.I)

def safe_json_dump(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(redact_text(json.dumps(obj, ensure_ascii=False, indent=2)), encoding="utf-8")

def jsonl_dump(records: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(redact_text(json.dumps(rec, ensure_ascii=False)) + "\n")

def contact_sheet(frames: List[Any], cols: int = 4, scale: float = 0.5,
                  labels: Optional[List[str]] = None) -> Any:
    if not frames: return None
    ih,iw = frames[0].shape[:2]
    thumb_w, thumb_h = int(iw*scale), int(ih*scale)
    rows = (len(frames) + cols - 1) // cols
    sheet = 255 * np.ones((rows*thumb_h, cols*thumb_w, 3), dtype=np.uint8)
    for i, (fr, lb) in enumerate(zip(frames, labels or [None]*len(frames))):
        if fr is None: continue
        thumb = cv2.resize(fr, (thumb_w, thumb_h))
        r, c = divmod(i, cols)
        y0, x0 = r*thumb_h, c*thumb_w
        sheet[y0:y0+thumb_h, x0:x0+thumb_w] = thumb
        if lb:
            cv2.putText(sheet, lb, (x0+2, y0+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    return sheet

try:
    import numpy as np
except ImportError:
    np = None
