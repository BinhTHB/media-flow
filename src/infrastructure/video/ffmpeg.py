"""
FFmpeg & FFprobe Infrastructure Utilities & Adapter
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from src.core.contracts import VideoUtilPort


def extract_audio(video_path: str, output_audio_path: str) -> str:
    out_p = Path(output_audio_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_p),
    ]
    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return str(out_p)


def is_web_compatible_video(video_path: str | Path) -> bool:
    """Check if the video stream uses a codec and pixel format supported by HTML5 video."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "json",
            str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        if not streams:
            return False
        codec = streams[0].get("codec_name", "").lower()
        pix_fmt = streams[0].get("pix_fmt", "").lower()
        return codec in {"h264", "avc1"} and pix_fmt in {"yuv420p", "yuvj420p"}
    except Exception:
        return False


def ensure_web_compatible_video(video_path: str | Path) -> str:
    """Ensure the video file is web-playable in HTML5 browsers (H.264 / yuv420p).

    If the video is already web-safe or transcoding fails, leaves it untouched.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        return str(path)
    if is_web_compatible_video(path):
        return str(path)

    tmp_out = path.with_name(f"{path.stem}_webcompat.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        if tmp_out.is_file() and tmp_out.stat().st_size > 0:
            tmp_out.replace(path)
    except Exception:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
    return str(path)


def mux_audio_video(video_no_audio_path: str, audio_source_path: str, output_path: str) -> str:
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_source_path)

    force_aac = audio_path.suffix.lower() in {".wav", ".pcm", ".flac", ".ogg"}
    if force_aac:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_no_audio_path),
            "-i", str(audio_source_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(out_p),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_no_audio_path),
            "-i", str(audio_source_path),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            str(out_p),
        ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    except subprocess.CalledProcessError:
        cmd_fallback = [
            "ffmpeg",
            "-y",
            "-i", str(video_no_audio_path),
            "-i", str(audio_source_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(out_p),
        ]
        subprocess.run(cmd_fallback, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

    return str(out_p)


def probe_video(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    data = json.loads(res.stdout)

    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    duration = float(v_stream.get("duration") or data.get("format", {}).get("duration", 0.0))
    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    fps_str = v_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        n, d = map(float, fps_str.split("/"))
        fps = n / d if d > 0 else 30.0
    else:
        fps = float(fps_str)

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": a_stream is not None,
    }


class FFmpegVideoAdapter(VideoUtilPort):
    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        return extract_audio(video_path, output_audio_path)

    def mux_audio_video(self, video_no_audio_path: str, audio_source_path: str, output_path: str) -> str:
        return mux_audio_video(video_no_audio_path, audio_source_path, output_path)

    def probe_video(self, video_path: str) -> Dict[str, Any]:
        return probe_video(video_path)
