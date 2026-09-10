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
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return str(out_p)


def mux_audio_video(video_no_audio_path: str, audio_source_path: str, output_path: str) -> str:
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_source_path)

    # Dubbed timelines are canonical WAV/PCM. Some FFmpeg builds allow PCM inside
    # MP4, but browser/player support is inconsistent, so encode WAV/PCM to AAC
    # proactively instead of waiting for stream-copy to fail.
    force_aac = audio_path.suffix.lower() in {".wav", ".pcm"}
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
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        # Keep the video stream untouched and encode audio to universally supported AAC.
        cmd_fallback = [
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
        try:
            subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            cmd_reencode = [
                "ffmpeg",
                "-y",
                "-i", str(video_no_audio_path),
                "-i", str(audio_source_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(out_p),
            ]
            subprocess.run(cmd_reencode, capture_output=True, text=True, check=True)

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
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
