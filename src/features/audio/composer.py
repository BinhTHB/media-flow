"""Audio timeline composer.

Fits each synthesized clip to its speech-segment window, places it at the
original timestamp, and renders a real dubbed WAV with FFmpeg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from src.features.audio.models import TTSClip


class AudioComposer:
    def __init__(self, max_speedup: float = 1.30) -> None:
        self.max_speedup = max(1.0, float(max_speedup))

    def compose_dubbed_timeline(
        self,
        tts_clips: List[TTSClip],
        total_duration: float,
        output_audio_path: str,
        original_audio_path: Optional[str] = None,
        mix_original_volume: float = 0.0,
    ) -> str:
        output_path = Path(output_audio_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_duration = max(0.1, float(total_duration))

        if not tts_clips:
            if original_audio_path and os.path.exists(original_audio_path):
                shutil.copy2(original_audio_path, str(output_path))
                return str(output_path)
            raise RuntimeError("No TTS clips provided and no original audio available for AudioComposer.")

        fitted_dir = output_path.parent / "fitted"
        fitted_dir.mkdir(parents=True, exist_ok=True)
        fitted: list[tuple[TTSClip, Path]] = []

        for index, clip in enumerate(tts_clips):
            source = Path(clip.audio_clip_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"TTS clip not found: {source}")
            target_duration = max(0.08, float(clip.end_time - clip.start_time))
            fitted_path = fitted_dir / f"fit_{index:05d}.wav"
            self._fit_clip(source, fitted_path, target_duration)
            fitted.append((clip, fitted_path))

        inputs: list[str] = []
        filters: list[str] = []
        mix_labels: list[str] = []
        for idx, (clip, fitted_path) in enumerate(fitted):
            inputs.extend(["-i", str(fitted_path)])
            delay_ms = max(0, int(round(float(clip.start_time) * 1000.0)))
            label = f"dub{idx}"
            filters.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[{label}]")
            mix_labels.append(f"[{label}]")

        original_index: int | None = None
        if (
            original_audio_path
            and os.path.exists(original_audio_path)
            and float(mix_original_volume) > 0.0
        ):
            original_index = len(fitted)
            inputs.extend(["-i", str(Path(original_audio_path).resolve())])
            filters.append(
                f"[{original_index}:a]volume={float(mix_original_volume):.4f},"
                f"aresample=48000,apad=whole_dur={total_duration:.6f},"
                f"atrim=duration={total_duration:.6f}[orig]"
            )
            mix_labels.append("[orig]")

        filters.append(
            "".join(mix_labels)
            + f"amix=inputs={len(mix_labels)}:duration=longest:normalize=0,"
            + f"atrim=duration={total_duration:.6f},apad=whole_dur={total_duration:.6f}[mix]"
        )

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            *inputs,
            "-filter_complex", ";".join(filters),
            "-map", "[mix]",
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            str(output_path),
        ]
        self._run(cmd, "compose dubbed timeline")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("AudioComposer did not produce a non-empty dubbed timeline.")
        return str(output_path)

    def trim_tts_silence(self, source: Path, output: Path) -> Path:
        """Trim only leading/trailing digital silence while preserving internal pauses."""
        output.parent.mkdir(parents=True, exist_ok=True)
        trim_filter = (
            "silenceremove=start_periods=1:start_duration=0.03:start_threshold=-50dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_duration=0.03:start_threshold=-50dB,"
            "areverse"
        )
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-filter:a", trim_filter,
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            str(output),
        ]
        self._run(cmd, f"trim TTS silence {source.name}")
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Silence trim produced no audio: {source}")
        return output

    def _fit_clip(self, source: Path, output: Path, target_duration: float) -> None:
        source_duration = self.probe_duration(source)
        if source_duration <= 0.0:
            raise RuntimeError(f"Cannot determine TTS clip duration: {source}")

        speed = source_duration / target_duration
        filters: list[str] = []
        if speed > 1.01:
            if speed > self.max_speedup + 1e-6:
                raise RuntimeError(
                    f"TTS clip {source.name} requires {speed:.3f}x speed-up, above "
                    f"naturalness cap {self.max_speedup:.3f}x; rewrite the spoken text first."
                )
            filters.append(self._atempo_filter(speed))
        # Never slow a short utterance just to fill its subtitle/ASR window.
        # Keep natural delivery and pad the remaining time with silence instead.
        filters.extend([
            "aresample=48000",
            f"apad=whole_dur={target_duration:.6f}",
            f"atrim=duration={target_duration:.6f}",
        ])
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-filter:a", ",".join(filters),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            str(output),
        ]
        self._run(cmd, f"fit TTS clip {source.name}")

    @staticmethod
    def _atempo_filter(speed: float) -> str:
        speed = max(0.05, min(20.0, float(speed)))
        factors: list[float] = []
        while speed > 2.0:
            factors.append(2.0)
            speed /= 2.0
        while speed < 0.5:
            factors.append(0.5)
            speed /= 0.5
        factors.append(speed)
        return ",".join(f"atempo={factor:.6f}" for factor in factors)

    @staticmethod
    def probe_duration(path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        try:
            return float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(f"Invalid ffprobe duration for {path}: {result.stdout!r}") from exc

    @staticmethod
    def _run(cmd: list[str], action: str) -> None:
        try:
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"FFmpeg failed to {action}: {detail[-1200:]}") from exc
