import json
import os
import subprocess
from typing import List

from src.application.ports.dubbing_provider import DubbingProvider
from src.shared.exceptions import ProviderError


class FFmpegDubbingProvider(DubbingProvider):
    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout

    def merge_timeline(
        self, clips: List[dict], output_path: str, duration: float
    ) -> str:
        if not clips:
            raise ProviderError("Cannot merge an empty TTS manifest.")
        previous_end = 0.0
        for clip in clips:
            path = clip.get("audio_path", "")
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise ProviderError(f"TTS clip is missing or empty: {path}")
            start = float(clip.get("start", 0.0))
            end = float(clip.get("end", 0.0))
            if end <= start:
                raise ProviderError(f"TTS clip has an invalid target range: {path}")
            if start < previous_end - 0.001:
                raise ProviderError(f"TTS timeline contains overlapping clips: {path}")
            previous_end = end
        command = ["ffmpeg"]
        for clip in clips:
            command.extend(["-i", clip["audio_path"]])
        filters = []
        labels = []
        for index, clip in enumerate(clips):
            delay = max(0, round(clip["start"] * 1000))
            target_duration = float(clip["end"]) - float(clip["start"])
            actual_duration = self._probe_duration(clip["audio_path"])
            speed = actual_duration / target_duration
            tempo = self._atempo_filters(speed) if speed > 1.0 else []
            label = f"a{index}"
            chain = tempo + [
                f"apad=whole_dur={target_duration:.3f}",
                f"atrim=duration={target_duration:.3f}",
                f"adelay={delay}|{delay}",
            ]
            filters.append(f"[{index}:a]{','.join(chain)}[{label}]")
            labels.append(f"[{label}]")
        filters.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
            + f"apad=pad_dur={duration}[out]"
        )
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-t",
                str(duration),
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                "-y",
                output_path,
            ]
        )
        self._run(command, output_path)
        return output_path

    def render_dubbed_video(
        self,
        video_path: str,
        dubbed_audio_path: str,
        output_path: str,
        original_audio_mode: str,
        original_audio_volume: float,
    ) -> str:
        if original_audio_mode == "replace":
            command = [
                "ffmpeg", "-i", video_path, "-i", dubbed_audio_path,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-shortest", "-y", output_path,
            ]
        else:
            volume = 1.0 if original_audio_mode == "keep" else original_audio_volume
            command = [
                "ffmpeg", "-i", video_path, "-i", dubbed_audio_path,
                "-filter_complex",
                f"[0:a]volume={volume}[original];[original][1:a]amix=inputs=2:duration=first:normalize=0[mixed]",
                "-map", "0:v:0", "-map", "[mixed]", "-c:v", "copy",
                "-c:a", "aac", "-y", output_path,
            ]
        self._run(command, output_path)
        if original_audio_mode == "replace":
            streams = self._probe_audio_streams(output_path)
            if len(streams) != 1:
                raise ProviderError(
                    "Replace mode output must contain exactly one dubbed audio stream."
                )
        return output_path

    def _probe_duration(self, path: str) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", path,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=self.timeout,
            )
            duration = float(result.stdout.strip())
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"Could not measure TTS clip duration: {path}") from exc
        if duration <= 0:
            raise ProviderError(f"TTS clip duration must be positive: {path}")
        return duration

    def _probe_audio_streams(self, path: str) -> List[dict]:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=index,codec_name", "-of", "json", path,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=self.timeout,
            )
            return json.loads(result.stdout).get("streams", [])
        except (
            OSError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise ProviderError(f"Could not validate dubbed audio streams: {path}") from exc

    @staticmethod
    def _atempo_filters(speed: float) -> List[str]:
        filters = []
        while speed > 2.0:
            filters.append("atempo=2.000000")
            speed /= 2.0
        if speed > 1.000001:
            filters.append(f"atempo={speed:.6f}")
        return filters

    def _run(self, command: List[str], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=self.timeout)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"FFmpeg dubbing operation failed for {output_path}.") from exc
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ProviderError(f"FFmpeg did not create dubbed artifact at {output_path}.")
