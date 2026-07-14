import subprocess
import os
from pathlib import Path
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.application.ports.video_provider import VideoProvider
from src.infrastructure.runtime.ffmpeg_capability_detector import detect_ffmpeg_capabilities
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args
from src.infrastructure.runtime.resource_usage_report import write_resource_usage_report
from src.shared.exceptions import ProviderError

class FFmpegVideoProvider(VideoProvider):
    def __init__(
        self,
        output_dir: str = "output",
        ffmpeg_config: FFmpegConfig | None = None,
        resource_policy: ResourcePolicy | None = None,
        ocr_device: str = "not_used",
        asr_device: str = "not_used",
    ):
        self.output_dir = output_dir
        self.ffmpeg_config = ffmpeg_config or FFmpegConfig()
        self.resource_policy = resource_policy or ResourcePolicy()
        self.ocr_device = ocr_device
        self.asr_device = asr_device
        
    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise ProviderError("FFmpeg is not installed or not found in PATH.") from e

    def get_dimensions(self, video_path: str) -> tuple[int, int]:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=s=x:p=0", video_path],
                capture_output=True, check=True, text=True, timeout=30,
            )
            width, height = result.stdout.strip().split("x")
            return int(width), int(height)
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ProviderError(f"FFprobe could not read video dimensions: {video_path}") from exc

    def get_duration(self, video_path: str) -> float:
        if not os.path.exists(video_path):
            raise ProviderError(f"Input video file not found: {video_path}")
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=30,
            )
            duration = float(result.stdout.strip())
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
            raise ProviderError(f"FFprobe could not read video duration: {video_path}") from e
        if duration <= 0:
            raise ProviderError(f"Video duration must be positive: {video_path}")
        return duration
    
    def extract_audio(self, video_path: str, output_path: str | None = None) -> str:
        self._check_ffmpeg()
        if not os.path.exists(video_path):
            raise ProviderError(f"Input video file not found: {video_path}")
            
        if output_path:
            audio_path = output_path
        else:
            base = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(self.output_dir, f"{base}_extracted.wav")
            
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"FFmpeg audio extraction timed out for {video_path}") from e
        except subprocess.CalledProcessError as e:
            raise ProviderError(f"FFmpeg audio extraction failed: {e.stderr.decode('utf-8', errors='ignore')}") from e
        
        if not os.path.exists(audio_path):
            raise ProviderError(f"FFmpeg did not produce output file at {audio_path}")
        
        return audio_path
    
    def render_video(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str | None = None,
        style: dict | None = None
    ) -> str:
        self._check_ffmpeg()
        if not os.path.exists(video_path):
            raise ProviderError(f"Input video file not found: {video_path}")
        if not os.path.exists(subtitle_path):
            raise ProviderError(f"Subtitle file not found: {subtitle_path}")
            
        if output_path:
            render_path = output_path
        else:
            base, ext = os.path.splitext(os.path.basename(video_path))
            render_path = os.path.join(self.output_dir, f"{base}_rendered{ext}")
            
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        
        filter_name = "ass" if subtitle_path.lower().endswith(".ass") else "subtitles"
        escaped_subtitle_path = self._escape_filter_path(subtitle_path)
        filter_str = f"{filter_name}='{escaped_subtitle_path}'"
        capabilities = detect_ffmpeg_capabilities()
        plan = build_ffmpeg_encode_args(
            video_path, render_path, self.ffmpeg_config, True,
            capabilities,
        )
        cmd = list(plan.args)
        cmd[cmd.index("-i") + 2:cmd.index("-c:v")] = ["-vf", filter_str]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"FFmpeg video rendering timed out for {video_path}") from e
        except subprocess.CalledProcessError as e:
            raise ProviderError(f"FFmpeg video rendering failed: {e.stderr.decode('utf-8', errors='ignore')}") from e
            
        if not os.path.exists(render_path) or os.path.getsize(render_path) == 0:
            raise ProviderError(f"FFmpeg did not produce output file at {render_path}")
        render_file = Path(render_path)
        output_root = (
            render_file.parent.parent
            if render_file.parent.name == "videos"
            else render_file.parent
        )
        write_resource_usage_report(
            output_root, self.resource_policy, capabilities, plan,
            ocr_device=self.ocr_device, asr_device=self.asr_device,
            expensive_steps_skipped=[
                "inpaint_light", "full_frame_segmentation"
            ] if self.resource_policy.profile != "quality" else [
                "full_frame_segmentation"
            ],
        )
            
        return render_path

    def _escape_filter_path(self, path: str) -> str:
        escaped_path = path.replace(chr(92), "/")
        if ":" in escaped_path:
            drive, rest = escaped_path.split(":", 1)
            escaped_path = f"{drive}\\:{rest}"
        return escaped_path
