import os
import subprocess

from src.application.ports.tts_provider import TTSProvider
from src.shared.exceptions import ProviderError


class EdgeTTSProvider(TTSProvider):
    def __init__(self, executable: str = "edge-tts", timeout: int = 120) -> None:
        self.executable = executable
        self.timeout = timeout

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str,
        rate: str,
    ) -> str:
        if not text.strip():
            raise ProviderError("Cannot synthesize empty TTS text.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = [
            self.executable,
            "--text",
            text,
            "--voice",
            voice,
            "--rate",
            rate,
            "--write-media",
            output_path,
        ]
        try:
            subprocess.run(
                command,
                capture_output=True,
                check=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"Edge TTS synthesis failed for {output_path}.") from exc
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ProviderError(f"Edge TTS did not create audio at {output_path}.")
        return output_path
