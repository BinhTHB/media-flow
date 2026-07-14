from dataclasses import dataclass

from src.shared.exceptions import ValidationError


@dataclass
class TTSConfig:
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "+0%"
    original_audio_mode: str = "lower"
    original_audio_volume: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.voice, str) or not self.voice.strip():
            raise ValidationError("TTS voice must be a non-empty string.")
        if not isinstance(self.rate, str) or not self.rate.endswith("%"):
            raise ValidationError("TTS rate must be a percentage string.")
        if self.original_audio_mode not in ("keep", "lower", "replace"):
            raise ValidationError(
                "original_audio_mode must be one of: 'keep', 'lower', 'replace'."
            )
        if (
            not isinstance(self.original_audio_volume, (int, float))
            or not 0.0 <= self.original_audio_volume <= 1.0
        ):
            raise ValidationError("original_audio_volume must be between 0.0 and 1.0.")
