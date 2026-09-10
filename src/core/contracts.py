"""
Core Port Contracts

Defines pure Python abstract base classes and interfaces for features & infrastructure providers.
Zero external third-party SDK dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.models.bbox import BBox
from src.core.models.detection import Detection
from src.core.models.speech_segment import SpeechSegment


class TranslationPort(ABC):
    @abstractmethod
    def translate_texts(
        self, texts: List[str], target_language: str = "vi", source_language: str = "auto"
    ) -> List[str]:
        """Translate a batch of raw text strings."""
        pass


class DetectorPort(ABC):
    @abstractmethod
    def detect_video(self, video_path: str, stride: int = 1) -> List[Detection]:
        """Detect objects in a video file and return canonical Detection objects."""
        pass


class SceneBoundaryPort(ABC):
    @abstractmethod
    def detect_cuts(self, video_path: str) -> List[int]:
        """Return frame indices where a new scene starts."""
        pass


class AudioExtractionPort(ABC):
    @abstractmethod
    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        """Extract a canonical audio stream from a video file."""
        pass


class ASRPort(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "auto") -> List[SpeechSegment]:
        """Transcribe speech in an audio file into SpeechSegments."""
        pass


class ASRCorrectionPort(ABC):
    @abstractmethod
    def correct_segments(
        self,
        segments: List[SpeechSegment],
        language: str = "auto",
        audio_path: Optional[str] = None,
    ) -> List[SpeechSegment]:
        """Conservatively correct ASR text while preserving segment ids/timestamps."""
        pass


class SpokenRewritePort(ABC):
    @abstractmethod
    def rewrite_for_duration(
        self,
        text: str,
        source_text: str,
        target_duration: float,
        current_duration: float,
        target_language: str = "vi",
    ) -> str:
        """Shorten spoken copy to fit a target duration while preserving essential meaning."""
        pass


class OCRPort(ABC):
    @abstractmethod
    def read_crops(self, crop_images: List[Any]) -> List[str]:
        """Transcribe text from cropped images."""
        pass


class TTSPort(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> str:
        """Synthesize speech audio from text and save to output_path."""
        pass


class VideoUtilPort(ABC):
    @abstractmethod
    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        """Extract audio stream to WAV file."""
        pass

    @abstractmethod
    def mux_audio_video(self, video_no_audio_path: str, audio_source_path: str, output_path: str) -> str:
        """Mux video and audio streams together into output_path."""
        pass

    @abstractmethod
    def probe_video(self, video_path: str) -> Dict[str, Any]:
        """Probe video duration, resolution, FPS, and audio stream presence."""
        pass
