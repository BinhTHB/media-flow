"""
Core Port Contracts

Defines pure Python abstract base classes and interfaces for features & infrastructure providers.
Zero external third-party SDK dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.models.account import AccountIdentity
from src.core.models.detection import Detection
from src.core.models.ingest import IngestResult, SourceSpec
from src.core.models.publication import PublishItem, PublishRecord, VideoMetadata
from src.core.models.speech_segment import SpeechSegment
from src.core.models.variation import VariationProfile


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

    def open_encoded_frame_writer(self, output_path: str, fps: float, width: int, height: int) -> Any:
        """Return a frame writer exposing ``write(frame)``/``release()`` that
        encodes a web-compatible (H.264 / yuv420p) file in a single pass.

        Optional port extension: implementers may raise NotImplementedError,
        in which case callers fall back to their local default encoder.
        """
        raise NotImplementedError


class TransientPublishError(RuntimeError):
    """Publication failed in a way worth retrying later (rate limit, quota
    window, 5xx). Raised by PublicationPort providers; the publishing use
    case owns the retry/backoff policy."""


class PublicationPort(ABC):
    """Publishes one finished video to one platform.

    Providers must raise TransientPublishError for retryable failures so the
    use case can back off; any other exception marks the item FAILED."""

    platform_name: str = "unknown"

    @abstractmethod
    def publish(self, item: PublishItem) -> PublishRecord:
        """Publish a PublishItem; returns a PublishRecord (core models)."""
        pass


class MetadataGenerationPort(ABC):
    """Generates platform metadata (title/description/tags) for a video."""

    @abstractmethod
    def generate(self, video_path: str, hints: Optional[Dict[str, Any]] = None) -> VideoMetadata:
        """Return a VideoMetadata (core models); must not raise on model
        failure — providers fall back to safe filename-derived metadata."""
        pass


class VideoVariationPort(ABC):
    """Applies a :class:`VariationProfile` to a finished video.

    Implementations must be deterministic for a given (input, profile) pair so
    a re-run produces the same bytes and the publication state stays truthful.
    The port owns pixels and audio only; it never touches metadata or upload.
    """

    @abstractmethod
    def apply(self, video_path: str, profile: VariationProfile, output_path: str) -> str:
        """Write the varied video to ``output_path`` and return that path."""
        pass


class SessionStorePort(ABC):
    """Persists browser sessions (cookie jars) for cookie-based publishers.

    Only the publisher implementation interprets the payload; the store treats
    it as an opaque blob so no platform-specific cookie format leaks upward.
    """

    @abstractmethod
    def load(self, account: AccountIdentity) -> Optional[str]:
        """Return the stored session path for the account, or None."""
        pass

    @abstractmethod
    def save(self, account: AccountIdentity, session_path: str) -> str:
        """Register ``session_path`` as the account's current session."""
        pass

    @abstractmethod
    def invalidate(self, account: AccountIdentity, note: str = "") -> None:
        """Mark the account's session as no longer usable."""
        pass


class DownloaderPort(ABC):
    """Fetches one remote video into local storage.

    Implementations own the network and the downloader-specific options; the
    caller only sees a canonical ``IngestResult``. Providers must raise
    ``TransientIngestError`` for retryable failures (timeouts, 5xx, rate
    limits) so the use case can back off, and any other exception for
    permanent ones (private video, geo-block, unsupported URL).
    """

    @abstractmethod
    def download(self, source: SourceSpec, output_dir: str) -> IngestResult:
        """Download ``source`` under ``output_dir`` and report the outcome."""
        pass


class SourceCatalogPort(ABC):
    """Lists video URLs inside a channel, playlist, or single video page.

    The result is newest-first and capped by ``limit``. A single video URL
    comes back as that one URL. Implementations must not download media.
    """

    @abstractmethod
    def list_entries(self, url: str, *, limit: int) -> List[str]:
        """Return up to ``limit`` video URLs for ``url``."""
        pass


class TransientIngestError(RuntimeError):
    """Ingest failed in a way worth retrying (network, rate limit, 5xx)."""
