"""
Audio Feature Models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.models.speech_segment import SpeechSegment


@dataclass
class AudioRequest:
    enabled: bool = True
    audio_path: str = ""
    source_language: str = "auto"
    target_language: str = "vi"
    dubbing_enabled: bool = False
    voice: str = "vi-VN-HoaiMyNeural"


@dataclass
class TTSClip:
    segment_id: str
    start_time: float
    end_time: float
    audio_clip_path: str
    duration: float


@dataclass
class AudioResult:
    enabled: bool
    status: str  # SUCCESS | SKIPPED | PARTIAL | FAILED
    extracted_audio_path: Optional[str] = None
    dubbed_audio_path: Optional[str] = None
    segments: List[SpeechSegment] = field(default_factory=list)
    tts_clips: List[TTSClip] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
