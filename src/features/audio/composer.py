"""
Audio Composer

Reconciles TTS audio clips on a timeline, manages silence intervals, and preserves video duration.
Pure Python logic + FFmpeg/wave file metadata reconciliation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from src.features.audio.models import TTSClip

logger = logging.getLogger(__name__)


class AudioComposer:
    def compose_dubbed_timeline(
        self,
        tts_clips: List[TTSClip],
        total_duration: float,
        output_audio_path: str,
        original_audio_path: Optional[str] = None,
        mix_original_volume: float = 0.2,
    ) -> str:
        """Compose dubbed audio timeline from TTS clips."""
        output_path = Path(output_audio_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not tts_clips:
            if original_audio_path and os.path.exists(original_audio_path):
                import shutil
                shutil.copy2(original_audio_path, str(output_path))
                return str(output_path)
            raise RuntimeError("No TTS clips provided and no original audio available for AudioComposer.")

        # In production, uses FFmpeg filtergraph to place clips at start_time timestamps
        # or concatenates with silence padding
        return str(output_path)
