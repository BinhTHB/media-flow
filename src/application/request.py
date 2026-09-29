"""
Processing Request DTO
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from src.core.models.speech_segment import SpeechSegment
from src.features.audio.pipeline import AudioRequest
from src.features.visual.pipeline import VisualRequest


@dataclass
class ProcessingRequest:
    input_video: str
    output_video: str
    job_id: Optional[str] = None
    audio: AudioRequest = field(default_factory=AudioRequest)
    visual: VisualRequest = field(default_factory=VisualRequest)
    device: str = "auto"  # auto | cuda | cpu
    overwrite: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingRequest":
        audio_data = dict(data.get("audio", {}))
        if "segments" in audio_data and audio_data["segments"] is not None:
            audio_data["segments"] = [
                SpeechSegment.from_dict(s) if isinstance(s, dict) else s
                for s in audio_data["segments"]
            ]
        visual_data = dict(data.get("visual", {}))
        if "segments" in visual_data and visual_data["segments"] is not None:
            visual_data["segments"] = [
                SpeechSegment.from_dict(s) if isinstance(s, dict) else s
                for s in visual_data["segments"]
            ]
        return cls(
            input_video=str(data["input_video"]),
            output_video=str(data["output_video"]),
            job_id=data.get("job_id"),
            audio=AudioRequest(**{k: v for k, v in audio_data.items() if k in AudioRequest.__dataclass_fields__}),
            visual=VisualRequest(**{k: v for k, v in visual_data.items() if k in VisualRequest.__dataclass_fields__}),
            device=str(data.get("device", "auto")),
            overwrite=bool(data.get("overwrite", True)),
        )
