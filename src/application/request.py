"""
Processing Request DTO
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

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
        audio_data = data.get("audio", {})
        visual_data = data.get("visual", {})
        return cls(
            input_video=str(data["input_video"]),
            output_video=str(data["output_video"]),
            job_id=data.get("job_id"),
            audio=AudioRequest(**{k: v for k, v in audio_data.items() if k in AudioRequest.__dataclass_fields__}),
            visual=VisualRequest(**{k: v for k, v in visual_data.items() if k in VisualRequest.__dataclass_fields__}),
            device=str(data.get("device", "auto")),
            overwrite=bool(data.get("overwrite", True)),
        )
