"""
Processing Result DTO
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

ProcessingStatus = Literal["SUCCESS", "PARTIAL", "FAILED", "BLOCKED"]
StageStatus = Literal["SUCCESS", "SKIPPED", "PARTIAL", "FAILED", "BLOCKED"]


@dataclass
class ProcessingResult:
    job_id: str
    status: ProcessingStatus
    input_video_path: str
    output_video_path: str
    duration_sec: float = 0.0
    fps: float = 0.0
    resolution: str = ""
    audio_status: str = "SKIPPED"
    visual_status: str = "SKIPPED"
    detections_summary: Dict[str, int] = field(default_factory=dict)
    tracks_summary: Dict[str, int] = field(default_factory=dict)
    workspace_dir: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
