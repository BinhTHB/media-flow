"""
Action Plan & Rendering Plan Domain Entities

Defines machine-readable action plans for rendering video artifacts and exporting
outputs/<job_id>/action_plan.json for inspection/frontend consumption.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class FrameWindowAction:
    frame_index: int
    timestamp_sec: float
    bbox_xyxy: List[float]
    bbox_normalized: List[float]


@dataclass
class TrackActionPlan:
    track_id: str
    class_id: int
    class_name: str
    action_type: str  # blur | preserve | solid_cover | replace_text | replace_image
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)
    frame_windows: List[FrameWindowAction] = field(default_factory=list)
    replacement_text: str = ""
    replacement_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "action_type": self.action_type,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": round(self.confidence, 4),
            "params": self.params,
            "replacement_text": self.replacement_text,
            "replacement_image_path": self.replacement_image_path,
            "frame_windows": [asdict(fw) for fw in self.frame_windows],
        }


@dataclass
class RenderingPlan:
    job_id: str
    video_width: int
    video_height: int
    duration: float
    fps: float
    actions: List[TrackActionPlan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for act in self.actions:
            counts[act.action_type] = counts.get(act.action_type, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "duration": round(self.duration, 4),
            "fps": round(self.fps, 2),
            "action_counts": self.summary(),
            "actions": [act.to_dict() for act in self.actions],
            "metadata": self.metadata,
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path
