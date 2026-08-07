"""
Action Plan & Rendering Plan Domain Entities

Defines machine-readable action plans for multi-action (background + overlays) composable rendering.
Exports output/jobs/<job_id>/action_plan.json.
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
    status: str = "active"  # active | ignored
    reason: str = ""  # below_confidence_threshold | class_disabled | policy_preserve
    background: Dict[str, Any] = field(default_factory=dict)  # action: preserve|blur|solid_cover
    overlays: List[Dict[str, Any]] = field(default_factory=list)  # list of overlay actions
    start_time: float = 0.0
    end_time: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    confidence: float = 0.0
    frame_windows: List[FrameWindowAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "status": self.status,
            "reason": self.reason,
            "background": self.background,
            "overlays": self.overlays,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": round(self.confidence, 4),
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
            if act.status == "active":
                bg_act = act.background.get("action", "preserve")
                counts[bg_act] = counts.get(bg_act, 0) + 1
                for o in act.overlays:
                    o_act = o.get("action")
                    if o_act:
                        counts[o_act] = counts.get(o_act, 0) + 1
            else:
                counts["ignored"] = counts.get("ignored", 0) + 1
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
