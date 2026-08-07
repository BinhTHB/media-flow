"""
Processing Request and Result Data Transfer Objects

Defines the clean Application API boundary (ProcessingRequest / ProcessingResult)
so external consumers or future Web UI / REST API frontends can invoke the backend
by configuring typed data structures without knowing internal provider details.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.domain.entities.processing_policy import ProcessingPolicyConfig, PRESET_BLUR_ALL, get_preset_policy


@dataclass
class ProcessingRequest:
    input_video: str
    output_video: str
    job_id: Optional[str] = None
    device: str = "auto"  # auto | cuda | cpu
    preset: Optional[str] = None  # blur-all | translate-subtitle | preserve-all
    policy_config: Optional[ProcessingPolicyConfig] = None
    policy_file: Optional[str] = None
    speech_provider: str = "faster-whisper"
    translation_provider: str = "gemini"
    ocr_provider: str = "hybrid_gemini"
    target_language: str = "vi"
    overwrite: bool = True

    def resolve_policy(self) -> ProcessingPolicyConfig:
        if self.policy_config is not None:
            return self.policy_config
        if self.policy_file:
            return ProcessingPolicyConfig.from_yaml_file(self.policy_file)
        if self.preset:
            return get_preset_policy(self.preset)
        return PRESET_BLUR_ALL

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.policy_config is not None:
            result["policy_config"] = self.policy_config.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class ProcessingResult:
    job_id: str
    status: str  # SUCCESS | FAILED | BLOCKED
    input_video_path: str
    output_video_path: str
    duration_sec: float = 0.0
    fps: float = 0.0
    resolution: str = ""
    detections_summary: Dict[str, int] = field(default_factory=dict)
    tracks_summary: Dict[str, int] = field(default_factory=dict)
    actions_summary: Dict[str, int] = field(default_factory=dict)
    action_plan_path: Optional[str] = None
    workspace_dir: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
