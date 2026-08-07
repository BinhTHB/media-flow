"""
Preserve Action

Leaves frame pixels untouched.
"""

from __future__ import annotations

from typing import Any

from src.core.models.bbox import BBox


class PreserveAction:
    def execute(self, frame: Any, bbox: BBox) -> Any:
        return frame
