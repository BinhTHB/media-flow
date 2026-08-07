"""
Action Policy Engine

Evaluates class-aware spatial-temporal tracks against user/preset ProcessingPolicyConfig
and generates a deterministic RenderingPlan containing composable background and overlay TrackActionPlans.

Identifies required lazy enrichments (OCR, STT, Translation, Asset Loading)
so expensive operations only run when explicitly required by an active overlay action.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.domain.entities.action_plan import FrameWindowAction, RenderingPlan, TrackActionPlan
from src.domain.entities.canonical_detection import ClassAwareTrack
from src.domain.entities.processing_policy import ProcessingPolicyConfig, BackgroundActionType, OverlayActionType


class ActionPolicyEngine:
    def __init__(self, policy_config: ProcessingPolicyConfig):
        self.policy_config = policy_config

    def create_rendering_plan(
        self,
        job_id: str,
        tracks: List[ClassAwareTrack],
        video_width: int,
        video_height: int,
        duration: float,
        fps: float,
    ) -> RenderingPlan:
        plan_actions: List[TrackActionPlan] = []

        for track in tracks:
            class_policy = self.policy_config.get_class_policy(track.class_name)

            if not class_policy.enabled:
                status = "ignored"
                reason = "class_disabled"
                bg_dict = {"action": BackgroundActionType.PRESERVE.value}
                overlay_dicts = []
            elif track.mean_confidence < class_policy.confidence_threshold:
                status = "ignored"
                reason = "below_confidence_threshold"
                bg_dict = {"action": BackgroundActionType.PRESERVE.value}
                overlay_dicts = []
            else:
                status = "active"
                reason = "policy_preserve" if class_policy.background.action == BackgroundActionType.PRESERVE.value else ""
                bg_dict = class_policy.background.to_dict()
                # Sort overlays deterministically by z_order
                sorted_overlays = sorted(class_policy.overlays, key=lambda o: o.z_order)
                overlay_dicts = [o.to_dict() for o in sorted_overlays]

            # Construct frame window actions for exact temporal-spatial rendering
            frame_windows: List[FrameWindowAction] = []
            for group in track.groups:
                norm_box = list(group.union_bbox_normalized)
                xyxy_box = list(group.union_bbox_xyxy)
                frame_windows.append(
                    FrameWindowAction(
                        frame_index=group.frame_index,
                        timestamp_sec=group.timestamp_sec,
                        bbox_xyxy=xyxy_box,
                        bbox_normalized=norm_box,
                    )
                )

            action_plan = TrackActionPlan(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                status=status,
                reason=reason,
                background=bg_dict,
                overlays=overlay_dicts,
                start_time=track.start_time,
                end_time=track.end_time,
                start_frame=track.start_frame,
                end_frame=track.end_frame,
                confidence=track.mean_confidence,
                frame_windows=frame_windows,
            )
            plan_actions.append(action_plan)

        return RenderingPlan(
            job_id=job_id,
            video_width=video_width,
            video_height=video_height,
            duration=duration,
            fps=fps,
            actions=plan_actions,
            metadata={
                "policy": self.policy_config.to_dict(),
                "total_tracks": len(tracks),
            },
        )

    def get_tracks_requiring_ocr(self, plan: RenderingPlan) -> List[tuple[TrackActionPlan, Dict[str, Any]]]:
        results = []
        for act in plan.actions:
            if act.status != "active":
                continue
            for o in act.overlays:
                if o.get("action") == OverlayActionType.REPLACE_TEXT.value and o.get("text_source", "ocr") in ("ocr", "hybrid"):
                    results.append((act, o))
        return results

    def get_tracks_requiring_stt(self, plan: RenderingPlan) -> List[tuple[TrackActionPlan, Dict[str, Any]]]:
        results = []
        for act in plan.actions:
            if act.status != "active":
                continue
            for o in act.overlays:
                if o.get("action") == OverlayActionType.REPLACE_TEXT.value and o.get("text_source") in ("stt", "hybrid"):
                    results.append((act, o))
        return results

    def get_tracks_requiring_assets(self, plan: RenderingPlan) -> List[tuple[TrackActionPlan, Dict[str, Any]]]:
        results = []
        for act in plan.actions:
            if act.status != "active":
                continue
            for o in act.overlays:
                if o.get("action") == OverlayActionType.REPLACE_IMAGE.value and o.get("image_asset_path"):
                    results.append((act, o))
        return results
