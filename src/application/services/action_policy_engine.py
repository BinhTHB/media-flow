"""
Action Policy Engine

Evaluates class-aware spatial-temporal tracks against user/preset ProcessingPolicyConfig
and generates a deterministic RenderingPlan containing TrackActionPlans.

Differentiates active actions from ignored/below-threshold tracks.
Identifies required lazy enrichments (OCR, STT, Translation, Asset Loading)
so expensive operations only run when explicitly required by an active action plan.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from src.domain.entities.action_plan import FrameWindowAction, RenderingPlan, TrackActionPlan
from src.domain.entities.canonical_detection import ClassAwareTrack
from src.domain.entities.processing_policy import ActionType, ProcessingPolicyConfig


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
                action_type = ActionType.PRESERVE.value
            elif track.mean_confidence < class_policy.confidence_threshold:
                status = "ignored"
                reason = "below_confidence_threshold"
                action_type = ActionType.PRESERVE.value
            else:
                status = "active"
                reason = "policy_preserve" if class_policy.action == ActionType.PRESERVE.value else ""
                action_type = class_policy.action

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

            params: Dict[str, Any] = {
                "blur_strength": class_policy.blur_strength,
                "padding_px": class_policy.padding_px,
                "solid_color_hex": class_policy.solid_color_hex,
                "text_source": class_policy.text_source,
                "target_language": class_policy.target_language,
                "font": class_policy.font,
                "font_size": class_policy.font_size,
                "stroke_width": class_policy.stroke_width,
                "placement": class_policy.placement,
                "background_action": class_policy.background_action,
                "opacity": class_policy.opacity,
                "fit": class_policy.fit,
                "scale": class_policy.scale,
                "z_order": class_policy.z_order,
            }

            action_plan = TrackActionPlan(
                track_id=track.track_id,
                class_id=track.class_id,
                class_name=track.class_name,
                action_type=action_type,
                status=status,
                reason=reason,
                start_time=track.start_time,
                end_time=track.end_time,
                start_frame=track.start_frame,
                end_frame=track.end_frame,
                confidence=track.mean_confidence,
                params=params,
                frame_windows=frame_windows,
                replacement_text="",
                replacement_image_path=class_policy.image_asset_path,
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

    def get_tracks_requiring_ocr(self, plan: RenderingPlan) -> List[TrackActionPlan]:
        return [
            act for act in plan.actions
            if act.status == "active"
            and act.action_type == ActionType.REPLACE_TEXT.value
            and act.params.get("text_source", "ocr") in ("ocr", "hybrid")
        ]

    def get_tracks_requiring_stt(self, plan: RenderingPlan) -> List[TrackActionPlan]:
        return [
            act for act in plan.actions
            if act.status == "active"
            and act.action_type == ActionType.REPLACE_TEXT.value
            and act.params.get("text_source") in ("stt", "hybrid")
        ]

    def get_tracks_requiring_assets(self, plan: RenderingPlan) -> List[TrackActionPlan]:
        return [
            act for act in plan.actions
            if act.status == "active"
            and act.action_type == ActionType.REPLACE_IMAGE.value
            and act.replacement_image_path
        ]
