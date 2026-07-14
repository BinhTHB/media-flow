from dataclasses import dataclass


@dataclass(frozen=True)
class SubtitlePolicyDecision:
    decision: str
    reason: str | None = None


class SubtitleRenderPolicy:
    _OVERRIDE_REASONS = {
        "insufficient_track_evidence",
        "video_level_skip",
        "unstable_track",
    }

    @staticmethod
    def _risk(interval: dict, candidate: dict, name: str) -> float | None:
        value = candidate.get(name, interval.get(name))
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _valid_bbox(bbox: object) -> bool:
        return isinstance(bbox, dict) and (
            isinstance(bbox.get("left"), (int, float))
            and isinstance(bbox.get("top"), (int, float))
            and isinstance(bbox.get("right"), (int, float))
            and isinstance(bbox.get("bottom"), (int, float))
            and bbox["left"] < bbox["right"]
            and bbox["top"] < bbox["bottom"]
        )

    def adaptive_override_evidence(
        self,
        interval: dict | None,
        *,
        has_target_text: bool,
        video_decision: str | None = None,
    ) -> dict:
        if interval is None:
            return {"eligible": False, "applied": False, "reason": "missing_interval"}
        candidate = interval.get("selected_candidate") or {}
        original_reason = (
            interval.get("reason")
            or interval.get("review_reason")
            or interval.get("skip_reason")
        )
        eligible_reason = (
            original_reason
            if original_reason in self._OVERRIDE_REASONS
            else "video_level_skip"
            if video_decision == "SKIP_REPLACEMENT"
            else None
        )
        evidence = {
            "eligible": eligible_reason is not None,
            "applied": False,
            "reason": None,
            "original_reason": original_reason,
            "selected_candidate_bbox": candidate.get("bbox") or interval.get("source_bbox"),
            "local_candidate_confidence": candidate.get("final_candidate_score"),
            "target_text_present": has_target_text,
            "mask_area_ratio": interval.get("mask_area_ratio"),
            "scene_text_risk_score": self._risk(interval, candidate, "scene_text_risk_score"),
            "menu_signage_risk_score": self._risk(interval, candidate, "menu_signage_risk_score"),
            "wall_sign_risk_score": self._risk(
                interval, candidate, "perspective_attached_text_risk"
            ),
            "watermark_risk_score": self._risk(interval, candidate, "watermark_risk_score"),
            "person_risk_score": self._risk(interval, candidate, "person_risk_score"),
        }
        if not evidence["eligible"]:
            evidence["reason"] = "original_rejection_not_override_eligible"
        elif not has_target_text:
            evidence["reason"] = "missing_target_text"
        elif not self._valid_bbox(evidence["selected_candidate_bbox"]):
            evidence["reason"] = "missing_or_invalid_selected_candidate_bbox"
        elif (
            not isinstance(evidence["local_candidate_confidence"], (int, float))
            or evidence["local_candidate_confidence"] < 0.90
        ):
            evidence["reason"] = "local_candidate_confidence_below_0_90"
        elif not isinstance(evidence["mask_area_ratio"], (int, float)) or evidence["mask_area_ratio"] > 0.20:
            evidence["reason"] = "mask_area_above_0_20"
        else:
            for key, reason in (
                ("scene_text_risk_score", "scene_text_risk"),
                ("menu_signage_risk_score", "menu_signage_risk"),
                ("wall_sign_risk_score", "wall_sign_risk"),
                ("watermark_risk_score", "watermark_risk"),
                ("person_risk_score", "missing_person_risk_evidence"),
            ):
                value = evidence[key]
                if value is None:
                    evidence["reason"] = reason
                    break
                if value >= 0.20:
                    evidence["reason"] = (
                        "person_risk"
                        if key == "person_risk_score"
                        else reason
                    )
                    break
            else:
                evidence["applied"] = True
                evidence["reason"] = "safe_local_candidate_evidence"
        return evidence

    def adaptive_decide(
        self,
        interval: dict | None,
        *,
        has_target_text: bool,
        video_decision: str | None = None,
    ) -> SubtitlePolicyDecision:
        if not has_target_text:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "missing_target_text")
        if interval is None:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "missing_interval")
        if interval.get("decision") != "AUTO_REPLACE":
            override = self.adaptive_override_evidence(
                interval, has_target_text=has_target_text, video_decision=video_decision
            )
            if override["applied"]:
                return SubtitlePolicyDecision(
                    "AUTO_SOURCE_REPLACE", "candidate_level_override"
                )
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", override["reason"])
        if interval.get("layout_mode") not in {
            "stable_track_subtitle",
            "karaoke_subtitle",
            "floating_caption_subtitle",
        }:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "unsupported_layout")
        if interval.get("final_interval_confidence", 0.0) < 0.70:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "confidence_below_0_70")
        if interval.get("scene_text_risk_score", 1.0) >= 0.20:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "scene_text_risk")
        if interval.get("watermark_risk_score", 0.0) >= 0.20:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "watermark_risk")
        if interval.get("person_risk_score", 0.0) >= 0.20:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "person_risk")
        if interval.get("mask_area_ratio", 1.0) > 0.20:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "mask_area_above_0_20")
        if not interval.get("selected_candidate"):
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "missing_candidate")
        return SubtitlePolicyDecision("AUTO_SOURCE_REPLACE")

    def replacement_gate(
        self,
        *,
        confident_interval_count: int,
        output_mode_includes_replacement: bool,
    ) -> SubtitlePolicyDecision:
        if not output_mode_includes_replacement:
            return SubtitlePolicyDecision(
                "BLOCK_SOURCE_REPLACE", "output_mode_does_not_include_replacement"
            )
        if confident_interval_count < 1:
            return SubtitlePolicyDecision(
                "BLOCK_SOURCE_REPLACE", "no_high_confidence_source_replace_intervals"
            )
        return SubtitlePolicyDecision("AUTO_SOURCE_REPLACE")

    def sidecar_generation_gate(
        self, *, target_timeline_exists: bool
    ) -> SubtitlePolicyDecision:
        if not target_timeline_exists:
            return SubtitlePolicyDecision(
                "BLOCK_SIDECAR_GENERATION", "missing_target_timeline"
            )
        return SubtitlePolicyDecision("GENERATE_SIDECARS")

    def static_render_gate(
        self,
        *,
        target_timeline_exists: bool,
        timing_valid: bool,
        output_mode_includes_static: bool,
    ) -> SubtitlePolicyDecision:
        if not target_timeline_exists:
            return SubtitlePolicyDecision("BLOCK_STATIC_RENDER", "missing_target_timeline")
        if not timing_valid:
            return SubtitlePolicyDecision("BLOCK_STATIC_RENDER", "invalid_timing")
        if not output_mode_includes_static:
            return SubtitlePolicyDecision(
                "BLOCK_STATIC_RENDER", "output_mode_does_not_include_static"
            )
        return SubtitlePolicyDecision("STATIC_SUBTITLE_RENDER")

    def ux_decide(self, interval: dict | None, resource_profile: str) -> SubtitlePolicyDecision:
        if interval is None:
            return SubtitlePolicyDecision("static_fallback_allowed", "no_source_region")
        layout = interval.get("layout_mode", "")
        original_visible = interval.get("source_subtitle_present_score", 0.0) >= 0.70
        bottom_candidate = (
            layout == "stable_track_subtitle"
            and interval.get("source_bbox") is not None
            and interval.get("final_interval_confidence", 0.0) >= 0.80
            and interval.get("target_text_fit_score", 0.0) >= 0.90
            and interval.get("scene_text_risk_score", 1.0) < 0.20
        )
        if bottom_candidate:
            return SubtitlePolicyDecision("bottom_subtitle_replace_candidate")
        if layout == "karaoke_subtitle":
            return SubtitlePolicyDecision("karaoke_word_keep_or_review")
        if original_visible:
            return SubtitlePolicyDecision(
                "static_fallback_suppressed_due_to_duplicate_risk",
                "original_subtitle_visible",
            )
        if resource_profile in {"low", "balanced"}:
            return SubtitlePolicyDecision("static_fallback_allowed")
        return SubtitlePolicyDecision("review_required")

    def decide(
        self,
        interval: dict | None,
        *,
        has_target_text: bool,
        resource_profile: str,
    ) -> SubtitlePolicyDecision:
        if not has_target_text:
            return SubtitlePolicyDecision("SKIP_REPLACEMENT", "missing_target_text")
        if interval is None:
            return SubtitlePolicyDecision("STATIC_FALLBACK", "timeline_only")
        decision = interval.get("decision")
        layout = interval.get("layout_mode", "")
        confidence = interval.get("final_interval_confidence", 0.0)
        scene_risk = interval.get("scene_text_risk_score", 1.0)
        if (
            decision == "AUTO_REPLACE"
            and layout in {
                "stable_track_subtitle",
                "karaoke_subtitle",
                "floating_caption_subtitle",
            }
            and confidence >= (0.90 if resource_profile == "low" else 0.80)
            and scene_risk < 0.20
        ):
            return SubtitlePolicyDecision("AUTO_SOURCE_REPLACE")
        if decision == "REVIEW_REQUIRED" and interval.get("source_subtitle_present_score", 0.0) >= 0.70:
            return SubtitlePolicyDecision("REVIEW_REQUIRED", interval.get("review_reason"))
        if decision == "SKIP_REPLACEMENT":
            return SubtitlePolicyDecision("STATIC_FALLBACK", interval.get("skip_reason"))
        return SubtitlePolicyDecision("STATIC_FALLBACK", "replacement_not_confident")
