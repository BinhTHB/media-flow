"""R1 motion-coupling rule (V2.1 architecture).

Rejects tracks that move with the camera background (motion coupling).

Global motion estimation via Lucas-Kanade optical flow + RANSAC affine fit.
Reject ONLY when: camera is moving (|v_bg| >= 1px) AND bbox matches background
velocity in BOTH direction AND magnitude for >= K consecutive frames.

Camera static → NEUTRAL. Perspective distortion / inverted text are secondary
signals only — never reject on their own.

Bất biến V2.1: reject-only. Insufficient evidence → NEUTRAL.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class R1Result:
    rejected: bool  # True = reject this track as motion-coupled
    neutral: bool   # True = insufficient evidence (treated as NEUTRAL)
    reason: str
    motion_score: float = 0.0  # 0-1, how strongly coupled to background
    bg_speed_px: float = 0.0
    track_speed_px: float = 0.0
    frames_verified: int = 0


class R1MotionCoupling:
    """Reject tracks that move with camera background motion.

    Uses Lucas-Kanade optical flow to estimate global affine motion
    between consecutive frames, then checks if the candidate track's text
    region moves consistently with the background.

    Key design decisions:
    - Masks out the text region itself when computing background flow
    - Only rejects when camera is clearly moving AND bbox matches bg motion
    - Static camera → always NEUTRAL (safe)
    """

    MIN_BG_SPEED_PX: float = 1.0  # Minimum bg motion to trigger rejection
    K_CONSECUTIVE: int = 2  # Frames of consistent motion coupling
    MATCH_ANGLE_TOLERANCE_DEG: float = 30.0
    MATCH_SPEED_RATIO: Tuple[float, float] = (0.5, 2.0)  # Track/bg speed ratio range
    TEXT_MASK_MARGIN: int = 20  # Extra margin around text for masking

    def __init__(self, frame_height: int = 720, frame_width: int = 1280) -> None:
        self._frame_height = frame_height
        self._frame_width = frame_width

    def evaluate(
        self,
        track: Dict[str, Any],
        proposal_bundle: Any,
    ) -> R1Result:
        """Evaluate motion coupling for a single track.

        Parameters
        ----------
        track : dict
            Track with 'groups' containing per-frame data.
        proposal_bundle : ProposalBundle
            Video frame cache with decoded images.

        Returns
        -------
        R1Result with rejection decision.
        """
        groups = track.get("groups", [])
        if len(groups) < 2:
            return R1Result(
                rejected=False, neutral=True,
                reason="Single-frame track, insufficient for motion analysis",
            )

        # Compute background motion and track alignment for consecutive frame pairs
        frame_pairs_verified = 0
        bg_speeds: List[float] = []
        track_speeds: List[float] = []
        coupling_scores: List[float] = []

        for i in range(min(len(groups) - 1, self.K_CONSECUTIVE + 2)):
            g0 = groups[i]
            g1 = groups[i + 1]

            fi0 = g0["frame_index"]
            fi1 = g1["frame_index"]

            try:
                frame0 = proposal_bundle.frame_for_index(fi0)
                frame1 = proposal_bundle.frame_for_index(fi1)
            except (KeyError, AttributeError):
                continue

            img0 = getattr(frame0, "image", None)
            img1 = getattr(frame1, "image", None)
            if img0 is None or img1 is None:
                continue

            # Convert to grayscale for optical flow
            if len(img0.shape) == 3:
                gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray0 = img0
                gray1 = img1

            # Create mask excluding the text region
            bbox0 = g0["union_bbox_xyxy"]
            mask = np.ones_like(gray0, dtype=np.uint8) * 255
            x1 = max(0, int(round(bbox0[0])) - self.TEXT_MASK_MARGIN)
            y1 = max(0, int(round(bbox0[1])) - self.TEXT_MASK_MARGIN)
            x2 = min(gray0.shape[1], int(round(bbox0[2])) + self.TEXT_MASK_MARGIN)
            y2 = min(gray0.shape[0], int(round(bbox0[3])) + self.TEXT_MASK_MARGIN)
            mask[y1:y2, x1:x2] = 0

            # Lucas-Kanade optical flow
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    gray0, gray1, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0,
                )
            except cv2.error:
                continue

            # Masked flow → estimate global affine
            visible = mask > 0
            if visible.sum() < 100:
                continue

            y_idxs, x_idxs = np.where(visible)
            pts_src = np.column_stack([x_idxs, y_idxs]).astype(np.float32)
            flow_vectors = flow[y_idxs, x_idxs]
            pts_dst = pts_src + flow_vectors

            if len(pts_src) < 3:
                continue

            try:
                inlier_mask = None
                _, inliers = cv2.estimateAffinePartial2D(
                    pts_src, pts_dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=1000,
                    confidence=0.99,
                )
                if inliers is not None:
                    inlier_mask = inliers.ravel()
                    n_inliers = int(inlier_mask.sum())
                    if n_inliers < 10:
                        # Too few inliers → unreliable estimate
                        bg_vx, bg_vy = 0.0, 0.0
                    else:
                        # Use mean of inlier flow vectors
                        inlier_flow = flow_vectors[inlier_mask > 0]
                        bg_vx = float(np.median(inlier_flow[:, 0]))
                        bg_vy = float(np.median(inlier_flow[:, 1]))
                else:
                    bg_vx, bg_vy = 0.0, 0.0
            except cv2.error:
                bg_vx, bg_vy = 0.0, 0.0

            bg_speed = np.sqrt(bg_vx**2 + bg_vy**2)
            bg_speeds.append(bg_speed)

            # Estimate track text motion from bbox center displacement
            bbox1 = g1["union_bbox_xyxy"]
            c0_x = (bbox0[0] + bbox0[2]) / 2.0
            c0_y = (bbox0[1] + bbox0[3]) / 2.0
            c1_x = (bbox1[0] + bbox1[2]) / 2.0
            c1_y = (bbox1[1] + bbox1[3]) / 2.0
            track_speed = np.sqrt((c1_x - c0_x)**2 + (c1_y - c0_y)**2)
            track_speeds.append(track_speed)

            # Check coupling: does track move WITH background?
            if bg_speed < self.MIN_BG_SPEED_PX:
                coupling_scores.append(0.0)
                continue

            # Angle alignment
            track_angle = np.degrees(np.arctan2(c1_y - c0_y, c1_x - c0_x))
            bg_angle = np.degrees(np.arctan2(bg_vy, bg_vx))
            angle_diff = abs(track_angle - bg_angle)
            angle_diff = min(angle_diff, 360 - angle_diff)

            # Speed ratio
            if track_speed > 0 and bg_speed > 0:
                speed_ratio = track_speed / bg_speed
            else:
                speed_ratio = 0.0

            is_coupled = (
                angle_diff <= self.MATCH_ANGLE_TOLERANCE_DEG
                and self.MATCH_SPEED_RATIO[0] <= speed_ratio <= self.MATCH_SPEED_RATIO[1]
            )
            coupling_scores.append(1.0 if is_coupled else 0.0)
            frame_pairs_verified += 1

        if frame_pairs_verified == 0:
            return R1Result(
                rejected=False, neutral=True,
                reason="No frame pairs could be analyzed for motion",
            )

        avg_bg_speed = float(np.mean(bg_speeds)) if bg_speeds else 0.0
        avg_track_speed = float(np.mean(track_speeds)) if track_speeds else 0.0

        # Decision: reject only when bg is moving AND track consistently coupled
        if avg_bg_speed >= self.MIN_BG_SPEED_PX and frame_pairs_verified >= self.K_CONSECUTIVE:
            coupled_count = sum(1 for s in coupling_scores if s >= 0.5)
            if coupled_count >= self.K_CONSECUTIVE:
                return R1Result(
                    rejected=True,
                    neutral=False,
                    reason=f"Motion coupled: bg_speed={avg_bg_speed:.1f}px, "
                           f"track_speed={avg_track_speed:.1f}px, "
                           f"{coupled_count}/{frame_pairs_verified} frames coupled",
                    motion_score=coupled_count / max(frame_pairs_verified, 1),
                    bg_speed_px=round(avg_bg_speed, 2),
                    track_speed_px=round(avg_track_speed, 2),
                    frames_verified=frame_pairs_verified,
                )

        return R1Result(
            rejected=False, neutral=True,
            reason=f"No motion coupling: bg_speed={avg_bg_speed:.1f}px"
                   if avg_bg_speed < self.MIN_BG_SPEED_PX
                   else f"Track not consistently coupled to background motion",
            motion_score=0.0,
            bg_speed_px=round(avg_bg_speed, 2),
            track_speed_px=round(avg_track_speed, 2),
            frames_verified=frame_pairs_verified,
        )
