from __future__ import annotations
import cv2, json, logging, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .config import BenchmarkConfig
from .video_utils import (
    VideoMetadata, probe_video, draw_bboxes, COLOR_MAP,
    safe_json_dump, jsonl_dump,
)
from .core import (
    box_iou,
    group_frame_candidates, build_tracks, representative_group_ids,
    validate_gemini_decision, proposal_metrics,
    ALLOWED_CATEGORIES,
)
from .gemini_verifier import LiveGeminiTrackVerifier
from .yolo_proposal import YoloTextProposalProvider
from .safe_cover import SafeCoverRegionBuilder
from .gt_metrics import evaluate_tracks_vs_gt, compute_timing_mae

logger = logging.getLogger(__name__)


def classify_track_by_position(median_bbox_xyxy, frame_height, rep_ids,
                               track_id, bottom_ratio_threshold=0.65):
    """Fallback heuristic thuan tuy theo vi tri. Toa do dau vao la PIXEL.
    Khong duoc dung lam ground truth."""
    if not median_bbox_xyxy or not frame_height or frame_height <= 0:
        return {"track_id": track_id, "decisions": [],
                "uncertain": True, "error": "no_median_bbox_or_frame_height"}
    is_sub = (median_bbox_xyxy[3] / frame_height) > bottom_ratio_threshold
    decisions = [{"candidate_id": cid, "is_subtitle": is_sub,
                  "category": "dialogue_subtitle" if is_sub else "scene_text",
                  "confidence": 0.75, "transcription": None,
                  "reason_code": "position_bottom_heuristic" if is_sub
                                 else "position_not_bottom"}
                 for cid in rep_ids]
    return {"track_id": track_id, "decisions": decisions,
            "uncertain": False, "source": "fallback_heuristic"}


class HybridSubtitleBenchmarkRunner:
    def __init__(self, config: BenchmarkConfig,
                 output_dir: str,
                 gemini_runtime_client=None):
        self.cfg = config
        self.out = Path(output_dir)
        self._gemini = gemini_runtime_client
        self._yolo = YoloTextProposalProvider(
            weights_path=config.yolo_weights,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            imgsz=config.yolo_imgsz)
        self._safe_cover = SafeCoverRegionBuilder(
            glyph_margin=config.glyph_margin,
            outline_margin=config.outline_margin,
            shadow_margin=config.shadow_margin,
            motion_margin=config.motion_margin)
        if self._gemini and config.live_gemini:
            self._verifier = LiveGeminiTrackVerifier(gemini_runtime_client, config.gemini_model)
        else:
            self._verifier = None

    def run_video(self, video_path: str, sample_id: str,
                  fps_override: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        cache_dir = str(self.out / "cache")
        self.out.mkdir(parents=True, exist_ok=True)

        # 1. Probe & transcode if needed
        meta = probe_video(video_path, cache_dir=cache_dir)
        video_for_detect = meta.transcoded_path or video_path
        video_fps = fps_override or meta.fps or 30.0
        W, H = meta.width, meta.height

        # 2. YOLO proposal
        logger.info("[%s] YOLO proposal on %s (codec=%s)",
                    sample_id, video_path, meta.transcoded_codec or meta.codec_name)
        raw_dets = self._yolo.detect(video_for_detect)
        raw_dets = [d for d in raw_dets if d["frame_index"] < meta.frame_count]
        jsonl_dump(raw_dets, str(self.out / f"{sample_id}_raw_yolo_detections.jsonl"))

        # 3. Group per frame using core.group_frame_candidates
        frame_groups: Dict[int, List[Dict]] = {}
        by_frame: Dict[int, List[Dict]] = {}
        for d in raw_dets:
            by_frame.setdefault(d["frame_index"], []).append(d)
        all_grouped: List[Dict] = []
        for fi in sorted(by_frame):
            groups = group_frame_candidates(by_frame[fi], self.cfg.horizontal_gap_ratio)
            frame_groups[fi] = groups
            for g in groups:
                g["_frame_index"] = fi
            all_grouped.append({"frame": fi, "groups": groups})
        jsonl_dump(all_grouped, str(self.out / f"{sample_id}_grouped_candidates.jsonl"))

        # 4. Build temporal tracks using core.build_tracks
        flat_groups = [g for groups in frame_groups.values() for g in groups]
        tracks = build_tracks(flat_groups,
                              iou_threshold=self.cfg.track_iou_threshold,
                              max_gap_frames=self.cfg.track_max_gap_frames)
        # Annotate each track with safe-cover geometry
        for track in tracks:
            bboxes = [g["union_bbox_xyxy"] for g in track["groups"]]
            safe_bbox = self._safe_cover.build_safe_covers(
                bboxes, strategy=self.cfg.safe_cover_strategy)
            track["safe_cover_bbox"] = safe_bbox
            track["safe_cover_strategy"] = self.cfg.safe_cover_strategy
        safe_json_dump(tracks, str(self.out / f"{sample_id}_text_tracks.json"))

        # 5. Gemini semantic verification (per track)
        all_decisions, final_tracks = [], []
        cap = cv2.VideoCapture(str(video_for_detect))
        for track in tracks:
            bboxes = [g["union_bbox_xyxy"] for g in track["groups"]]
            rep_ids = representative_group_ids(track, self.cfg.max_rep_frames)
            rep_frames = self._select_rep_frames(track, cap)

            if self._verifier and rep_frames:
                result = self._verifier.classify(track, rep_frames, rep_ids, cap)
            else:
                result = self._rule_classify(track, rep_ids, H)

            result["track_id"] = track["track_id"]
            all_decisions.append(result)

            decisions = result.get("decisions", [])
            if decisions and any(d.get("is_subtitle") for d in decisions):
                final_tracks.append(track)
                track["verdict_color"] = "green"
                track["review_flag"] = self._safe_cover.check_review_flag(bboxes, "green", False)
            else:
                track["verdict_color"] = "red"
                track["review_flag"] = False
        cap.release()

        jsonl_dump(all_decisions, str(self.out / f"{sample_id}_gemini_decisions.jsonl"))
        safe_json_dump(final_tracks, str(self.out / f"{sample_id}_final_subtitle_tracks.json"))

        # 6. Annotate videos
        self._annotate_videos(video_for_detect, raw_dets, frame_groups,
                               tracks, final_tracks, sample_id, W, H, video_fps)

        # 7. Compute pipeline metrics
        metrics = self._compute_pipeline_metrics(
            raw_dets, frame_groups, tracks, final_tracks, meta, sample_id)
        metrics["gemini_calls"] = len(all_decisions)
        metrics["duration_seconds"] = round(time.time() - t0, 1)

        # 8. Load manual GT if available
        gt_path = self.out / f"{sample_id}_manual_gt.json"
        if gt_path.exists():
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                gt_m = proposal_metrics(final_tracks, gt_data, iou_threshold=0.5)
                if gt_m:
                    metrics["ground_truth_metrics"] = gt_m
                    metrics["ground_truth_available"] = True
            except Exception as exc:
                logger.error("GT metrics failed for %s: %s", sample_id, exc)

        safe_json_dump(metrics, str(self.out / f"{sample_id}_metrics.json"))
        logger.info("[%s] Done in %.1fs | tracks=%d final=%d",
                    sample_id, time.time()-t0, len(tracks), len(final_tracks))
        return {"sample_id": sample_id, "metrics": metrics}

    def _select_rep_frames(self, track: Dict, cap) -> List[Tuple[int, Any, str]]:
        groups = track["groups"]
        n = len(groups)
        if n == 0:
            return []
        result = []
        positions = [0, n // 2, n - 1]
        if n >= 1:
            result.append((groups[0]["frame_index"], groups[0]["union_bbox_xyxy"], "first"))
        if n >= 2:
            result.append((groups[-1]["frame_index"], groups[-1]["union_bbox_xyxy"], "last"))
        if n >= 3:
            result.append((groups[n // 2]["frame_index"], groups[n // 2]["union_bbox_xyxy"], "middle"))
        return [(fi, bbox, role) for fi, bbox, role in result[:self.cfg.max_rep_frames]]

    def _rule_classify(self, track: Dict, rep_ids: List[str],
                       frame_height: float) -> Dict:
        """Fallback: position-only heuristic. Must not be used as ground truth."""
        median = (track.get("median_bbox_xyxy")
                  or track.get("representative_bbox_xyxy") or [])
        return classify_track_by_position(median, frame_height, rep_ids,
                                          track["track_id"])

    def _annotate_videos(self, video_path: str,
                          raw_dets: List[Dict],
                          frame_groups: Dict[int, List[Dict]],
                          tracks: List[Dict],
                          final_tracks: List[Dict],
                          sample_id: str, W: int, H: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for name, det_gen in [
            ("raw_yolo", lambda fi: [(d["bbox_xyxy"], "yellow") for d in raw_dets if d["frame_index"] == fi]),
            ("grouped_candidates", lambda fi: [(g["union_bbox_xyxy"], "yellow") for g in frame_groups.get(fi, [])]),
            ("gemini_filtered", self._gemini_layer(tracks, raw_dets)),
            ("final_safe_cover", self._cover_layer(final_tracks)),
        ]:
            out_path = str(self.out / f"{sample_id}_{name}.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            cap = cv2.VideoCapture(str(video_path))
            fi = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                boxes = det_gen(fi)
                vis = [{"bbox_xyxy": b, "color_key": c, "thickness": 2} for b, c in boxes]
                annotated = draw_bboxes(frame, vis, COLOR_MAP)
                writer.write(annotated)
                fi += 1
            cap.release()
            writer.release()
            logger.info("  Wrote %s", out_path)

    def _gemini_layer(self, tracks, raw_dets):
        return lambda fi: [(d["bbox_xyxy"], "yellow") for d in raw_dets if d["frame_index"] == fi] + [
            (t.get("median_bbox_xyxy") or t.get("representative_bbox_xyxy"), "green")
            for t in tracks if t.get("verdict_color") == "green"
            and (t["start_frame"] == fi or t["end_frame"] == fi)]

    def _cover_layer(self, final_tracks):
        def inner(fi):
            result = []
            for t in final_tracks:
                cover = t.get("safe_cover_bbox") or t.get("median_bbox_xyxy") or t.get("representative_bbox_xyxy")
                color = "green"
                for check_fi in range(t["start_frame"], t["end_frame"] + 1,
                                       max(1, t["frame_count"] // 5)):
                    if check_fi == fi:
                        result.append((cover, color))
            return result
        return inner

    def _compute_pipeline_metrics(self, raw_dets, frame_groups, tracks,
                                   final_tracks, meta, sample_id):
        total_frames = meta.frame_count
        det_frames = len(frame_groups)
        return {
            "sample_id": sample_id,
            "video_codec": meta.codec_name,
            "transcoded_codec": meta.transcoded_codec,
            "width": meta.width,
            "height": meta.height,
            "fps": round(meta.fps, 3),
            "total_frames": total_frames,
            "duration_seconds": round(meta.duration_seconds, 2),
            "frames_with_yolo_detection": det_frames,
            "detection_rate": round(det_frames / max(total_frames, 1), 4),
            "total_raw_yolo_boxes": len(raw_dets),
            "avg_boxes_per_detected_frame": round(len(raw_dets) / max(det_frames, 1), 2),
            "total_tracks": len(tracks),
            "final_accepted_tracks": len(final_tracks),
            "track_fragmentation": round(
                (len(raw_dets) / max(len(final_tracks), 1)) - 1, 2),
            "safe_cover_strategy": self.cfg.safe_cover_strategy,
            "yolo_conf": self.cfg.yolo_conf,
            "yolo_imgsz": self.cfg.yolo_imgsz,
            "gemini_live": self.cfg.live_gemini,
            "ground_truth_available": False,
            "note": "Recall/Precision/F1 require reviewed manual GT annotations "
                    "in sample_id_manual_gt.json. No reviewed GT in this run.",
        }
