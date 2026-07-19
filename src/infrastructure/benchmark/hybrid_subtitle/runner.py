from __future__ import annotations
import cv2, json, logging, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
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
from .yolo_proposal import YoloTextProposalProvider
from .safe_cover import SafeCoverRegionBuilder
from .gt_metrics import evaluate_tracks_vs_gt, compute_timing_mae

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    yolo_weights: str = "best.pt"
    yolo_conf: float = 0.05
    yolo_iou: float = 0.70
    yolo_imgsz: int = 1024
    horizontal_gap_ratio: float = 2.0
    track_iou_threshold: float = 0.25
    track_max_gap_frames: int = 5
    glyph_margin: float = 12.0
    outline_margin: float = 4.0
    shadow_margin: float = 2.0
    motion_margin: float = 5.0
    safe_cover_strategy: str = "percentile"
    gemini_model: str = "gemini-3.1-flash-lite"
    live_gemini: bool = False
    max_rep_frames: int = 5


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
            self._verifier = _LiveGeminiVerifier(gemini_runtime_client, config.gemini_model)
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


class _LiveGeminiVerifier:
    """Wraps Gemini client for benchmark; reuses existing runtime infrastructure."""
    def __init__(self, runtime_client, model: str):
        self._client = runtime_client
        self._model = model

    def classify(self, track: Dict, rep_frames: List[Tuple[int, Any, str]],
                 candidate_ids: List[str], cap) -> Dict:
        if not rep_frames:
            return {"track_id": track["track_id"], "decisions": [],
                    "uncertain": True}
        import base64, json as _json
        content_parts = [{"text": self._build_prompt(candidate_ids, track)}]
        for fi, bbox, role in rep_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            # Draw candidate ID label near the bbox
            label = f"{role}:{candidate_ids[rep_frames.index((fi, bbox, role))] if (fi, bbox, role) in rep_frames else ''}"
            annotated = draw_bboxes(frame, [{"bbox_xyxy": bbox, "color_key": "yellow",
                                           "label": f"{role[0].upper()}", "thickness": 2}], COLOR_MAP)
            _, png_buf = cv2.imencode(".png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            content_parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(png_buf.tobytes()).decode("ascii")
                }
            })
        try:
            from src.infrastructure.gemini.contracts import GeminiRequest
            req = GeminiRequest(
                contents=content_parts,
                model=self._model,
                generation_config={"temperature": 0.1},
                system_instruction=(
                    "Geometry is provided by a detector. Do NOT modify or re-evaluate "
                    "coordinates. Only classify candidate IDs. Return valid JSON. "
                    "Allowed categories: dialogue_subtitle, narration_subtitle, "
                    "sound_caption, karaoke_or_lyric_subtitle, scene_text, watermark, "
                    "logo, ui_text, social_overlay, title_card, credits, decorative_text, unknown."
                ),
                request_id=track["track_id"],
                source_kind="benchmark_subtitle_verifier",
                api_surface="benchmark",
            )
            resp = self._client.generate_content(req, use_cache=True)
            text = resp.text if hasattr(resp, "text") else ""
            parsed = _json.loads(text)
            validated = validate_gemini_decision(parsed, track["track_id"], set(candidate_ids))
            return validated
        except Exception as exc:
            return {"track_id": track["track_id"], "decisions": [],
                    "uncertain": True, "error": str(exc)[:200]}

    def _build_prompt(self, candidate_ids: List[str], track: Dict) -> str:
        import json as _json
        return (
            "Classify each candidate ID as subtitle or non-subtitle.\\n"
            "Candidate IDs in this track (" + track["track_id"] + "):\\n"
            + _json.dumps(candidate_ids) + "\\n\\n"
            "Track frames: " + str(len(track["groups"])) + " frames "
            "from frame " + str(track.get("start_frame")) + " to " + str(track.get("end_frame")) + ".\\n\\n"
            "Return: {\\\"track_id\\\": \\\"" + track["track_id"] + "\\\", "
            "\\\"decisions\\\": [{\\\"candidate_id\\\": \\\"<id>\\\", "
            "\\\"is_subtitle\\\": true/false, \\\"category\\\": \\\"<category>\\\", "
            "\\\"confidence\\\": 0.0-1.0, \\\"transcription\\\": null/str, "
            "\\\"reason_code\\\": \\\"<short>\\\"}], \\\"uncertain\\\": true/false}"
        )


# raw_dets passed via closure in _annotate_videos
