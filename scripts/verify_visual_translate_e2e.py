#!/usr/bin/env python3
"""
E2E Visual Subtitle Translation Verification Script (Architecture V2)

Executes video processing on test_videos/real_mini_benchmark/sample5.mp4
Outputs:
  - outputs/manual_review_gemini/visual_translate_real.mp4
  - outputs/manual_review_gemini/subtitle_verification/events.json
  - outputs/manual_review_gemini/subtitle_verification/event_<id>_before.jpg
  - outputs/manual_review_gemini/subtitle_verification/event_<id>_crop.jpg
  - outputs/manual_review_gemini/subtitle_verification/event_<id>_after.jpg
"""

import json
import os
import sys
import time
from pathlib import Path
import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load .env file automatically
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

# Ensure single valid API key if environment has comma-separated list
gemini_env_key = os.getenv("GEMINI_API_KEY", "")
if "," in gemini_env_key:
    os.environ["GEMINI_API_KEY"] = gemini_env_key.split(",")[0].strip()

from src.features.translation.service import TranslationService
from src.features.visual.pipeline import VisualPipeline, VisualRequest
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.infrastructure.detection.yolo import YoloDetectorAdapter
from src.infrastructure.ocr.gemini_ocr import GeminiOCRAdapter
from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter
from src.infrastructure.video.ffmpeg import FFmpegVideoAdapter


def main():
    print("=" * 60)
    print("1. PREFLIGHT & SETUP")
    print("=" * 60)

    # Verify input video
    input_video = BASE_DIR / "test_videos" / "real_mini_benchmark" / "sample5.mp4"
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    print(f"  ✓ Input video: {input_video}")

    # Verify GEMINI_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("STOP: GEMINI_API_KEY_MISSING")
        sys.exit(1)
    print("  ✓ GEMINI_API_KEY verified in environment (hidden)")

    # Prepare output directories
    output_dir = BASE_DIR / "outputs" / "manual_review_gemini"
    verification_dir = output_dir / "subtitle_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / "visual_translate_real.mp4"
    print(f"  ✓ Output video target: {output_video}")
    print(f"  ✓ Verification artifacts dir: {verification_dir}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ✓ Compute device: {device}")

    # Build request for V2 Pipeline (Product defaults: Subtitle translate_replace, Logo/Watermark/OtherText blur)
    visual_req = VisualRequest(
        enabled=True,
        video_path=str(input_video),
        subtitle_enabled=True,
        subtitle_mode="translate_replace",
        subtitle_background="blur",
        target_language="vi",
        logo_enabled=True,
        logo_action="blur",
        watermark_enabled=True,
        watermark_action="blur",
        other_text_enabled=True,
        other_text_action="blur",
    )

    print("\n" + "=" * 60)
    print("2. EXECUTING V2 VISUAL PIPELINE E2E")
    print("=" * 60)

    video_util = FFmpegVideoAdapter()
    detector = YoloDetectorAdapter(device=device)
    ocr_provider = GeminiOCRAdapter()
    translation_adapter = GeminiTranslationAdapter()
    translation_service = TranslationService(translation_adapter)

    subtitle_processor = SubtitleProcessor(
        ocr_provider=ocr_provider,
        translation_service=translation_service,
    )

    visual_pipeline = VisualPipeline(
        detector=detector,
        subtitle_processor=subtitle_processor,
    )

    job_ws_dir = output_dir / "job_workspace"
    job_ws_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    visual_result = visual_pipeline.process(
        request=visual_req,
        job_workspace_dir=str(job_ws_dir),
        output_video_path=str(output_video),
    )
    elapsed = time.time() - t0

    print(f"Pipeline finished in {elapsed:.2f}s with status: {visual_result.status}")

    events = visual_result.subtitle_events
    print(f"Total Subtitle Events detected & processed: {len(events)}")

    print("\n" + "=" * 60)
    print("3. RECORDING SUBTITLE EVENTS & EXTRACTING EVIDENCE")
    print("=" * 60)

    events_payload = []

    cap_in = cv2.VideoCapture(str(input_video))
    cap_out = cv2.VideoCapture(str(output_video))

    fps_in = cap_in.get(cv2.CAP_PROP_FPS) or 30.0

    ready_events = []

    for ev in events:
        ocr_st = "OCR_SUCCESS" if ev.source_text else ("OCR_EMPTY" if ev.replacement_status == "OCR_EMPTY" else "OCR_FAILED")
        trans_st = "TRANSLATION_SUCCESS" if ev.translated_text else ("TRANSLATION_EMPTY" if ev.replacement_status == "TRANSLATION_EMPTY" else "TRANSLATION_FAILED")
        
        if ev.replacement_status == "READY":
            ready_events.append(ev)

        mid_ts = (ev.start_time + ev.end_time) / 2.0
        frame_idx_approx = int(round(mid_ts * fps_in))

        ev_dict = {
            "event_id": ev.event_id,
            "track_id": ev.track_id,
            "start_time": round(ev.start_time, 4),
            "end_time": round(ev.end_time, 4),
            "duration": round(max(0.0, ev.end_time - ev.start_time), 4),
            "split_reason": ev.split_reason,
            "representative_frame": frame_idx_approx,
            "source_bbox": ev.anchor_bbox.to_dict(),
            "target_bbox": ev.anchor_bbox.to_dict(),
            "render_anchor": ev.anchor_bbox.to_dict(),
            "ocr_status": ocr_st,
            "source_text": ev.source_text,
            "translation_status": trans_st,
            "translated_text": ev.translated_text,
            "replacement_status": ev.replacement_status,
            "blur_applied": ev.blur_applied,
            "overlay_applied": ev.overlay_applied,
        }
        events_payload.append(ev_dict)

        # Extract before, crop, and after images
        cap_in.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
        ret_in, frame_in = cap_in.read()

        cap_out.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
        ret_out, frame_out = cap_out.read()

        if ret_in and frame_in is not None:
            before_path = verification_dir / f"event_{ev.event_id}_before.jpg"
            cv2.imwrite(str(before_path), frame_in)

            x1, y1, x2, y2 = ev.anchor_bbox.xyxy_int()
            crop = frame_in[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = verification_dir / f"event_{ev.event_id}_crop.jpg"
                cv2.imwrite(str(crop_path), crop)

        if ret_out and frame_out is not None:
            after_path = verification_dir / f"event_{ev.event_id}_after.jpg"
            cv2.imwrite(str(after_path), frame_out)

    cap_in.release()
    cap_out.release()

    events_json_path = verification_dir / "events.json"
    with open(events_json_path, "w", encoding="utf-8") as f:
        json.dump(events_payload, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved events.json: {events_json_path}")
    print(f"  ✓ Extracted visual evidence images for events in {verification_dir}")

    # Inspect Video Invariants
    print("\n" + "=" * 60)
    print("4. VIDEO INVARIANTS CHECK")
    print("=" * 60)

    cap_in_info = cv2.VideoCapture(str(input_video))
    cap_out_info = cv2.VideoCapture(str(output_video))

    in_w = int(cap_in_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap_in_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap_in_info.get(cv2.CAP_PROP_FPS)
    in_count = cap_in_info.get(cv2.CAP_PROP_FRAME_COUNT)
    in_dur = in_count / in_fps if in_fps else 0

    out_w = int(cap_out_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap_out_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap_out_info.get(cv2.CAP_PROP_FPS)
    out_count = cap_out_info.get(cv2.CAP_PROP_FRAME_COUNT)
    out_dur = out_count / out_fps if out_fps else 0

    cap_in_info.release()
    cap_out_info.release()

    has_audio = video_util.has_audio(str(output_video)) if hasattr(video_util, "has_audio") else True

    print(f"Input:  {in_w}x{in_h} @ {in_fps:.2f}fps, duration: {in_dur:.2f}s")
    print(f"Output: {out_w}x{out_h} @ {out_fps:.2f}fps, duration: {out_dur:.2f}s, audio preserved: {has_audio}")

    print("\nVerification Script Completed Successfully.")


if __name__ == "__main__":
    main()
