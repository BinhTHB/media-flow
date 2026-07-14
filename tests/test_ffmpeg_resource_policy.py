from pathlib import Path

from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.infrastructure.runtime.ffmpeg_capability_detector import FFmpegCapabilityReport
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args
from src.infrastructure.runtime.resource_usage_report import write_resource_usage_report


NVENC = FFmpegCapabilityReport(True, True, True, ("h264_nvenc",), ("scale_cuda", "cuda"), "nvenc")
CPU = FFmpegCapabilityReport(True, False, False, (), (), "cpu")


def test_auto_selects_nvenc_and_preserves_audio():
    plan = build_ffmpeg_encode_args("input.mp4", "output.mp4", FFmpegConfig(), False, NVENC)

    assert plan.encoder_used == "h264_nvenc"
    assert ("-c:a", "copy") == tuple(plan.args[plan.args.index("-c:a"):plan.args.index("-c:a") + 2])
    assert "-cq" in plan.args


def test_encoder_falls_back_to_libx264_without_nvenc():
    plan = build_ffmpeg_encode_args("input.mp4", "output.mp4", FFmpegConfig(), False, CPU)

    assert plan.encoder_used == "libx264"
    assert "-crf" in plan.args


def test_gpu_decode_is_not_forced_for_cpu_filters():
    plan = build_ffmpeg_encode_args(
        "input.mp4", "output.mp4",
        FFmpegConfig(decoder="cuda", use_gpu_decode=True),
        True, NVENC,
    )

    assert plan.gpu_decode_used is False
    assert "-hwaccel" not in plan.args


def test_resource_profile_gates_expensive_operations():
    assert ResourcePolicy("low").watermark_mode == "gaussian"
    assert ResourcePolicy("balanced").watermark_mode == "blur_fill_feathered"
    assert not ResourcePolicy("balanced").can_inpaint_watermark(0.01)
    assert ResourcePolicy("quality", allow_inpaint_light=True).can_inpaint_watermark(0.049)
    assert not ResourcePolicy("quality", allow_inpaint_light=True).can_inpaint_watermark(0.05)
    assert ResourcePolicy("balanced").expensive_validation_enabled is False


def test_resource_usage_report_is_generated(tmp_path: Path):
    policy = ResourcePolicy()
    plan = build_ffmpeg_encode_args("input.mp4", "output.mp4", FFmpegConfig(), True, NVENC)

    path = write_resource_usage_report(
        tmp_path, policy, NVENC, plan,
        expensive_steps_skipped=["inpaint_light", "watermark_auto_blur_safety_gate"],
    )

    assert path == tmp_path / "debug" / "resource_usage_report.json"
    assert '"resource_profile": "balanced"' in path.read_text(encoding="utf-8")
    assert '"ffmpeg_encoder_used": "h264_nvenc"' in path.read_text(encoding="utf-8")
    assert "watermark_auto_blur_safety_gate" in path.read_text(encoding="utf-8")


from src.application.services.watermark_detection_service import WatermarkDetectionService
import json

def test_person_region_candidate_rejected_in_balanced_profile():
    candidate = {
        "candidate_id": "watermark_001",
        "bbox": {"left": 200, "top": 150, "right": 300, "bottom": 250},
        "classification": "logo",
        "position_stability_score": 0.70,
        "size_stability_score": 0.80,
        "corner_or_edge_score": 0.50,
        "watermark_score": 0.80,
        "frame_coverage_ratio": 0.70,
    }
    res = WatermarkDetectionService.apply_auto_blur_safety(candidate, 1000, 600, 1, "balanced")
    assert not res["balanced_auto_blur_allowed"]
    assert res["person_overlap_risk"] > 0.25
    assert res["classification"] == "uncertain"

def test_clothing_object_attached_logo_not_treated_as_watermark():
    candidate = {
        "candidate_id": "watermark_001",
        "bbox": {"left": 10, "top": 10, "right": 80, "bottom": 80},
        "classification": "logo",
        "position_stability_score": 0.75,
        "size_stability_score": 0.50,
        "corner_or_edge_score": 0.90,
        "watermark_score": 0.75,
        "frame_coverage_ratio": 0.70,
    }
    res = WatermarkDetectionService.apply_auto_blur_safety(candidate, 1000, 600, 1, "balanced")
    assert not res["balanced_auto_blur_allowed"]
    assert res["object_attachment_risk"] > 0.30

def test_stable_corner_overlay_passes():
    candidate = {
        "candidate_id": "watermark_001",
        "bbox": {"left": 10, "top": 10, "right": 80, "bottom": 80},
        "classification": "logo",
        "position_stability_score": 0.98,
        "size_stability_score": 0.98,
        "corner_or_edge_score": 0.95,
        "watermark_score": 0.90,
        "frame_coverage_ratio": 1.0,
    }
    res = WatermarkDetectionService.apply_auto_blur_safety(candidate, 1000, 600, 1, "balanced")
    assert res["balanced_auto_blur_allowed"]
    assert res["person_overlap_risk"] <= 0.25
    assert res["object_attachment_risk"] <= 0.30

def test_moving_object_attached_candidate_fails_overlay_independence():
    candidate = {
        "candidate_id": "watermark_001",
        "bbox": {"left": 100, "top": 100, "right": 200, "bottom": 200},
        "classification": "logo",
        "position_stability_score": 0.60,
        "size_stability_score": 0.60,
        "corner_or_edge_score": 0.70,
        "watermark_score": 0.70,
        "frame_coverage_ratio": 0.80,
    }
    res = WatermarkDetectionService.apply_auto_blur_safety(candidate, 1000, 600, 1, "balanced")
    assert res["overlay_independence_score"] == 0.60
    assert not res["balanced_auto_blur_allowed"]

def test_final_candidates_greater_than_one_triggers_review_in_balanced():
    candidate = {
        "candidate_id": "watermark_001",
        "bbox": {"left": 10, "top": 10, "right": 80, "bottom": 80},
        "classification": "logo",
        "position_stability_score": 0.98,
        "size_stability_score": 0.98,
        "corner_or_edge_score": 0.95,
        "watermark_score": 0.90,
        "frame_coverage_ratio": 1.0,
    }
    res = WatermarkDetectionService.apply_auto_blur_safety(candidate, 1000, 600, 2, "balanced")
    assert not res["balanced_auto_blur_allowed"]
    assert res["rejection_reason"] == "multiple_distinct_candidates"

def test_person_overlap_risk_prevents_blur():
    candidate = {
        "candidate_id": "watermark_001",
        "bbox": {"left": 200, "top": 200, "right": 300, "bottom": 300},
        "classification": "logo",
        "position_stability_score": 0.80,
        "size_stability_score": 0.90,
        "corner_or_edge_score": 0.60,
        "watermark_score": 0.80,
        "frame_coverage_ratio": 0.80,
    }
    res = WatermarkDetectionService.apply_auto_blur_safety(candidate, 1000, 600, 1, "balanced")
    assert not res["balanced_auto_blur_allowed"]
    assert res["rejection_reason"] == "person_overlap_risk"

def test_balanced_profile_does_not_enable_heavy_detector_by_default():
    policy = ResourcePolicy("balanced")
    assert not policy.expensive_validation_enabled

def test_resource_usage_report_records_skipped_auto_blur_reason(tmp_path):
    policy = ResourcePolicy("balanced")
    plan = build_ffmpeg_encode_args("input.mp4", "output.mp4", FFmpegConfig(), True, NVENC)
    path = write_resource_usage_report(
        tmp_path, policy, NVENC, plan, expensive_steps_skipped=["inpaint_light"]
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["expensive_steps_skipped"] == ["inpaint_light"]
    assert report["reason_for_skipping_heavy_algorithms"] is not None
