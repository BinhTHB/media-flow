from pathlib import Path

from src.application.services.artifact_cache_validator import ArtifactCacheValidator
from src.domain.entities.stage_control_config import StageControlConfig


def test_cache_signature_invalidates_input_or_device_change(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    validator = ArtifactCacheValidator()
    cpu = validator.signature(
        str(video), 1.0, "paddle", "3", "ch", "cpu", 10.0, "1440x2560"
    )
    gpu = validator.signature(
        str(video), 1.0, "paddle", "3", "ch", "gpu", 10.0, "1440x2560"
    )

    assert validator.is_valid(cpu, dict(cpu))
    assert not validator.is_valid(cpu, gpu)
    video.write_bytes(b"changed")
    changed = validator.signature(
        str(video), 1.0, "paddle", "3", "ch", "cpu", 10.0, "1440x2560"
    )
    assert not validator.is_valid(cpu, changed)


def test_rerender_controls_disable_expensive_stages():
    controls = StageControlConfig(
        run_asr=False,
        run_ocr=False,
        run_subtitle_region_detection=False,
        run_translation=False,
        run_normalization=False,
        run_segmentation=False,
    )

    assert not controls.run_asr
    assert not controls.run_ocr
    assert controls.run_render
    assert controls.run_tts
