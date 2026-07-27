import cv2
import numpy as np

from src.application.services.ass_generator import ASSGenerator
from src.application.services.mask_authority import DecisionStore, MaskAuthority
from src.domain.entities.frame import Frame
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle
from src.infrastructure.providers.subtitle.hybrid_subtitle_region_detector_provider import (
    HybridSubtitleRegionDetectorProvider,
)


class Proposal:
    def detect_images(self, image_paths):
        return [
            {
                "frame_index": index,
                "timestamp_ms": index * 500,
                "candidate_id": f"F{index:05d}_C01",
                "bbox_xyxy": [300.0, 520.0, 900.0, 650.0],
                "bbox_normalized": [0.23, 0.72, 0.70, 0.90],
                "confidence": 0.9,
                "class": "text",
            }
            for index, _ in enumerate(image_paths)
        ]


class Verifier:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def classify_track(self, track, frame_bgr, candidate_ids, candidate_bboxes):
        if self.error:
            raise self.error
        return self.response


def frames(tmp_path):
    result = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        assert cv2.imwrite(str(path), np.full((720, 1280, 3), 30, dtype=np.uint8))
        result.append(Frame(timestamp=index * 0.5, image_path=str(path)))
    return result


def provider(verifier):
    return HybridSubtitleRegionDetectorProvider(
        proposal_provider=Proposal(), semantic_verifier=verifier
    )


def test_positive_verifier_decision_creates_confirmed_mask_region(tmp_path):
    result = provider(Verifier({
        "track_id": "T00001",
        "decisions": [{
            "candidate_id": "G00000_01",
            "is_subtitle": True,
            "category": "dialogue_subtitle",
            "confidence": 0.95,
            "reason_code": "spoken_dialogue",
        }],
        "model": "v21_pipeline",
        "uncertain": False,
    })).detect_regions(frames(tmp_path))

    assert len(result) == 1
    assert result[0].classification_id
    assert result[0].is_subtitle is True
    assert result[0].classifier_model == "v21_pipeline"
    assert result[0].proposal_ids == ["F00000_C01", "F00001_C01", "F00002_C01"]
    assert result[0].track_id is not None


def test_negative_verifier_decision_does_not_create_mask_region(tmp_path):
    result = provider(Verifier({
        "track_id": "T00001",
        "decisions": [{
            "candidate_id": "G00000_01",
            "is_subtitle": False,
            "category": "logo",
            "confidence": 0.99,
            "reason_code": "brand_mark",
        }],
        "model": "v21_pipeline",
        "uncertain": False,
    })).detect_regions(frames(tmp_path))

    assert result == []


def test_verifier_exception_fails_closed_without_mask_region(tmp_path, caplog):
    result = provider(Verifier(error=TimeoutError("network timeout"))).detect_regions(frames(tmp_path))

    assert result == []
    assert "LOUD-WARN" in caplog.text


def test_missing_matching_candidate_decision_fails_closed_without_mask_region(tmp_path, caplog):
    result = provider(Verifier({
        "track_id": "T00001",
        "decisions": [{
            "candidate_id": "unknown",
            "is_subtitle": True,
            "category": "dialogue_subtitle",
            "confidence": 0.95,
            "reason_code": "wrong_id",
        }],
        "model": "v21_pipeline",
        "uncertain": False,
    })).detect_regions(frames(tmp_path))

    assert result == []
    assert "LOUD-WARN" in caplog.text


def test_ass_generator_does_not_render_unconfirmed_region_as_mask():
    segment = Segment(id="s", start=0.0, end=1.0, text="source", translated_text="dịch", source="ocr")
    unconfirmed = SubtitleRegion(0.0, 1.0, 0.1, 0.7, 0.4, 0.1, 0.9)

    ass = ASSGenerator.generate(
        [segment], SubtitleStyle(), play_res_x=720, play_res_y=1280, mask_regions=[unconfirmed]
    )

    assert not [
        line for line in ass.splitlines()
        if line.startswith("Dialogue: 0,") and "m 0 0" in line
    ]


def test_ass_generator_renders_confirmed_region_with_classification_id():
    segment = Segment(id="s", start=0.0, end=1.0, text="source", translated_text="dịch", source="ocr")
    store = DecisionStore("test-run")
    confirmed = MaskAuthority(store).decide(
        {
            "start_time": 0.0, "end_time": 1.0, "x": 0.1, "y": 0.7,
            "width": 0.4, "height": 0.1, "confidence": 0.9,
            "track_id": "T00001", "proposal_ids": ["F00000_C01"],
        },
        {"is_subtitle": True, "category": "dialogue_subtitle"},
        model="v21_pipeline",
    )
    assert confirmed is not None

    ass = ASSGenerator.generate(
        [segment], SubtitleStyle(), play_res_x=720, play_res_y=1280,
        mask_regions=[confirmed], decision_store=store,
    )

    mask_lines = [
        line for line in ass.splitlines()
        if line.startswith("Dialogue: 0,") and "m 0 0" in line
    ]
    assert len(mask_lines) == 1
    assert f"provenance={confirmed.classification_id}" in mask_lines[0]


def test_region_provider_uses_real_frame_indices_for_sparse_bundle(tmp_path):
    from src.application.dto.proposal_bundle import ProposalBundle

    bundle = ProposalBundle("video.mp4", fps=30.0, stride=2, video_width=1280, video_height=720)
    for frame_index in (0, 5, 10):
        bundle.add_frame(
            frame_index=frame_index,
            timestamp_sec=frame_index / 30.0,
            image=np.full((720, 1280, 3), 30, dtype=np.uint8),
            image_path=None,
            detections=[{
                "frame_index": frame_index,
                "timestamp_ms": frame_index / 30.0 * 1000,
                "candidate_id": f"F{frame_index:05d}_C01",
                "bbox_xyxy": [300.0, 520.0, 900.0, 650.0],
                "confidence": 0.9,
            }],
        )
    detector = provider(Verifier({
        "track_id": "T00001",
        "decisions": [{
            "candidate_id": "G00000_01", "is_subtitle": True,
            "category": "dialogue_subtitle", "confidence": 0.95, "reason_code": "spoken",
        }], "model": "v21_pipeline", "uncertain": False,
    }))
    detector.proposal_bundle = bundle

    result = detector.detect_regions(frames(tmp_path))

    assert result
    assert result[0].start_time == 0.0
    assert result[0].end_time > 4 / 30.0


def test_provenance_not_rendered_on_screen():
    segment = Segment(
        id="s", start=0.0, end=1.0, text="source", translated_text="dich", source="ocr"
    )
    store = DecisionStore("test-run")
    confirmed = MaskAuthority(store).decide(
        {
            "start_time": 0.0, "end_time": 1.0, "x": 0.1, "y": 0.7,
            "width": 0.4, "height": 0.1, "confidence": 0.9,
            "track_id": "T00001", "proposal_ids": ["F00000_C01"],
        },
        {"is_subtitle": True, "category": "dialogue_subtitle"},
        model="v21_pipeline",
    )
    assert confirmed is not None

    ass = ASSGenerator.generate(
        [segment], SubtitleStyle(), play_res_x=720, play_res_y=1280,
        mask_regions=[confirmed], decision_store=store,
    )

    mask_lines = [
        line for line in ass.splitlines()
        if line.startswith("Dialogue: 0,") and "m 0 0" in line
    ]
    assert len(mask_lines) == 1
    assert f"provenance={confirmed.classification_id}" in mask_lines[0]
    # Split on the drawing close tag to check what comes after
    split_text = r'{\p0}'
    parts = mask_lines[0].split(split_text)
    assert len(parts) > 1, f"Could not split on {split_text!r} in: {mask_lines[0]}"
    after_p0 = parts[1].strip()
    if after_p0:
        # Anything after {\\p0} must be inside {} brackets (ASS comment block)
        assert after_p0.startswith("{") and after_p0.endswith("}"),             f"Found visible text after {{@}}p0: {after_p0!r}"


def test_heuristic_negative_is_rejected_immediately_without_gemini(tmp_path):
    class TopProposal:
        def detect_images(self, image_paths):
            return [
                {
                    "frame_index": index,
                    "timestamp_ms": index * 500,
                    "candidate_id": f"F{index:05d}_C01",
                    "bbox_xyxy": [300.0, 50.0, 900.0, 100.0], # Top of frame, y2=100.0
                    "bbox_normalized": [0.23, 0.07, 0.70, 0.14],
                    "confidence": 0.9,
                    "class": "text",
                }
                for index, _ in enumerate(image_paths)
            ]

    class MockSpyVerifier:
        def __init__(self):
            self.called = False
        def classify_track(self, track, frame_bgr, candidate_ids, candidate_bboxes):
            self.called = True
            return {
                "track_id": "T00001",
                "decisions": [{"candidate_id": candidate_ids[0], "is_subtitle": True, "category": "dialogue_subtitle", "confidence": 0.95}],
                "model": "v21_pipeline",
                "uncertain": False
            }

    spy = MockSpyVerifier()
    det = HybridSubtitleRegionDetectorProvider(
        proposal_provider=TopProposal(), semantic_verifier=spy
    )
    result = det.detect_regions(frames(tmp_path))

    assert result == []
    assert spy.called is False
