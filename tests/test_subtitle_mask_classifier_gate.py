import cv2
import numpy as np

from src.application.services.ass_generator import ASSGenerator
from src.application.services.mask_authority import DecisionStore
from src.application.dto.proposal_bundle import ProposalBundle
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


class BatchClassifier:
    def __init__(self, is_subtitle=True, error=None):
        self.is_subtitle = is_subtitle
        self.error = error
        self.called = False

    def classify_batch(self, tracks, _bundle):
        self.called = True
        if self.error:
            raise self.error
        return {
            track["track_id"]: {
                "is_subtitle": self.is_subtitle,
                "model": "v4-test",
                "response_ref": "responses/request_01.json",
                "cache_hit": False,
            }
            for track in tracks
        }


def frames(tmp_path):
    result = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        assert cv2.imwrite(str(path), np.full((720, 1280, 3), 30, dtype=np.uint8))
        result.append(Frame(timestamp=index * 0.5, image_path=str(path)))
    return result


def provider(classifier, store, static_variance_threshold=0.0):
    return HybridSubtitleRegionDetectorProvider(
        proposal_provider=Proposal(),
        crop_batch_classifier=classifier,
        decision_store=store,
        static_variance_threshold=static_variance_threshold,
    )


def _store_with_response(tmp_path):
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir()
    response.write_text("[]", encoding="utf-8")
    return DecisionStore("test-run", path=tmp_path / "mask_decisions.jsonl")


def _attach_proposal_bundle(detector, frame_list, proposal):
    image = cv2.imread(frame_list[0].image_path)
    height, width = image.shape[:2]
    detections = proposal.detect_images([frame.image_path for frame in frame_list])
    bundle = ProposalBundle("", fps=1.0, stride=1, video_width=width, video_height=height)
    for index, frame in enumerate(frame_list):
        bundle.add_frame(
            frame_index=index,
            timestamp_sec=frame.timestamp,
            image=cv2.imread(frame.image_path),
            image_path=frame.image_path,
            detections=[item for item in detections if item["frame_index"] == index],
        )
    detector.proposal_bundle = bundle


def test_positive_classifier_decision_creates_confirmed_mask_regions(tmp_path):
    frame_list = frames(tmp_path)
    detector = provider(BatchClassifier(), _store_with_response(tmp_path))
    _attach_proposal_bundle(detector, frame_list, Proposal())
    result = detector.detect_regions(frame_list)

    assert len(result) == 3
    assert all(region.track_id == "T001" for region in result)


def test_negative_verifier_decision_does_not_create_mask_region(tmp_path):
    frame_list = frames(tmp_path)
    detector = provider(BatchClassifier(is_subtitle=False), _store_with_response(tmp_path))
    _attach_proposal_bundle(detector, frame_list, Proposal())
    result = detector.detect_regions(frame_list)

    assert result == []


def test_verifier_exception_fails_closed_without_mask_region(tmp_path, caplog):
    frame_list = frames(tmp_path)
    detector = provider(BatchClassifier(error=TimeoutError("network timeout")), _store_with_response(tmp_path))
    _attach_proposal_bundle(detector, frame_list, Proposal())
    result = detector.detect_regions(frame_list)

    assert result == []
    assert "LOUD-WARN" in caplog.text


def test_missing_matching_candidate_decision_fails_closed_without_mask_region(tmp_path, caplog):
    class MissingDecisionClassifier:
        def classify_batch(self, _tracks, _bundle):
            return {}

    frame_list = frames(tmp_path)
    detector = provider(MissingDecisionClassifier(), _store_with_response(tmp_path))
    _attach_proposal_bundle(detector, frame_list, Proposal())
    result = detector.detect_regions(frame_list)

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


def test_ass_generator_renders_confirmed_region_with_track_provenance(tmp_path):
    segment = Segment(id="s", start=0.0, end=1.0, text="source", translated_text="dịch", source="ocr")
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir()
    response.write_text("[]", encoding="utf-8")
    store = DecisionStore("test-run", path=tmp_path / "mask_decisions.jsonl")
    store.record(
        track_id="T00001",
        is_subtitle=True,
        model="v21_pipeline",
        response_ref="responses/request_01.json",
        cache_hit=False,
    )
    confirmed = SubtitleRegion(0.0, 1.0, 0.1, 0.7, 0.4, 0.1, 0.9, track_id="T00001")

    ass = ASSGenerator.generate(
        [segment], SubtitleStyle(), play_res_x=720, play_res_y=1280,
        mask_regions=[confirmed], decision_store=store,
    )

    mask_lines = [
        line for line in ass.splitlines()
        if line.startswith("Dialogue: 0,") and "m 0 0" in line
    ]
    assert len(mask_lines) == 1
    assert f"track={confirmed.track_id}" in mask_lines[0]


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
    detector = provider(BatchClassifier(), _store_with_response(tmp_path))
    detector.proposal_bundle = bundle

    result = detector.detect_regions(frames(tmp_path))

    assert result
    assert result[0].start_time == 0.0
    assert max(region.end_time for region in result) > 4 / 30.0


def test_track_provenance_not_rendered_on_screen(tmp_path):
    segment = Segment(
        id="s", start=0.0, end=1.0, text="source", translated_text="dich", source="ocr"
    )
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir()
    response.write_text("[]", encoding="utf-8")
    store = DecisionStore("test-run", path=tmp_path / "mask_decisions.jsonl")
    store.record(
        track_id="T00001",
        is_subtitle=True,
        model="v21_pipeline",
        response_ref="responses/request_01.json",
        cache_hit=False,
    )
    confirmed = SubtitleRegion(0.0, 1.0, 0.1, 0.7, 0.4, 0.1, 0.9, track_id="T00001")

    ass = ASSGenerator.generate(
        [segment], SubtitleStyle(), play_res_x=720, play_res_y=1280,
        mask_regions=[confirmed], decision_store=store,
    )

    mask_lines = [
        line for line in ass.splitlines()
        if line.startswith("Dialogue: 0,") and "m 0 0" in line
    ]
    assert len(mask_lines) == 1
    assert f"track={confirmed.track_id}" in mask_lines[0]
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

    class MockSpyClassifier:
        def __init__(self):
            self.called = False
        def classify_batch(self, tracks, bundle):
            self.called = True
            return {}

    spy = MockSpyClassifier()
    det = HybridSubtitleRegionDetectorProvider(
        proposal_provider=TopProposal(),
        crop_batch_classifier=spy,
        decision_store=_store_with_response(tmp_path),
    )
    frame_list = frames(tmp_path)
    _attach_proposal_bundle(det, frame_list, TopProposal())
    result = det.detect_regions(frame_list)

    assert result == []
    assert spy.called is False
