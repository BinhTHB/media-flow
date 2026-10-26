"""Regression tests: subtitle translation must accept partial success.

A real dubbing job (outputs/jobs/job_v2_1790573383) marked all 21 subtitle
events TRANSLATION_FAILED because one provider outage failed the single
all-or-nothing batch call."""

import threading
import time

from src.core.models.bbox import BBox
from src.core.models.subtitle_event import SubtitleEvent
from src.features.visual.subtitle.processor import SubtitleProcessor


def _event(idx: int) -> SubtitleEvent:
    return SubtitleEvent(
        event_id=f"E{idx}",
        track_id="T1",
        start_time=float(idx),
        end_time=float(idx) + 1.0,
        anchor_bbox=BBox(10, 20, 100, 60, 320, 240),
        source_text=f"text {idx}",
    )


class _FirstChunkFails:
    """Fails the first translate call, succeeds afterwards."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def translate_texts(self, texts, target_language="vi", source_language="auto"):
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            raise RuntimeError("All translation providers failed. gemini: The read operation timed out")
        return [f"{text}-vi" for text in texts]


def test_failed_chunk_is_retried_once_and_recovers(monkeypatch):
    monkeypatch.setattr("src.features.visual.subtitle.processor.time.sleep", lambda seconds: None)
    service = _FirstChunkFails()
    processor = SubtitleProcessor(translation_service=service, translation_chunk_size=2)
    events = [_event(i) for i in range(4)]

    processor._translate_events(events, "vi")

    # Round 1: chunk [E0,E1] fails, [E2,E3] succeeds. Round 2 retries only the
    # failed chunk, which now succeeds.
    statuses = {event.event_id: event.replacement_status for event in events}
    assert statuses == {"E0": "SKIPPED", "E1": "SKIPPED", "E2": "SKIPPED", "E3": "SKIPPED"}
    assert events[0].translated_text == "text 0-vi"
    assert events[1].translated_text == "text 1-vi"
    assert events[2].translated_text == "text 2-vi"
    assert events[3].translated_text == "text 3-vi"
    assert service.calls == 3  # 2 chunks in round 1 + 1 retried chunk


def test_translation_length_mismatch_keeps_fail_safe_status():
    class _ShortProvider:
        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            return ["dịch"]  # fewer results than inputs

    processor = SubtitleProcessor(translation_service=_ShortProvider(), translation_chunk_size=8)
    events = [_event(i) for i in range(3)]

    processor._translate_events(events, "vi")

    assert events[0].translated_text == "dịch"
    assert events[1].translated_text == ""
    assert events[2].translated_text == ""
    assert events[1].replacement_status == "SKIPPED"
    assert events[2].replacement_status == "SKIPPED"


def test_all_chunks_failing_marks_every_event_translation_failed(monkeypatch):
    monkeypatch.setattr("src.features.visual.subtitle.processor.time.sleep", lambda seconds: None)

    class _AlwaysFails:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = threading.Lock()

        def translate_texts(self, texts, target_language="vi", source_language="auto"):
            with self._lock:
                self.calls += 1
            raise RuntimeError("provider down")

    service = _AlwaysFails()
    processor = SubtitleProcessor(translation_service=service, translation_chunk_size=2)
    events = [_event(i) for i in range(3)]

    processor._translate_events(events, "vi")

    assert all(event.replacement_status == "TRANSLATION_FAILED" for event in events)
    assert all(not event.translated_text for event in events)
    # Retry budget is bounded: 2 chunks x (1 first round + 1 second round).
    assert service.calls == 4


def _ocr_event(idx: int) -> SubtitleEvent:
    event = _event(idx)
    event.source_text = ""  # OCR stage starts from empty source text
    return event


def test_ocr_batch_failure_is_retried_once_and_recovers(monkeypatch):
    monkeypatch.setattr("src.features.visual.subtitle.processor.time.sleep", lambda seconds: None)

    class _FirstBatchFails:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = threading.Lock()

        def read_crops(self, crops):
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                raise RuntimeError("request payload too large")
            return [f"ocr-{i}" for i, _ in enumerate(crops)]

    provider = _FirstBatchFails()
    processor = SubtitleProcessor(ocr_provider=provider, ocr_chunk_size=2)
    events = [_ocr_event(i) for i in range(4)]
    crops = [object()] * 4
    metadata = [(event, 0) for event in events]

    processor._ocr_event_crops(crops, metadata, events)

    # Round 1: batch [E0,E1] fails, [E2,E3] succeeds. Round 2 retries the
    # failed batch, which now succeeds.
    assert provider.calls == 3
    assert events[0].source_text == "ocr-0"
    assert events[1].source_text == "ocr-1"
    assert events[2].source_text == "ocr-0"
    assert events[3].source_text == "ocr-1"
    assert all(event.replacement_status == "SKIPPED" for event in events)


def test_ocr_all_batches_failing_is_bounded(monkeypatch):
    monkeypatch.setattr("src.features.visual.subtitle.processor.time.sleep", lambda seconds: None)

    class _AlwaysFails:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = threading.Lock()

        def read_crops(self, crops):
            with self._lock:
                self.calls += 1
            raise RuntimeError("model overloaded")

    provider = _AlwaysFails()
    processor = SubtitleProcessor(ocr_provider=provider, ocr_chunk_size=2)
    events = [_ocr_event(i) for i in range(4)]
    crops = [object()] * 4
    metadata = [(event, 0) for event in events]

    processor._ocr_event_crops(crops, metadata, events)

    assert all(event.replacement_status == "OCR_FAILED" for event in events)
    # Bounded: 2 batches x (1 first round + 1 retry round).
    assert provider.calls == 4


def test_ocr_empty_event_gets_upscale_verification_pass(monkeypatch):
    # Evidence rule companion: before concluding "no caption here", the
    # processor re-reads an empty event's crop upscaled 2x — small or
    # stylized burned-in text often only reads at 2x. A recovery re-enters
    # the normal translate/READY flow; a second empty result stays OCR_EMPTY
    # and (per the pipeline rule) is never redacted.
    import numpy as np
    import cv2 as cv2_module

    from src.core.models.bbox import BBox
    from src.features.visual.subtitle.processor import SubtitleProcessor

    class _FakeCapture:
        def __init__(self, _path):
            self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def set(self, *_args):
            return None

        def read(self):
            return True, self.frame

        def release(self):
            return None

    class _FakeOcr:
        def __init__(self):
            self.shapes = []

        def read_crops(self, crops):
            self.shapes.append([c.shape for c in crops])
            return [""]

    monkeypatch.setattr(cv2_module, "VideoCapture", _FakeCapture)

    ocr = _FakeOcr()
    processor = SubtitleProcessor(ocr_provider=ocr)
    event = SubtitleEvent(
        event_id="hallucination",
        track_id="T1",
        start_time=19.50,
        end_time=19.77,
        anchor_bbox=BBox(x1=456, y1=716, x2=936, y2=843, frame_width=1920, frame_height=1080),
        bbox_trajectory=[],
        bbox_frame_indices=[],
        source_text="",
        translated_text="",
        replacement_status="OCR_EMPTY",
        blur_applied=False,
        overlay_applied=False,
    )

    processor._verify_ocr_empty_events([event], "fake.mp4")

    # The crop was upscaled 2x: shape (h, w, c) = ((843-716)*2, (936-456)*2, 3).
    assert ocr.shapes and ocr.shapes[0][0][0] == (843 - 716) * 2
    assert ocr.shapes[0][0][1] == (936 - 456) * 2
    # Still empty after verification: empty status, no invented text.
    assert event.replacement_status == "OCR_EMPTY"
    assert not event.source_text


def test_ocr_verify_pass_recovers_text_into_ready_flow(monkeypatch):
    import numpy as np
    import cv2 as cv2_module

    from src.core.models.bbox import BBox
    from src.features.visual.subtitle.processor import SubtitleProcessor

    class _FakeCapture:
        def __init__(self, _path):
            self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def set(self, *_args):
            return None

        def read(self):
            return True, self.frame

        def release(self):
            return None

    class _FakeOcr:
        def read_crops(self, _crops):
            return ["  双手轻柔  "]

    monkeypatch.setattr(cv2_module, "VideoCapture", _FakeCapture)

    processor = SubtitleProcessor(ocr_provider=_FakeOcr())
    event = SubtitleEvent(
        event_id="small_text",
        track_id="T1",
        start_time=10.0,
        end_time=11.0,
        anchor_bbox=BBox(x1=456, y1=716, x2=936, y2=843, frame_width=1920, frame_height=1080),
        bbox_trajectory=[],
        bbox_frame_indices=[],
        source_text="",
        translated_text="",
        replacement_status="OCR_EMPTY",
        blur_applied=False,
        overlay_applied=False,
    )

    processor._verify_ocr_empty_events([event], "fake.mp4")

    assert event.source_text == "双手轻柔"
    assert event.replacement_status == "SKIPPED"  # re-enters translate/READY flow


def test_bilingual_pair_translates_to_single_collapsed_line():
    # Real case (7s): a merged bilingual event ('第一条: 要温柔\nOne: be
    # gentle') came back with two near-identical Vietnamese lines — the model
    # translated each language row separately. A human places ONE line, so
    # the collapse keeps the first and drops the near-duplicate; genuinely
    # different lines survive.
    processor = SubtitleProcessor(ocr_provider=None, translation_service=None)

    twin = SubtitleEvent(
        event_id="twin",
        track_id="T1",
        start_time=0.0,
        end_time=1.0,
        anchor_bbox=None,
        bbox_trajectory=[],
        bbox_frame_indices=[],
        source_text="第一条: 要温柔\nOne: be gentle",
        translated_text="Điều thứ nhất: Phải dịu dàng\nThứ nhất: Phải dịu dàng",
        replacement_status="SKIPPED",
        blur_applied=False,
        overlay_applied=False,
    )
    distinct = SubtitleEvent(
        event_id="distinct",
        track_id="T2",
        start_time=2.0,
        end_time=3.0,
        anchor_bbox=None,
        bbox_trajectory=[],
        bbox_frame_indices=[],
        source_text="第一行\n第二行",
        translated_text="Mẹ luôn dạy chúng con\nĐừng bao giờ bỏ cuộc",
        replacement_status="SKIPPED",
        blur_applied=False,
        overlay_applied=False,
    )

    processor._collapse_duplicate_translation_lines([twin, distinct])

    assert twin.translated_text == "Điều thứ nhất: Phải dịu dàng"
    assert distinct.translated_text == "Mẹ luôn dạy chúng con\nĐừng bao giờ bỏ cuộc"


class _ConcurrencyTracking:
    """Records the peak number of overlapping calls to prove parallelism."""

    def __init__(self, delay: float = 0.08) -> None:
        self._delay = delay
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0

    def _enter(self) -> None:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        # Busy-wait instead of time.sleep: the suite patches processor time.sleep
        # to no-op the retry pauses, which would also erase this delay.
        deadline = time.perf_counter() + self._delay
        while time.perf_counter() < deadline:
            pass

    def _exit(self) -> None:
        with self._lock:
            self._active -= 1


class _ParallelTranslationService(_ConcurrencyTracking):
    def translate_texts(self, texts, target_language="vi", source_language="auto"):
        self._enter()
        try:
            return [f"{text}-vi" for text in texts]
        finally:
            self._exit()


class _ParallelOCRProvider(_ConcurrencyTracking):
    def read_crops(self, crops):
        self._enter()
        try:
            return [f"{crop}" for crop in crops]
        finally:
            self._exit()


def test_parallel_translation_chunks_overlap(monkeypatch):
    monkeypatch.setattr("src.features.visual.subtitle.processor.time.sleep", lambda seconds: None)
    service = _ParallelTranslationService()
    processor = SubtitleProcessor(
        translation_service=service,
        translation_chunk_size=1,
        translation_max_workers=3,
    )
    events = [_event(i) for i in range(9)]  # 9 chunks -> 3 workers overlap

    processor._translate_events(events, "vi")

    assert service.max_concurrent >= 2
    assert all(event.translated_text == f"text {event.event_id[1:]}-vi" for event in events)
    assert all(event.replacement_status == "SKIPPED" for event in events)


def test_parallel_ocr_batches_overlap_and_keep_mapping(monkeypatch):
    monkeypatch.setattr("src.features.visual.subtitle.processor.time.sleep", lambda seconds: None)
    provider = _ParallelOCRProvider()
    processor = SubtitleProcessor(
        ocr_provider=provider,
        ocr_chunk_size=2,
        ocr_max_workers=3,
    )
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    events = [_ocr_event(i) for i in range(6)]
    crops = words[:]
    metadata = [(event, 0) for event in events]

    processor._ocr_event_crops(crops, metadata, events)

    assert provider.max_concurrent >= 2
    # Concurrent batches must still land each text on its own event.
    assert [event.source_text for event in events] == words
    assert all(event.replacement_status == "SKIPPED" for event in events)
