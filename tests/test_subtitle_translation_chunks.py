"""Regression tests: subtitle translation must accept partial success.

A real dubbing job (outputs/jobs/job_v2_1790573383) marked all 21 subtitle
events TRANSLATION_FAILED because one provider outage failed the single
all-or-nothing batch call."""

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

    def translate_texts(self, texts, target_language="vi", source_language="auto"):
        self.calls += 1
        if self.calls == 1:
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

        def translate_texts(self, texts, target_language="vi", source_language="auto"):
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

        def read_crops(self, crops):
            self.calls += 1
            if self.calls == 1:
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

        def read_crops(self, crops):
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
