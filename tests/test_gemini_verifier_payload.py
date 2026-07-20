import base64
import cv2
import numpy as np

from src.infrastructure.benchmark.hybrid_subtitle.gemini_verifier import GeminiTrackVerifier


class _CannedResponse:
    def __init__(self, text, model="gemini-3.1-flash-lite"):
        self.text = text
        self.model = model
        self.cache_hit = False
        self.finish_reason = "STOP"


class FakeGeminiClient:
    """Capture the GeminiRequest and return a canned valid classification."""

    def __init__(self):
        self.captured = None

    def generate_content(self, req, use_cache=True):
        self.captured = req
        # Minimal valid JSON body for _parse_response_text / validate_gemini_decision
        body = (
            '{"track_id": "T1", "decisions": ['
            '{"candidate_id": "C1", "is_subtitle": true, '
            '"category": "dialogue_subtitle", "confidence": 0.9, '
            '"transcription": null, "reason_code": "v1"}], '
            '"uncertain": false}'
        )
        return _CannedResponse(body)


def _make_frames(n):
    return [np.zeros((80, 140, 3), dtype=np.uint8) for _ in range(n)]


def _make_track(track_id, candidate_ids):
    class TrackWrapper:
        def __init__(self, tid):
            self.track_id = tid

    return TrackWrapper(track_id)


def test_classify_track_requests_well_formed_contents_envelope():
    client = FakeGeminiClient()
    verifier = GeminiTrackVerifier(client, model="gemini-3.1-flash-lite")

    track = _make_track("T1", ["C1"])
    frames = _make_frames(2)

    result = verifier.classify_track(track, frames, ["C1"], [[10, 20, 30, 40]])

    req = client.captured
    assert req is not None
    # contents must be a list of exactly one dict wrapping "parts"
    assert isinstance(req.contents, list)
    assert len(req.contents) == 1
    envelope = req.contents[0]
    assert "parts" in envelope
    assert "text" not in envelope
    assert "inline_data" not in envelope
    parts = envelope["parts"]
    assert parts[0] == {"text": parts[0]["text"]} or isinstance(parts[0].get("text"), str)
    # first part is the text prompt; later parts are image inline_data
    assert isinstance(parts[0].get("text"), str)
    for part in parts[1:]:
        assert set(part.keys()) == {"inline_data"}
        inline = part["inline_data"]
        assert inline.get("mime_type") == "image/png"
        assert isinstance(inline.get("data"), str) and len(inline.get("data")) > 0
    # system instruction and model are forwarded
    assert isinstance(req.system_instruction, str) and req.system_instruction.strip()
    assert req.model == "gemini-3.1-flash-lite"
    # returned decisions parsed from the canned body
    assert any(d.get("is_subtitle") for d in result.get("decisions", []))


def test_classify_requests_well_formed_contents_envelope():
    client = FakeGeminiClient()
    verifier = GeminiTrackVerifier(client, model="gemini-3.1-flash-lite")

    track = {"track_id": "T2", "groups": [{"candidate_group_id": "C1"}]}
    rep_frames = [(0, [10, 20, 30, 40], "first"), (1, [12, 22, 32, 42], "last")]

    # cap stub: returns a dummy frame per requested position
    class _Cap:
        def set(self, *a, **k):
            pass

        def read(self):
            return True, np.zeros((80, 140, 3), dtype=np.uint8)

    result = verifier.classify(track, rep_frames, ["C1"], _Cap())

    req = client.captured
    assert req is not None
    assert isinstance(req.contents, list)
    assert len(req.contents) == 1
    envelope = req.contents[0]
    assert "parts" in envelope
    assert "text" not in envelope
    assert "inline_data" not in envelope
    parts = envelope["parts"]
    assert isinstance(parts[0].get("text"), str)
    for part in parts[1:]:
        assert set(part.keys()) == {"inline_data"}
        inline = part["inline_data"]
        assert inline.get("mime_type") == "image/png"
        assert isinstance(inline.get("data"), str) and len(inline.get("data")) > 0
    assert isinstance(req.system_instruction, str) and req.system_instruction.strip()
    assert req.model == "gemini-3.1-flash-lite"
    # the second classify path validates via validate_gemini_decision
    assert "decisions" in result


def test_classify_sends_clean_frames_and_bbox_geometry():
    client = FakeGeminiClient()
    verifier = GeminiTrackVerifier(client, model="gemini-3.1-flash-lite")
    frame = np.full((80, 140, 3), 17, dtype=np.uint8)
    track = {"track_id": "T2", "groups": [{"candidate_group_id": "C1"}]}

    class _Cap:
        def set(self, *args, **kwargs):
            pass

        def read(self):
            return True, frame.copy()

    verifier.classify(track, [(0, [10, 20, 30, 40], "first")], ["C1"], _Cap())

    parts = client.captured.contents[0]["parts"]
    assert "[10, 20, 30, 40]" in parts[0]["text"]
    payload = parts[1]["inline_data"]["data"]
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, frame)
