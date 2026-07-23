"""R2 language detection rule (V2.1 architecture).

Rejects non-target language subtitles while allowing bilingual exceptions.

Target language is set via config (--source-sub-language) or auto-detected
from sure-YES filtered tracks. Reject ONLY when >= 2 distinct PaddleOCR
samples on DIFFERENT frames confidently agree on non-target language.

Track containing non-target text BUT temporally adjacent + spatially adjacent
to a target-language track at the same cadence → NEUTRAL (bilingual exception).

Bất biến V2.1: reject-only. No language samples → R2 SKIPPED (neutral all).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Target language detection via character category heuristics
# No heavy language model — uses Unicode block analysis
LANGUAGE_HINTS: Dict[str, Set[Tuple[int, int]]] = {
    "zh": {(0x4E00, 0x9FFF), (0x3400, 0x4DBF)},  # CJK Unified
    "ja": {(0x3040, 0x309F), (0x30A0, 0x30FF)},    # Hiragana, Katakana
    "ko": {(0xAC00, 0xD7AF)},                       # Hangul
    "th": {(0x0E00, 0x0E7F)},
    "ar": {(0x0600, 0x06FF)},
    "ru": {(0x0400, 0x04FF)},
    "vi": set(),  # Vietnamese uses Latin + diacritics — not distinguishable by unicode blocks
    "en": set(),  # English uses Latin
    "other": set(),
}


@dataclass
class R2Result:
    rejected: bool
    neutral: bool     # True = insufficient evidence
    reason: str
    detected_lang: str = ""        # e.g. "zh", "en", "vi", "unknown"
    is_target: bool = True
    samples_detected: int = 0
    samples_checked: int = 0
    bilingual_neighbor: bool = False


class R2LanguageDetector:
    """Reject non-target language subtitle tracks.

    Language detection uses PaddleOCR text results from the track's crops.
    Works as a filter on top of TierClassifier's gray zone tracks.
    """

    MIN_SAMPLES_FOR_REJECTION: int = 2  # Need >= 2 confident non-target samples
    BILINGUAL_SPATIAL_TOLERANCE: float = 0.15  # Fraction of frame height
    BILINGUAL_TEMPORAL_TOLERANCE_SEC: float = 0.5

    def __init__(self, target_lang: str | None = None) -> None:
        """Initialize R2.

        Parameters
        ----------
        target_lang : str | None
            Override target language. If None, auto-detect from sure-YES tracks.
            Default: None (auto-detect from sure-YES filtered tracks).
        """
        self._target_lang = target_lang
        self._auto_detected = False

    def set_target_from_sure_yes(
        self, sure_yes_tracks: List[Dict[str, Any]], proposal_bundle: Any
    ) -> str | None:
        """Auto-detect target language from sure-YES filtered tracks.

        Sets the internal target language and returns it. Returns None if
        insufficient data (R2 will be SKIPPED).
        """
        if self._target_lang is not None:
            return self._target_lang

        texts = self._extract_texts_from_tracks(sure_yes_tracks, proposal_bundle)
        if not texts:
            self._auto_detected = True
            return None

        lang_votes: Dict[str, int] = {}
        for text in texts:
            lang = self._detect_language(text)
            lang_votes[lang] = lang_votes.get(lang, 0) + 1

        if lang_votes:
            self._target_lang = max(lang_votes, key=lang_votes.get)
            self._auto_detected = True
            logger.info(
                "R2 auto-detected target language '%s' from %d texts: %s",
                self._target_lang, len(texts), lang_votes,
            )
        return self._target_lang

    def evaluate(
        self,
        track: Dict[str, Any],
        proposal_bundle: Any,
        all_tracks: List[Dict[str, Any]],
        target_lang_tracks: List[Dict[str, Any]],
    ) -> R2Result:
        """Evaluate a single track for language-based rejection.

        Parameters
        ----------
        track : dict
            The track to evaluate.
        proposal_bundle : ProposalBundle
            Source of frame images and cached crops.
        all_tracks : list[dict]
            All tracks in the video (for bilingual neighbor detection).
        target_lang_tracks : list[dict]
            Tracks already accepted as target language (for adjacency check).

        Returns
        -------
        R2Result with rejection decision.
        """
        if self._target_lang is None and not self._auto_detected:
            return R2Result(
                rejected=False, neutral=True,
                reason="No target language configured or auto-detected — R2 SKIPPED",
            )

        groups = track.get("groups", [])
        if not groups:
            return R2Result(
                rejected=False, neutral=True,
                reason="No groups — insufficient language samples",
            )

        # Extract text samples from PaddleOCR via existing crop infrastructure
        texts = self._extract_track_texts(track, proposal_bundle)
        if len(texts) < self.MIN_SAMPLES_FOR_REJECTION:
            return R2Result(
                rejected=False, neutral=True,
                reason=f"Only {len(texts)} text samples (< {self.MIN_SAMPLES_FOR_REJECTION})",
                samples_checked=len(texts),
            )

        # Detect language per sample
        non_target_count = 0
        target_count = 0
        detected_langs: List[str] = []

        for text in texts:
            if not text.strip():
                continue
            lang = self._detect_language(text)
            detected_langs.append(lang)
            if lang == self._target_lang:
                target_count += 1
            elif self._is_ambiguous_text(text):
                # Icons, short symbols, numbers — neutral
                continue
            else:
                non_target_count += 1

        # Check bilingual neighbor exception
        has_bilingual_neighbor = False
        if non_target_count >= self.MIN_SAMPLES_FOR_REJECTION:
            has_bilingual_neighbor = self._check_bilingual_neighbor(
                track, target_lang_tracks
            )

        majority_lang = max(set(detected_langs), key=detected_langs.count) if detected_langs else "unknown"

        if non_target_count >= self.MIN_SAMPLES_FOR_REJECTION and not has_bilingual_neighbor:
            return R2Result(
                rejected=True,
                neutral=False,
                reason=f"Non-target language detected: "
                       f"{non_target_count} non-target vs {target_count} target "
                       f"(majority={majority_lang}, target={self._target_lang})",
                detected_lang=majority_lang,
                is_target=False,
                samples_detected=len(texts),
                samples_checked=len(texts),
                bilingual_neighbor=False,
            )

        return R2Result(
            rejected=False,
            neutral=False,
            reason=f"Language OK: target={target_count}, non-target={non_target_count}, "
                   f"{'bilingual neighbor' if has_bilingual_neighbor else 'accepted'}",
            detected_lang=majority_lang,
            is_target=target_count >= non_target_count or has_bilingual_neighbor,
            samples_detected=len(texts),
            samples_checked=len(texts),
            bilingual_neighbor=has_bilingual_neighbor,
        )

    def _detect_language(self, text: str) -> str:
        """Detect language of a text string using Unicode block heuristics."""
        if not text:
            return "unknown"

        # Score each language family
        scores: Dict[str, int] = {}
        for char in text:
            cp = ord(char)
            for lang, ranges in LANGUAGE_HINTS.items():
                for start, end in ranges:
                    if start <= cp <= end:
                        scores[lang] = scores.get(lang, 0) + 1
                        break

        if not scores:
            # Latin/diacritic — could be English or Vietnamese
            return "en"  # Default Latin

        return max(scores, key=scores.get)

    def _is_ambiguous_text(self, text: str) -> bool:
        """Check if text is ambiguous (icons, numbers, short symbols).

        These should NOT trigger language rejection.
        """
        stripped = text.strip()
        if not stripped:
            return True
        # All digits/punctuation
        alnum = sum(c.isalnum() for c in stripped)
        if alnum == 0:
            return True
        # Very short
        if len(stripped) <= 2:
            return True
        return False

    def _extract_track_texts(
        self, track: Dict[str, Any], proposal_bundle: Any
    ) -> List[str]:
        """Extract OCR text samples from a track using cached crops."""
        texts: List[str] = []
        groups = track.get("groups", [])
        for group in groups:
            fi = group.get("frame_index", -1)
            try:
                decoded = proposal_bundle.frame_for_index(fi)
            except (KeyError, AttributeError):
                continue

            # Try to read text from cached crops
            if decoded.crops and decoded.detections:
                candidate_ids = set(group.get("member_candidate_ids", []))
                for det_idx, det in enumerate(decoded.detections):
                    if det.get("candidate_id", "") in candidate_ids and det_idx < len(decoded.crops):
                        text = self._ocr_crop(decoded.crops[det_idx])
                        if text.strip():
                            texts.append(text)
                        break
        return texts

    def _extract_texts_from_tracks(
        self, tracks: List[Dict[str, Any]], proposal_bundle: Any
    ) -> List[str]:
        """Extract OCR text samples from multiple tracks for language detection."""
        all_texts: List[str] = []
        for track in tracks:
            all_texts.extend(self._extract_track_texts(track, proposal_bundle))
        return all_texts

    def _ocr_crop(self, crop: Any) -> str:
        """Simple OCR on a crop using PaddleOCR.

        Falls back to no-op if PaddleOCR is unavailable.
        """
        try:
            import paddleocr
            if not hasattr(self, "_paddle"):
                self._paddle = paddleocr.PaddleOCR(
                    lang="ch",  # Multi-language model
                    device="gpu",
                    enable_mkldnn=False,
                    engine_config={"paddle_static": {"enable_new_ir": False}},
                    show_log=False,
                )
            result = self._paddle.ocr(crop)
            page = result[0] if result else []
            if isinstance(page, dict):
                lines = page.get("rec_texts", [])
            else:
                lines = [str(line[1][0]) for line in page if len(line) == 2]
            return " ".join(lines)
        except Exception:
            return ""

    def _check_bilingual_neighbor(
        self,
        track: Dict[str, Any],
        target_lang_tracks: List[Dict[str, Any]],
    ) -> bool:
        """Check if a non-target track has a target-language neighbor.

        Bilingual exception: non-target track that is temporally adjacent
        AND spatially adjacent (directly above/below) a target-language track
        at the same cadence position.
        """
        bbox = (
            track.get("median_bbox_xyxy")
            or track.get("temporal_union_bbox_xyxy")
            or []
        )
        if not bbox or len(bbox) < 4:
            return False

        groups = track.get("groups", [])
        track_t_start = groups[0].get("timestamp_ms", 0) / 1000.0 if groups else 0
        track_t_end = groups[-1].get("timestamp_ms", 0) / 1000.0 if groups else 0

        for target in target_lang_tracks:
            t_bbox = (
                target.get("median_bbox_xyxy")
                or target.get("temporal_union_bbox_xyxy")
                or []
            )
            if not t_bbox or len(t_bbox) < 4:
                continue
            t_groups = target.get("groups", [])
            t_start = t_groups[0].get("timestamp_ms", 0) / 1000.0 if t_groups else 0
            t_end = t_groups[-1].get("timestamp_ms", 0) / 1000.0 if t_groups else 0

            # Temporal adjacency
            time_overlap = max(0.0, min(track_t_end, t_end) - max(track_t_start, t_start))
            if time_overlap <= 0:
                gap = min(
                    abs(track_t_end - t_start),
                    abs(track_t_end - t_end),
                    abs(track_t_start - t_start),
                    abs(track_t_start - t_end),
                )
                if gap > self.BILINGUAL_TEMPORAL_TOLERANCE_SEC:
                    continue

            # Spatial adjacency (directly above or below)
            # Vertical gap between track and target
            v_gap = max(0.0, t_bbox[1] - bbox[3], bbox[1] - t_bbox[3])
            h_overlap = max(0.0, min(bbox[2], t_bbox[2]) - max(bbox[0], t_bbox[0]))
            frame_h = max(bbox[3], t_bbox[3], 1.0)

            if h_overlap > 0.1 * (bbox[2] - bbox[0]) and v_gap < self.BILINGUAL_SPATIAL_TOLERANCE * frame_h:
                return True

        return False
