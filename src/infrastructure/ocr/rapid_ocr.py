"""Local ONNX OCR adapter (RapidOCR / PP-OCR).

Reads supplied subtitle crops on-device. Does not translate, invent text, or
move bboxes. Gemini remains a separate adapter for crops this engine leaves
empty.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any, List, Optional

from src.core.contracts import OCRPort

logger = logging.getLogger(__name__)

_SHARED_ENGINE: Any = None


def rapidocr_available() -> bool:
    return find_spec("rapidocr") is not None or find_spec("rapidocr_onnxruntime") is not None


class RapidOCRAdapter(OCRPort):
    def __init__(self, engine: Any = None) -> None:
        self.model = "rapidocr"
        self._engine = engine

    @property
    def available(self) -> bool:
        return self._engine is not None or rapidocr_available()

    def read_crops(self, crop_images: List[Any]) -> List[str]:
        if not crop_images:
            return []
        engine = self._engine or self._load_engine()
        texts: list[str] = []
        for crop in crop_images:
            if crop is None or getattr(crop, "size", 0) == 0:
                texts.append("")
                continue
            fitted = self._fit_crop(crop)
            try:
                # YOLO already boxed the line. Recognition-only skips the
                # detector, which was several seconds per 1080p crop.
                raw = engine(fitted, use_det=False, use_cls=False, use_rec=True)
            except Exception as exc:
                logger.warning("RapidOCR failed on one crop: %s", exc)
                texts.append("")
                continue
            text = self._text_from_result(raw)
            if not text:
                # An empty recognition pass is retried with the detector.
                # Stylized plates often fail recognition-only and still read
                # once the line is located inside the crop.
                try:
                    raw = engine(fitted, use_det=True, use_cls=False, use_rec=True)
                except Exception as exc:
                    logger.warning("RapidOCR detector retry failed on one crop: %s", exc)
                else:
                    text = self._text_from_result(raw)
            texts.append(text)
        return texts

    def _load_engine(self) -> Any:
        global _SHARED_ENGINE
        if _SHARED_ENGINE is not None:
            self._engine = _SHARED_ENGINE
            return self._engine
        if find_spec("rapidocr") is not None:
            from rapidocr import RapidOCR

            _SHARED_ENGINE = RapidOCR()
        elif find_spec("rapidocr_onnxruntime") is not None:
            from rapidocr_onnxruntime import RapidOCR

            _SHARED_ENGINE = RapidOCR()
        else:
            raise RuntimeError("RapidOCR is not installed. pip install rapidocr")
        self._engine = _SHARED_ENGINE
        return self._engine

    @staticmethod
    def _fit_crop(crop: Any, max_edge: int = 960) -> Any:
        """Downscale wide 1080p plates. Detector cost grows with pixel count."""
        height, width = crop.shape[:2]
        long_edge = max(height, width)
        if long_edge <= max_edge:
            return crop
        import cv2

        scale = max_edge / float(long_edge)
        return cv2.resize(
            crop,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _text_from_result(raw: Any) -> str:
        if raw is None:
            return ""
        txts = getattr(raw, "txts", None)
        if txts is not None:
            return "\n".join(str(part).strip() for part in txts if str(part).strip())
        payload = raw[0] if isinstance(raw, tuple) and raw else raw
        if not payload:
            return ""
        lines: list[str] = []
        if isinstance(payload, str):
            return payload.strip()
        for row in payload:
            if row is None:
                continue
            if isinstance(row, str):
                if row.strip():
                    lines.append(row.strip())
                continue
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                text = row[1]
                if isinstance(text, (list, tuple)) and text:
                    text = text[0]
                value = str(text).strip()
                if value:
                    lines.append(value)
        return "\n".join(lines)
