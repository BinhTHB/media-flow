import re
from typing import List, Dict, Any
from src.domain.entities.segment import Segment

class SourceTextLeakageDetector:
    @staticmethod
    def is_vietnamese(text: str) -> bool:
        # Include common Scandinavian and accent misrecognitions (like å, æ, ø) as well
        vi_chars = re.compile(
            r"[đĐáàảãạâầấẩẫậăằắẳẵặéèẻẽẹêềếểễệíìỉĩịóòỏõọôồốổỗộơờớởỡợúùủũụưừứửữựýỳỷỹỵåæø]",
            re.IGNORECASE
        )
        return bool(vi_chars.search(text))

    @staticmethod
    def detect(segments: List[Segment], source_language: str = "zh") -> List[Dict[str, Any]]:
        leaks = []
        for idx, s in enumerate(segments):
            text = s.text.strip()
            translated = s.translated_text.strip()
            if not text or not translated:
                continue

            # Case 1: Chinese source language leakage
            if source_language == "zh":
                has_chinese = bool(re.search(r"[\u4e00-\u9fff]", translated))
                if has_chinese:
                    leaks.append({
                        "index": idx + 1,
                        "timestamp": f"{s.start:.3f} --> {s.end:.3f}",
                        "text": text,
                        "translated_text": translated,
                        "source": s.source,
                        "reason": "Chinese characters remaining in translation"
                    })
                # If source language is zh, Latin texts (English/Vietnamese) are not leaks of Chinese.
                continue

            # Case 2: English/General source text leakage
            is_number = bool(re.match(r"^\d+([.,]\d+)?$", text))
            is_acronym = bool(re.match(r"^[A-Z]{2,}$", text))
            if translated.lower() == text.lower() and not (is_number or is_acronym):
                if any(c.isalpha() for c in text) and not SourceTextLeakageDetector.is_vietnamese(text):
                    leaks.append({
                        "index": idx + 1,
                        "timestamp": f"{s.start:.3f} --> {s.end:.3f}",
                        "text": text,
                        "translated_text": translated,
                        "source": s.source,
                        "reason": "Untranslated text identical to source"
                    })

        return leaks

    @staticmethod
    def count(segments: List[Segment], source_language: str = "zh") -> int:
        return len(SourceTextLeakageDetector.detect(segments, source_language))
