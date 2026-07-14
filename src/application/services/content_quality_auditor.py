import re
from typing import Tuple, List

class ContentQualityAuditor:
    # Common Vietnamese incomplete endings
    INCOMPLETE_VI_ENDINGS = {
        "cho", "và", "nhưng", "thì", "để", "của", "tại", "vì", "bởi", "với", "hoặc", "nếu", 
        "anh", "chị", "ông", "bà", "nó", "tôi", "em", "ta", "họ", "chúng", "tuyên", "mang", "mặc"
    }
    
    PLACEHOLDERS = {".", "..", "...", "---", "***", "ah", "oh", "uh", "um", "hmm"}

    @classmethod
    def audit(cls, text: str, source: str) -> Tuple[float, List[str]]:
        text = text.strip()
        issues = []
        
        # 1. Block placeholder-only
        clean_text = re.sub(r"[^\w\s]", "", text).lower().strip()
        if not clean_text or text in cls.PLACEHOLDERS or clean_text in cls.PLACEHOLDERS:
            return 0.0, ["placeholder_only"]

        # 2. Block extremely low-information
        words = text.split()
        if len(words) == 1:
            w = words[0].lower()
            if not bool(re.match(r"^[\u4e00-\u9fff]$", w)):
                if len(w) <= 1 and not w.isdigit():
                    return 0.0, ["extremely_low_information"]

        # 3. Block malformed OCR text after filtering
        if source == "ocr":
            if "å" in text.lower():
                issues.append("malformed_ocr_spelling")
            for w in words:
                w_clean = re.sub(r"[^\w]", "", w).lower()
                if w_clean in {"diu", "khin", "båo", "him"}:
                    issues.append("malformed_ocr_spelling")

        # 4. Flag incomplete Vietnamese fragments
        if len(words) > 0:
            last_word = re.sub(r"[^\w]", "", words[-1]).lower()
            if last_word in cls.INCOMPLETE_VI_ENDINGS:
                issues.append("incomplete_vietnamese_fragment")

        # Calculate score
        score = 1.0
        if "malformed_ocr_spelling" in issues:
            score -= 0.4
        if "incomplete_vietnamese_fragment" in issues:
            score -= 0.3
        if len(words) <= 2:
            score -= 0.2
            
        score = max(0.0, min(1.0, score))
        return round(score, 2), issues