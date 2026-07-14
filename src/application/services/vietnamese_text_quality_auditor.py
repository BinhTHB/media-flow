import re


class VietnameseTextQualityAuditor:
    CORRECTIONS = (
        (
            "Tao đó, tuyên bố nó đã vừa qua mỗi vòng kiểm tra và cho phép đi tiếp.",
            "Sau khi kiểm tra, cảnh sát cho chiếc xe được đi tiếp.",
        ),
        ("ảnh xét", "cảnh sát"),
        ("cảnh xác", "cảnh sát"),
        ("bắt ngờ", "bất ngờ"),
        ("vẽ mặc", "vẻ mặt"),
        ("người đang ông", "người đàn ông"),
        ("điều khiển từ xe", "điều khiển từ xa"),
        ("đi giả phố", "đi dạo phố"),
        ("anh giá trình đường", "cảnh sát giao thông"),
        ("Ngày lúc", "Ngay lúc"),
        ("tật thâu", "tịch thu"),
        ("vế phạc", "vé phạt"),
        ("chic xe sē", "chiếc xe sẽ"),
        ("cảnh chiếc xe", "cạnh chiếc xe"),
        ("bếp mặt", "vẻ mặt"),
        ("nghiêm tốt", "nghiêm túc"),
        ("xúc trình", "xuất trình"),
        ("giáy tờ", "giấy tờ"),
        ("Tao đó", "Sau đó"),
        ("mỗi vòng", "mọi vòng"),
        ("đi được trước", "đi được chứ"),
        (
            "Sau đó, tuyên bố nó đã vừa vượt qua mọi vòng kiểm tra và cho phép đi tiếp.",
            "Sau khi kiểm tra, cảnh sát cho chiếc xe được đi tiếp.",
        ),
        (
            "Sau đó, tuyên bố nó đã vừa vượt qua\nmọi vòng kiểm tra và cho phép đi tiếp.",
            "Sau khi kiểm tra, cảnh sát cho chiếc xe được đi tiếp.",
        ),
    )

    @classmethod
    def audit(cls, text: str) -> dict:
        corrections = []
        suggested = text
        for incorrect, replacement in cls.CORRECTIONS:
            match = re.search(re.escape(incorrect), text, re.IGNORECASE)
            if not match:
                continue
            suggested = re.sub(
                re.escape(incorrect),
                replacement,
                suggested,
                flags=re.IGNORECASE,
            )
            corrections.append({
                "suspected_issue": "high_confidence_vietnamese_spelling",
                "from": match.group(0),
                "to": replacement,
                "confidence": 0.99,
            })
        return {
            "original_text": text,
            "suggested_text": suggested,
            "corrections": corrections,
            "confidence": max(
                (correction["confidence"] for correction in corrections),
                default=None,
            ),
            "actual_action": "flag" if corrections else "keep",
        }

    @classmethod
    def apply(cls, text: str, enabled: bool) -> dict:
        audit = cls.audit(text)
        if enabled and audit["corrections"]:
            audit["actual_action"] = "correct"
            audit["corrected_text"] = audit["suggested_text"]
        else:
            audit["corrected_text"] = text
        return audit
