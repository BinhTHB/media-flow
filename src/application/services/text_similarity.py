from difflib import SequenceMatcher

class TextSimilarityService:
    @staticmethod
    def get_similarity(text1: str, text2: str) -> float:
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()
        return SequenceMatcher(None, t1, t2).ratio()
