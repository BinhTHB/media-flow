from src.domain.entities.segment import Segment


class CaptionCoverageService:
    @staticmethod
    def calculate(segments: list[Segment], video_duration: float) -> dict:
        ordered = sorted(segments, key=lambda segment: segment.start)
        covered = 0.0
        longest_gap = 0.0
        cursor = 0.0
        for segment in ordered:
            start = max(0.0, min(segment.start, video_duration))
            end = max(start, min(segment.end, video_duration))
            longest_gap = max(longest_gap, start - cursor)
            if end > cursor:
                covered += end - max(start, cursor)
                cursor = end
        longest_gap = max(longest_gap, video_duration - cursor)
        count = len(ordered)
        return {
            "video_duration": video_duration,
            "total_caption_duration": covered,
            "caption_coverage_ratio": covered / video_duration if video_duration else 0.0,
            "final_caption_count": count,
            "average_caption_duration": covered / count if count else 0.0,
            "longest_gap_without_captions": longest_gap,
        }
