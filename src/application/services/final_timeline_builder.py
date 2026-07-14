import hashlib
import json
from datetime import datetime, timezone
from typing import List, Sequence

from src.domain.entities.final_timeline import FinalTimeline, TimelineItem
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_region import SubtitleRegion
from src.shared.exceptions import ValidationError


class FinalTimelineBuilder:
    def build(
        self,
        narration_segments: Sequence[Segment],
        display_segments: Sequence[Segment],
        regions: List[SubtitleRegion],
        placements: List[SubtitlePlacement],
        replacement_required: bool,
    ) -> FinalTimeline:
        narration = tuple(self._item(segment) for segment in narration_segments)
        display = tuple(self._item(segment) for segment in display_segments)
        replacement_regions = []
        placement_entries = []
        for index, segment in enumerate(display_segments):
            placement = placements[index] if index < len(placements) else None
            placement_dict = placement.to_dict() if placement else {}
            placement_entries.append(
                {"segment_id": segment.id, **placement_dict}
            )
            if replacement_required:
                if placement is None or placement.mask_bbox is None:
                    raise ValidationError(
                        f"Replacement placement for {segment.id} requires mask_bbox."
                    )
                replacement_regions.append(
                    {
                        "segment_id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "mask_bbox": dict(placement.mask_bbox),
                    }
                )
        payload = {
            "source_segment_ids": [item.segment_id for item in narration],
            "tracks": {
                "narration": [item.to_dict() for item in narration],
                "display_subtitles": [item.to_dict() for item in display],
                "source_subtitle_replacements": replacement_regions,
                "placements": placement_entries,
            },
        }
        timeline_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return FinalTimeline(
            timeline_id=timeline_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_segment_ids=tuple(payload["source_segment_ids"]),
            narration=narration,
            display=display,
            replacement_regions=tuple(replacement_regions),
            placements=tuple(placement_entries),
        )

    @staticmethod
    def _item(segment: Segment) -> TimelineItem:
        text = (segment.translated_text or segment.text).strip()
        return TimelineItem(segment.id, segment.start, segment.end, text)
