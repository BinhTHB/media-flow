"""Plan stable subtitle regions from observed geometry, separately for each scene."""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass
from src.core.models.bbox import BBox
from src.core.models.detection import Detection


def same_lane(a: BBox, b: BBox) -> bool:
    overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    if overlap / max(1.0, min(a.width, b.width)) < 0.45:
        return False
    gap = max(0.0, max(a.y1, b.y1) - min(a.y2, b.y2))
    return gap <= max(4.0, min(a.height, b.height) * 0.70)


def union_boxes(boxes: list[BBox]) -> BBox:
    first = boxes[0]
    return BBox(min(b.x1 for b in boxes), min(b.y1 for b in boxes),
                max(b.x2 for b in boxes), max(b.y2 for b in boxes),
                first.frame_width, first.frame_height)


@dataclass(frozen=True)
class SubtitleRegion:
    scene: int
    bbox: BBox
    observations: int


class SubtitleRegionPlanner:
    def __init__(self, detections: list[Detection], scene_cuts: list[int] | None = None):
        self.cuts = sorted(set(scene_cuts or []))
        groups: dict[int, list[list[BBox]]] = {}
        # Highest-confidence boxes seed lanes, avoiding single weak detections
        # creating a bridge between otherwise independent subtitle regions.
        for det in sorted(detections, key=lambda item: -item.confidence):
            if det.class_name != "subtitle":
                continue
            scene = bisect_right(self.cuts, det.frame_index)
            scene_groups = groups.setdefault(scene, [])
            match = next((g for g in scene_groups if same_lane(g[0], det.bbox)), None)
            if match is None:
                scene_groups.append([det.bbox])
            else:
                match.append(det.bbox)
        self.regions: dict[int, list[SubtitleRegion]] = {}
        for scene, lanes in groups.items():
            self.regions[scene] = []
            for lane in lanes:
                # Use robust extents for the stable plate; current event/detection
                # geometry is added at render time so outliers cannot shrink it.
                def q(values, fraction):
                    values = sorted(values)
                    return values[int((len(values) - 1) * fraction)]
                first = lane[0]
                bbox = BBox(q([b.x1 for b in lane], .02), q([b.y1 for b in lane], .02),
                            q([b.x2 for b in lane], .98), q([b.y2 for b in lane], .98),
                            first.frame_width, first.frame_height)
                self.regions[scene].append(SubtitleRegion(scene, bbox, len(lane)))

    def region_for(self, frame_index: int, anchor: BBox) -> BBox:
        scene = bisect_right(self.cuts, frame_index)
        matches = [r for r in self.regions.get(scene, []) if same_lane(r.bbox, anchor)]
        if not matches:
            return anchor
        # Prefer the lane with the strongest temporal support. A sparse fragment
        # can have near-perfect IoU with a partial current detection and would
        # otherwise beat the stable full-caption lane, shrinking the suppression
        # plate for a few frames. The current anchor is still unioned below, so
        # genuine geometry changes can expand the selected stable region.
        best = max(matches, key=lambda r: (r.observations, r.bbox.iou(anchor)))
        return union_boxes([best.bbox, anchor])

    def to_dict(self):
        return {"scene_cuts": self.cuts, "regions": [
            {"scene": r.scene, "observations": r.observations, "bbox": r.bbox.to_dict()}
            for regions in self.regions.values() for r in regions
        ]}
