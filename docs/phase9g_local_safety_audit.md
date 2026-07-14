# Phase 9G Local Safety Audit

## Summary

- Total rejected: 14
- Correct reject: 3
- Likely false reject: 9
- Uncertain review: 2
- Top causes: {'person_overlap_only': 9, 'missing_source_subtitle_bbox': 3, 'large_person_overlap': 2}

## Finding

The binary HOG person-overlap rule is too strict. It rejects localized bottom subtitle
boxes even when person overlap is small and scene-text and watermark risks are zero.
Missing source bboxes remain correct fail-closed rejections.

## Proposed Phase 9H rule

Allow a local cover override only for `person_overlap_only` when a localized source bbox
exists, all non-person risks are below `0.20`, and source/person overlap is below `0.35`.
Keep missing-bbox rejections and require manual review for larger overlaps.

Production Ready: NO
