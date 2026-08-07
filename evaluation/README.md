# Evaluation dataset

Layout:
- `manifests/manifest.csv` — sampled frames (source video + timestamp).
- `manifests/exclusion_manifest.csv` — videos/frames excluded and why.
- `frames/` — extracted frame images.
- `labels/` — per-frame labels. **UNVERIFIED pre-labels** until reviewed.
- `review/` — human review package (previews + index + instructions).

Do not treat pre-labels as ground truth. Metrics are only computed from frames marked `verified=true`.
