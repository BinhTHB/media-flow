# Reviewing evaluation labels

1. Open `review_index.html` in a browser and inspect the drawn boxes.
2. For each frame, edit `../labels/<frame_id>.json`: fix class ids, boxes (`bbox_xyxy`), remove wrong boxes, add missing ones, and set `verified: true`.
3. Set `verified` to `true` in `../manifests/manifest.csv` for that row.
4. Re-run `scripts/evaluate_yolo_detection.py` — it refuses to compute Precision/Recall/F1 until every evaluated frame is verified.
