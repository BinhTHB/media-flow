# Script to extract crops for the 6 False Positives for taxonomy classification
import os
import json
import cv2

regions_path = "output/round27-sample5/debug/subtitle_regions.json"
video_path = "input/sample5.mp4"
output_dir = "artifacts/round28/fp_taxonomy"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

with open(regions_path, "r", encoding="utf-8") as f:
    regions = json.load(f)

# Group regions by track_id
tracks_regions = {}
for r in regions:
    tid = r["track_id"]
    tracks_regions.setdefault(tid, []).append(r)

fp_tracks = ["T041", "T055", "T071", "T105", "T044", "T047"]

print("Extracting False Positive track crops...")

for tid in fp_tracks:
    if tid not in tracks_regions:
        print(f"Warning: track {tid} not found in subtitle_regions.json")
        continue
    
    regs = tracks_regions[tid]
    # Pick the middle region as representative context
    mid_reg = regs[len(regs) // 2]
    ts = (mid_reg["start_time"] + mid_reg["end_time"]) / 2
    
    # Seek to frame
    frame_idx = int(round(ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame for {tid} at {ts}s")
        continue
        
    H, W = frame.shape[:2]
    
    # Calculate box in pixels
    x1 = int(round(mid_reg["x"] * W))
    y1 = int(round(mid_reg["y"] * H))
    x2 = int(round((mid_reg["x"] + mid_reg["width"]) * W))
    y2 = int(round((mid_reg["y"] + mid_reg["height"]) * H))
    
    # Draw rectangle
    context_with_bbox = frame.copy()
    cv2.rectangle(context_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        context_with_bbox, f"FP:{tid} ({ts:.2f}s)", (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )
    
    # Crop with margin
    margin = 5
    cy1 = max(0, y1 - margin)
    cy2 = min(H, y2 + margin)
    cx1 = max(0, x1 - margin)
    cx2 = min(W, x2 + margin)
    crop = frame[cy1:cy2, cx1:cx2]
    
    cv2.imwrite(os.path.join(output_dir, f"{tid}_context.jpg"), context_with_bbox)
    if crop.size > 0:
        cv2.imwrite(os.path.join(output_dir, f"{tid}_crop.jpg"), crop)
    else:
        print(f"Empty crop for {tid}")

cap.release()
print("Done extracting FP crops!")
