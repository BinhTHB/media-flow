# script to extract context and crop for auditing the 19 GT positives
import os
import json
import cv2

video_path = "input/sample5.mp4"
gt_path = "artifacts/round26/gt_v2_2.json"
output_dir = "artifacts/round28/fn_diagnosis"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)

positives = [e for e in gt["entries"] if e.get("is_subtitle") is True]

print(f"Extracting {len(positives)} positive entries from video...")

for entry in positives:
    eid = entry["id"]
    ts = entry["timestamp_s"]
    bbox = entry["bbox_xyxy"] # [x1, y1, x2, y2] in pixels
    
    # Seek to frame
    frame_idx = int(round(ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at {ts}s (frame_idx={frame_idx})")
        continue
    
    # Create copies
    context_with_bbox = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Draw green rectangle for bbox
    cv2.rectangle(context_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        context_with_bbox, f"{eid} ({ts}s)", (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    
    # Extra context margin for crop
    margin = 5
    h_max, w_max = frame.shape[:2]
    cy1 = max(0, y1 - margin)
    cy2 = min(h_max, y2 + margin)
    cx1 = max(0, cx1 := x1 - margin)
    cx2 = min(w_max, x2 + margin)
    crop = frame[cy1:cy2, cx1:cx2]
    
    # Save the output images
    cv2.imwrite(os.path.join(output_dir, f"{eid}_context.jpg"), context_with_bbox)
    if crop.size > 0:
        cv2.imwrite(os.path.join(output_dir, f"{eid}_crop.jpg"), crop)
    else:
        print(f"Empty crop for {eid} with bbox {bbox}")

cap.release()
print("Done extracting!")
