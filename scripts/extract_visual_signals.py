"""Extract visual signals from all 443 crops for efficient manual review."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

ROOT = Path("artifacts/hybrid_subtitle_benchmark")
OUT = ROOT / "semantic_subset"

# Load candidates
candidates = []
with open(OUT / "semantic_subset_candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        candidates.append(json.loads(line.strip()))

print(f"Loaded {len(candidates)} candidates", file=sys.stderr)

# Load tracks for coordinate context
tracks_by_sample = {}
for sample in ["sample1","sample2","sample3","sample4","sample5"]:
    tpath = ROOT / f"{sample}_text_tracks.json"
    if tpath.exists():
        tracks_by_sample[sample] = {t["track_id"]: t for t in json.loads(tpath.read_text(encoding="utf-8"))}

# Load video dimensions
videos = {}
for sample in ["sample1","sample2","sample3","sample4","sample5"]:
    vidpath = ROOT / "cache" / f"{sample}_h264.mp4"
    cap = cv2.VideoCapture(str(vidpath))
    fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    videos[sample] = (int(fw), int(fh))
    cap.release()

# Init EasyOCR if available
reader = None
if EASYOCR_AVAILABLE:
    try:
        reader = easyocr.Reader(["vi","en"], gpu=False)
        print("EasyOCR ready", file=sys.stderr)
    except Exception as e:
        print(f"EasyOCR init failed: {e}", file=sys.stderr)
        reader = None

results = []
for i, cand in enumerate(candidates):
    crop_path = OUT / "candidate_crops" / f"{cand["scene_id"]}_{cand["candidate_id"]}.jpg"
    if not crop_path.exists():
        results.append({**cand, "error": "crop not found", "ocr_text": "", "visual_features": {}})
        continue
    
    img = cv2.imread(str(crop_path))
    h, w = img.shape[:2]
    
    # Basic features
    mean_brightness = float(np.mean(img))
    std_brightness = float(np.std(img))
    
    # Position of bbox in video frame
    scene_id = cand["scene_id"]
    sample = scene_id.split("_scene_")[0]
    track_id = cand["track_id"]
    bbox = cand["bbox_xyxy"]
    vw, vh = videos.get(sample, (1440, 2160))
    x1,y1,x2,y2 = bbox
    rel_x_center = ((x1+x2)/2) / vw
    rel_y_min = y1 / vh
    rel_y_max = y2 / vh
    rel_width = (x2-x1) / vw
    rel_height = (y2-y1) / vh
    aspect_ratio = (x2-x1) / max(1, y2-y1)
    
    # Check if bright/white text (common in subtitles)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_pixels = float(np.sum(gray > 200)) / max(1, gray.size)
    dark_pixels = float(np.sum(gray < 50)) / max(1, gray.size)
    
    # OCR text
    ocr_text = ""
    if reader:
        try:
            ocr_result = reader.readtext(str(crop_path))
            ocr_text = " | ".join([t[1] for t in ocr_result if t[1].strip()])
        except Exception as e:
            ocr_text = f"[OCR error: {e}]"
    
    features = {
        "crop_size_pixels": f"{w}x{h}",
        "rel_x_center": round(rel_x_center, 3),
        "rel_y_min": round(rel_y_min, 3),
        "rel_y_max": round(rel_y_max, 3),
        "rel_width": round(rel_width, 3),
        "rel_height": round(rel_height, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "mean_brightness": round(mean_brightness, 1),
        "std_brightness": round(std_brightness, 1),
        "bright_pixels_200": round(bright_pixels, 3),
        "dark_pixels_50": round(dark_pixels, 3),
    }
    
    results.append({**cand, "ocr_text": ocr_text, "visual_features": features})
    
    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{len(candidates)}", file=sys.stderr)

print(f"Done! Processed {len(results)} candidates", file=sys.stderr)

# Write full results
with open(OUT / "candidate_visual_signals.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Write a summary CSV for quick review
import csv
csv_path = OUT / "candidate_review_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scene_id","candidate_id","track_id","selection_reason","is_subtitle","category","review_status","review_notes",
                "ocr_text","rel_x_center","rel_y_min","rel_y_max","rel_width","rel_height","aspect_ratio","bright_pixels_200","error"])
    for r in results:
        vf = r.get("visual_features", {})
        w.writerow([
            r["scene_id"], r["candidate_id"], r["track_id"], r["selection_reason"],
            "", "", "", "",  # manual review columns to fill
            r.get("ocr_text",""),
            vf.get("rel_x_center",""), vf.get("rel_y_min",""), vf.get("rel_y_max",""),
            vf.get("rel_width",""), vf.get("rel_height",""), vf.get("aspect_ratio",""),
            vf.get("bright_pixels_200",""),
            r.get("error","")
        ])

print(f"Written: {csv_path}")
