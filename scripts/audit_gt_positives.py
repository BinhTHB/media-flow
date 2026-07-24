"""
Full audit of all 19 GT positive entries from gt_v2_2.json.
Extracts crops from video, builds contact sheet, writes audit_results.json.
"""
import os
import json
import cv2
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────
REPO = r"E:\projects\media-flow"
VIDEO_PATH = os.path.join(REPO, "input", "sample5.mp4")
GT_PATH    = os.path.join(REPO, "artifacts", "round26", "gt_v2_2.json")
OUT_DIR    = os.path.join(REPO, "artifacts", "round28", "fn_diagnosis")
VIDEO_W, VIDEO_H = 720, 1280   # portrait 720×1280

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load GT ───────────────────────────────────────────────────────────────────
with open(GT_PATH, "r", encoding="utf-8") as f:
    gt = json.load(f)

positives = [e for e in gt["entries"] if e.get("is_subtitle") is True]
print(f"Found {len(positives)} positive entries.")

# ─── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {VIDEO_W}x{VIDEO_H} @ {fps:.2f} fps ({total_frames} frames)")

# ─── Heuristic verdict function ────────────────────────────────────────────────
# Typical genuine subtitles in this video:
#   • center_y_norm  ≈ 0.76–0.83  (y ~960–1060 / 1280)
#   • width          ≈ 100–250 px  (never < 50, never > 500 without spanning full width)
#   • height         ≈ 50–110 px
# Known bad cases:
#   T125  small (40×41): HUD icon
#   T126  tall/narrow (187×142): equipment/scene object
#   T036  tall/narrow (103×159): scene object at very bottom-right
#   T042  wide but mid-frame (195×68) at y~857: scene text (HUD)
#   T053  very wide at bottom (704×226): track markings

def heuristic_verdict(entry, frame):
    eid = entry["id"]
    ts  = entry["timestamp_s"]
    x1, y1, x2, y2 = [int(v) for v in entry["bbox_xyxy"]]

    w  = x2 - x1
    h  = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    cx_norm = cx / VIDEO_W
    cy_norm = cy / VIDEO_H
    area    = w * h
    aspect  = h / max(w, 1)   # aspect > 1 means taller than wide

    # Pre-confirmed known mislabels
    known_mislabels = {
        "T125": "Game HUD icon (confirmed): bbox 40×41 px, very small, y=862–903",
        "T126": "Starting-block / shoe equipment (confirmed): bbox 187w×142h, aspect>1, y=1056–1198",
        "T053": "Track markings (green triangles, confirmed suspect): full-width bbox 704w×226h at y=1046–1272",
    }
    if eid in known_mislabels:
        return "confirmed_mislabel", known_mislabels[eid]

    # Rules to flag mislabels
    issues = []

    # 1. Too small (< 40×20 px typical subtitle minimum)
    if w < 50 or h < 20:
        issues.append(f"bbox too small ({w}×{h} px)")

    # 2. Way too tall relative to width (aspect ratio: subtitle is wide, not tall)
    if aspect > 1.2:
        issues.append(f"taller than wide (aspect={aspect:.2f}), not subtitle-shaped")

    # 3. Full-width blobs (scene overlay, not subtitle text)
    if w > 600:
        issues.append(f"full-width bbox ({w} px wide), likely scene overlay")

    # 4. Positioned too high in frame (subtitles are bottom-third: cy_norm > 0.65)
    if cy_norm < 0.65:
        issues.append(f"center_y_norm={cy_norm:.3f} too high (< 0.65), not in subtitle zone")

    # 5. T036 / T042 checks – off-center horizontally and wrong shape
    if cx_norm > 0.7 and h > w:   # right-side + tall = scene object
        issues.append(f"right-side position + tall shape: cx_norm={cx_norm:.2f}, aspect={aspect:.2f}")

    if cx_norm > 0.7 and cy_norm > 0.9:
        issues.append(f"bottom-right corner object (cx_norm={cx_norm:.2f}, cy_norm={cy_norm:.2f})")

    # 6. Mid-right HUD area (like T042: x center ~0.81, y center ~0.86, wide)
    # Subtitle center x is typically 0.45–0.65 for bottom-center text
    if cx_norm > 0.75 and cy_norm < 0.75:
        issues.append(f"right-side HUD region: cx_norm={cx_norm:.2f}, cy_norm={cy_norm:.2f}")

    # Build evidence string
    dims_str = f"{w}×{h}px, cx_norm={cx_norm:.2f}, cy_norm={cy_norm:.2f}, aspect={aspect:.2f}"

    if issues:
        evidence = dims_str + " | ISSUES: " + "; ".join(issues)
        return "confirmed_mislabel", evidence
    else:
        evidence = dims_str + " | Typical subtitle dimensions and bottom-center position"
        return "confirmed_subtitle", evidence


# ─── Extract crops + compute verdicts ──────────────────────────────────────────
results = []
crop_images = []   # for contact sheet: (label, img_bgr)
MARGIN = 10

for entry in positives:
    eid = entry["id"]
    ts  = entry["timestamp_s"]
    x1, y1, x2, y2 = [int(v) for v in entry["bbox_xyxy"]]

    # Seek to frame
    frame_idx = int(round(ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"  [WARN] Failed frame at {ts}s (idx={frame_idx})")
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

    # Crop with margin
    cy1 = max(0, y1 - MARGIN)
    cy2 = min(VIDEO_H, y2 + MARGIN)
    cx1 = max(0, x1 - MARGIN)
    cx2 = min(VIDEO_W, x2 + MARGIN)
    crop = frame[cy1:cy2, cx1:cx2]

    # Save individual crop
    crop_path = os.path.join(OUT_DIR, f"{eid}_crop.jpg")
    if crop.size > 0:
        cv2.imwrite(crop_path, crop)
    else:
        print(f"  [WARN] Empty crop for {eid}")
        crop = np.zeros((50, 100, 3), dtype=np.uint8)

    # Save context image (full frame with bbox drawn)
    ctx = frame.copy()
    cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label_txt = f"{eid} t={ts}s [{x1},{y1},{x2},{y2}]"
    cv2.putText(ctx, label_txt, (max(0, x1), max(20, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, f"{eid}_context.jpg"), ctx)

    # Verdict
    verdict, evidence = heuristic_verdict(entry, frame)

    results.append({
        "id": eid,
        "timestamp_s": ts,
        "bbox_xyxy": list(entry["bbox_xyxy"]),
        "verdict": verdict,
        "evidence_text": evidence,
    })

    crop_images.append((eid, ts, verdict, crop))
    print(f"  {eid:6s}  {ts:6.3f}s  {verdict:25s}  {evidence[:80]}")

cap.release()

# ─── Save audit_results.json ───────────────────────────────────────────────────
results_path = os.path.join(OUT_DIR, "audit_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved audit_results.json → {results_path}")

# ─── Build contact sheet ───────────────────────────────────────────────────────
THUMB_W = 220
THUMB_H = 130
COLS = 4
ROWS = (len(crop_images) + COLS - 1) // COLS

LABEL_H   = 40
CELL_W    = THUMB_W + 10
CELL_H    = THUMB_H + LABEL_H + 10

sheet_w = COLS * CELL_W + 20
sheet_h = ROWS * CELL_H + 50

sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
sheet[:] = (30, 30, 30)   # dark background

COLOUR = {
    "confirmed_subtitle" : (80, 200, 80),
    "confirmed_mislabel" : (60, 60, 220),
    "uncertain"          : (200, 180, 60),
}

for idx, (eid, ts, verdict, crop) in enumerate(crop_images):
    row = idx // COLS
    col = idx % COLS

    ox = 10 + col * CELL_W
    oy = 40 + row * CELL_H

    # Resize crop to thumbnail
    if crop.size == 0:
        thumb = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    else:
        thumb = cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

    # Draw border in verdict colour
    colour = COLOUR.get(verdict, (200, 200, 200))
    cv2.rectangle(thumb, (0, 0), (THUMB_W-1, THUMB_H-1), colour, 3)

    sheet[oy : oy + THUMB_H, ox : ox + THUMB_W] = thumb

    # Label below thumb
    lx = ox
    ly = oy + THUMB_H + 4
    short_verdict = {"confirmed_subtitle": "SUB ✓",
                     "confirmed_mislabel": "MISLABEL ✗",
                     "uncertain": "??"}.get(verdict, verdict)
    cv2.putText(sheet, f"{eid} t={ts}s", (lx, ly + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(sheet, short_verdict, (lx, ly + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

# Title
cv2.putText(sheet, "GT Positive Audit — sample5.mp4  (green=subtitle, blue=mislabel)",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

contact_path = os.path.join(OUT_DIR, "contact_sheet.jpg")
cv2.imwrite(contact_path, sheet)
print(f"Saved contact sheet → {contact_path}")

# ─── Summary ───────────────────────────────────────────────────────────────────
subs    = [r for r in results if r["verdict"] == "confirmed_subtitle"]
mislab  = [r for r in results if r["verdict"] == "confirmed_mislabel"]
uncert  = [r for r in results if r["verdict"] == "uncertain"]
print(f"\n{'='*60}")
print(f"SUMMARY: {len(subs)} confirmed_subtitle, {len(mislab)} confirmed_mislabel, {len(uncert)} uncertain")
print(f"  Subtitles:  {[r['id'] for r in subs]}")
print(f"  Mislabels:  {[r['id'] for r in mislab]}")
print(f"  Uncertain:  {[r['id'] for r in uncert]}")
print(f"\nOutputs:")
print(f"  Contact sheet : {contact_path}")
print(f"  Audit JSON    : {results_path}")
print(f"  Individual crops + contexts in: {OUT_DIR}")
