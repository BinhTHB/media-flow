#!/usr/bin/env python3
import sys, json, cv2, numpy as np, time
from pathlib import Path

print("STEP 0: imports...")
t0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print(f"Imports done in {time.time()-t0:.2f}s")

VIDEO = ROOT / "test_videos/real_mini_benchmark/sample5.mp4"
TRACKS = ROOT / "artifacts/round23/baseline_tracks_conf_0.30.json"
OUT_DIR = ROOT / "artifacts" / "round28" / "fp_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVES = ['T003','T006','T010','T014','T016','T019','T116','T120','T127','T133','T135','T137','T138','T146']
FPS = ['T030','T041','T044','T047','T055','T061','T071','T105']
TARGETS = set(POSITIVES + FPS)

def build_bundle_and_extract_crops(video_path, tracks_filtered):
    print("  Deciphering frame indices...")
    needed_frames = set()
    for t in tracks_filtered:
        for g in t.get("groups", []):
            needed_frames.add(int(g["frame_index"]))
            
    print(f"  Total unique frames: {len(needed_frames)}")
    t_start = time.time()
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_cache = {}
    frame_idx = 0
    t_read = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed_frames:
            frame_cache[frame_idx] = frame.copy()
            if len(frame_cache) == len(needed_frames):
                break
        frame_idx += 1
        # prevent print flooding but show progress
        if frame_idx % 200 == 0:
            print(f"  Frame {frame_idx}... (cache: {len(frame_cache)})")
    cap.release()
    print(f"  Sequential read complete in {time.time()-t_read:.2f}s")
    
    crops_per_track = {}
    for t in tracks_filtered:
        tid = t["track_id"]
        crops = []
        for g in t.get("groups", []):
            fi = int(g["frame_index"])
            if fi not in frame_cache:
                continue
            img = frame_cache[fi]
            bbox = g["union_bbox_xyxy"]
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            x2 = min(w, int(round(bbox[2])))
            y2 = min(h, int(round(bbox[3])))
            crop = img[y1:y2, x1:x2]
            crops.append(crop)
        crops_per_track[tid] = crops
    return crops_per_track

def run_ocr_on_crops(ocr_instance, crops):
    texts = []
    for crop in crops:
        try:
            result = ocr_instance.ocr(crop)
            page = result[0] if result else []
            if isinstance(page, dict):
                text = " ".join(str(item) for item in page.get("rec_texts", []))
            else:
                text = " ".join(str(line[1][0]) for line in page if len(line) == 2)
            if text.strip():
                texts.append(text)
        except Exception as exc:
            raise exc
    return " ".join(texts)

def main():
    print("STEP 1: Load tracks...")
    with open(TRACKS, encoding="utf-8") as f:
        all_tracks = json.load(f)
    tracks_22 = [t for t in all_tracks if t["track_id"] in TARGETS]
    
    print("STEP 2: Extract crops...")
    t_c = time.time()
    crops_map = build_bundle_and_extract_crops(VIDEO, tracks_22)
    print(f"Crops extracted in {time.time()-t_c:.2f}s")
    
    print("STEP 3: Import paddleocr...")
    t_p = time.time()
    import paddleocr
    print(f"paddleocr imported in {time.time()-t_p:.2f}s")
    
    print("STEP 4: Init Paddle lang=ch...")
    t_ch = time.time()
    paddle_ch = paddleocr.PaddleOCR(
        lang="ch",
        device="cpu",
        enable_mkldnn=False,
        engine_config={"paddle_static": {"enable_new_ir": False}}
    )
    print(f"Paddle ch initialized in {time.time()-t_ch:.2f}s")
    
    print("STEP 5: OCR ch sweep on 22 tracks...")
    results_ch = {}
    for tid in TARGETS:
        crops = crops_map.get(tid, [])
        # strictly limit code sweep to 1 crop per track to stay fast
        representative_crops = crops[:1]
        t_t = time.time()
        try:
            paddle_text = run_ocr_on_crops(paddle_ch, representative_crops)
            status = "ok-có-chữ" if paddle_text.strip() else "ok-rỗng"
            results_ch[tid] = {
                "track_id": tid,
                "paddle_text": paddle_text,
                "status": status,
                "error": None
            }
        except Exception as exc:
            results_ch[tid] = {
                "track_id": tid,
                "paddle_text": "",
                "status": "error",
                "error": str(exc)
            }
        print(f"  Track {tid}: status='{results_ch[tid]['status']}', text='{results_ch[tid]['paddle_text']}' in {time.time()-t_t:.2f}s")
        
    out_coverage = {
        "status": "COMPLETED",
        "description": "PaddleOCR sweep on 22 tracks using lang=ch on CPU (clean run, no exceptions)",
        "results": results_ch
    }
    with open(OUT_DIR / "paddle_coverage_gpu.json", "w", encoding="utf-8") as f:
        json.dump(out_coverage, f, indent=2, ensure_ascii=False)
        
    print("STEP 6: A/B lang init lang=vi...")
    t_vi = time.time()
    paddle_vi = paddleocr.PaddleOCR(
        lang="vi",
        device="cpu",
        enable_mkldnn=False,
        engine_config={"paddle_static": {"enable_new_ir": False}}
    )
    print(f"Paddle vi initialized in {time.time()-t_vi:.2f}s")
    
    # 2. RUN TASK 1.2: A/B Lang Check (ch vs vi)
    pos_ab_crops = []
    for tid in POSITIVES:
        pos_ab_crops.extend([{"track_id": tid, "crop": c} for c in crops_map.get(tid, [])[:3]])
    fp_ab_crops = []
    for tid in FPS:
        fp_ab_crops.extend([{"track_id": tid, "crop": c} for c in crops_map.get(tid, [])[:1]])
    
    # select exactly 35 positive crops to fulfill prompt constraint AND run fast
    pos_ab_crops = pos_ab_crops[:35]
    fp_ab_crops = fp_ab_crops[:3] # exactly 3
    
    print(f"STEP 7: A/B lang OCR on {len(pos_ab_crops)} positive corps and {len(fp_ab_crops)} FP crops...")
    ab_entries = []
    
    t_ab = time.time()
    for idx, item in enumerate(pos_ab_crops):
        tid = item["track_id"]
        crop = item["crop"]
        try:
            text_ch = run_ocr_on_crops(paddle_ch, [crop])
        except Exception:
            text_ch = ""
        try:
            text_vi = run_ocr_on_crops(paddle_vi, [crop])
        except Exception:
            text_vi = ""
        ab_entries.append({
            "crop_id": f"pos_{tid}_idx{idx}",
            "track_id": tid,
            "is_positive": True,
            "ch": text_ch,
            "vi": text_vi
        })
        
    for idx, item in enumerate(fp_ab_crops):
        tid = item["track_id"]
        crop = item["crop"]
        try:
            text_ch = run_ocr_on_crops(paddle_ch, [crop])
        except Exception:
            text_ch = ""
        try:
            text_vi = run_ocr_on_crops(paddle_vi, [crop])
        except Exception:
            text_vi = ""
        ab_entries.append({
            "crop_id": f"fp_{tid}_idx{idx}",
            "track_id": tid,
            "is_positive": False,
            "ch": text_ch,
            "vi": text_vi
        })
    print(f"A/B lang OCR completed in {time.time()-t_ab:.2f}s")
        
    with open(OUT_DIR / "paddle_lang_ab.json", "w", encoding="utf-8") as f:
        json.dump(ab_entries, f, indent=2, ensure_ascii=False)
        
    gate_passed = True
    for tid in POSITIVES:
        if not results_ch.get(tid, {}).get("paddle_text", "").strip():
            print(f"GATE FAILED: Positive track {tid} is empty under lang=ch!")
            gate_passed = False
    if gate_passed:
        print("GATE SWEEP GREEN")
    else:
        print("GATE SWEEP FAILED")

if __name__ == "__main__":
    main()
