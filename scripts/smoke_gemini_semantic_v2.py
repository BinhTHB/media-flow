"""Scoped live-Gemini semantic smoke test v2 on 10 stratified scenes."""
from __future__ import annotations
import sys
sys.path.insert(0, '.')

import base64, json, logging, random, time, yaml
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path("artifacts/hybrid_subtitle_benchmark/semantic_subset")
OUT_DIR   = ROOT / "smoke_output"
MANIFEST  = ROOT / "semantic_subset_manifest.json"
GT_FILE   = ROOT / "manual_semantic_ground_truth.jsonl"
CACHE_DIR = ROOT / "review_frames"
CROP_DIR  = ROOT / "candidate_crops"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load manifest, GT ─────────────────────────────────────────────────────────
manifest  = json.loads(MANIFEST.read_text(encoding="utf-8"))["scenes"]
gt_rows   = { (r["scene_id"], r["candidate_id"]): r for r in
              [json.loads(l) for l in GT_FILE.read_text(encoding="utf-8").splitlines() if l.strip()] }

# ── load gemini runtime ────────────────────────────────────────────────────────
cfg = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8"))
api_key = cfg["gemini"]["api_key"]
if not api_key or len(api_key) < 10:
    raise SystemExit("No valid gemini api_key in config.yml")

from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.gemini.contracts import GeminiRequest

runtime = GeminiRuntime(RuntimeConfig(api_key=api_key, cache_enabled=False,
                                     max_retries=0, min_request_spacing_seconds=0))
client = runtime.get_client("gemini-3.1-flash-lite", api_key=api_key)

# ── Hardcode exact same 10 scenes from v1 smoke test ──────────────────────────
SMOKE_SCENE_IDS = [
    "sample1_scene_005",
    "sample1_scene_001",
    "sample2_scene_005",
    "sample2_scene_009",
    "sample2_scene_006",
    "sample5_scene_010",
    "sample1_scene_013",
    "sample5_scene_013",
    "sample4_scene_020",
    "sample1_scene_019"
]

selected = [sc for sc in manifest if sc["scene_id"] in SMOKE_SCENE_IDS]
# Preserve exact order
selected.sort(key=lambda s: SMOKE_SCENE_IDS.index(s["scene_id"]))

# Pool naming map from v1 selection reasons
pools_map = {
    "sample1_scene_005": "clear_subtitle",
    "sample1_scene_001": "clear_subtitle",
    "sample2_scene_005": "clear_subtitle",
    "sample2_scene_009": "mixed",
    "sample2_scene_006": "mixed",
    "sample5_scene_010": "mixed",
    "sample1_scene_013": "no_subtitle",
    "sample5_scene_013": "no_subtitle",
    "sample4_scene_020": "difficult",
    "sample1_scene_019": "difficult"
}
for sc in selected:
    sc["_pool"] = pools_map[sc["scene_id"]]

assert len(selected) == 10, f"Selected {len(selected)} != 10"

# ── per-scene ground truth lookup ──────────────────────────────────────────────
def gt_for(scene_id):
    return {r["candidate_id"]: r for sid, r in gt_rows.items() if sid == scene_id}

# ── system prompt v2 ──────────────────────────────────────────────────────────
SYSTEM_V2 = (
    "You are a vision-language classifier for on-screen text in video frames.\n"
    "Geometry is provided by a local detector. Do NOT re-evaluate or modify coordinates. "
    "Only classify the semantic category of each provided candidate ID.\n\n"
    "Your task is semantic role classification, not generic text detection. "
    "Determine what role the text plays in the video context, not merely where it is located or how readable it is.\n\n"
    "Subtitle Definition:\n"
    "- Text added to the video to represent spoken dialogue, narration, lyrics, or an audio event.\n"
    "- It may appear directly on the main video OR inside an embedded device screen (such as a phone screen, computer monitor, TV, or picture-in-picture video being filmed).\n"
    "- Do NOT reject a candidate only because it is inside an embedded device screen or filmed device screen. If it represents dialogue/speech in that embedded media, it is still a subtitle.\n\n"
    "Non-Subtitle Categories to Reject:\n"
    "- persistent watermarks (e.g. usernames, logos, watermark badges)\n"
    "- UI controls / labels (e.g. share buttons, heart icons, TikTok safety/legal disclaimers, score overlays, system badges)\n"
    "- decorative text or graphics (e.g. pink question mark stickers, sticker graphics, emojis)\n"
    "- scene text (e.g. text printed physically on signs, walls, products in the scene)\n"
    "- title cards / credits / logos / headers\n\n"
    "Allowed categories:\n"
    "- dialogue_subtitle, narration_subtitle, sound_caption, karaoke_or_lyric_subtitle\n"
    "- scene_text, watermark, logo, ui_text, social_overlay, title_card, credits, decorative_text, unknown.\n\n"
    "General examples for reference:\n"
    "- Dialogue text inside a phone video -> dialogue_subtitle\n"
    "- Fixed username in a corner -> watermark\n"
    "- Pink decorative question-mark sticker -> decorative_text\n"
    "- Safety disclaimer displayed independently of speech -> ui_text\n"
    "- Text physically printed on a sign in the background -> scene_text\n\n"
    "Return JSON only matching the schema exactly. Do NOT invent candidate IDs."
)

@dataclass
class RequestMetaV2:
    scene_id: str
    model: str
    image_count: int
    image_dimensions: str
    total_image_payload_bytes: int
    latency_ms: float
    attempt_count: int
    http_status: int
    api_status: str
    schema_valid: bool
    candidate_ids_valid: bool
    cache_hit: bool
    number_of_full_frames: int
    number_of_crops: int
    temporal_span_ms: float
    candidate_count: int

@dataclass
class SceneResultV2:
    scene_id: str
    pool: str
    candidate_ids: list
    http_status: int
    api_status: str
    latency_ms: float
    attempt: int
    schema_valid: bool
    tp: int; fp: int; tn: int; fn: int
    precision: float; recall: float; f1: float; specificity: float
    gemini_decisions: list
    error: str

errors_log = []
request_meta = []
scene_results = []

ALLOWED_CATS = {"dialogue_subtitle","narration_subtitle","sound_caption",
                "karaoke_or_lyric_subtitle","scene_text","watermark","logo",
                "ui_text","social_overlay","title_card","credits","decorative_text","unknown"}

SUBTITLE_CATS = {"dialogue_subtitle", "narration_subtitle", "sound_caption", "karaoke_or_lyric_subtitle"}

for idx, scene in enumerate(selected, 1):
    sid = scene["scene_id"]
    pool_name = scene["_pool"]
    cand_ids = scene["candidate_ids"]
    gt = gt_for(sid)

    print(f"[{idx}/10] {sid} ({pool_name})")

    # ── build full frames ─────────────────────────────────────────────────────
    frames = []
    for fi in scene["frame_indices"]:
        frame_path = CACHE_DIR / f"{sid}_F{fi:05d}.jpg"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            frames.append(img)

    # ── build crops ───────────────────────────────────────────────────────────
    crops = []
    for cid in cand_ids:
        crop_path = CROP_DIR / f"{sid}_{cid}.jpg"
        if crop_path.exists():
            img_c = cv2.imread(str(crop_path))
            crops.append((cid, img_c))

    if not frames:
        err = f"No frames found for {sid}"
        print(f"  SKIP: {err}")
        errors_log.append({"scene_id": sid, "error": err, "type": "missing_frames"})
        continue

    # ── build prompt ─────────────────────────────────────────────────────────
    user_prompt = (
        f"Classify candidate IDs in the following frames (scene {sid}).\n"
        f"Candidate IDs: {json.dumps(cand_ids)}\n\n"
        "Return exactly this JSON schema:\n"
        "{\n"
        f"  \"scene_id\": \"{sid}\",\n"
        "  \"decisions\": [\n"
        "    {\n"
        "      \"candidate_id\": \"<id>\",\n"
        "      \"category\": \"<allowed_category>\",\n"
        "      \"confidence\": 0.0-1.0,\n"
        "      \"evidence\": {\n"
        "        \"represents_spoken_or_audio_content\": true/false,\n"
        "        \"embedded_media_context\": true/false,\n"
        "        \"physically_attached_to_scene_object\": true/false,\n"
        "        \"persistent_brand_or_ui_element\": true/false\n"
        "      },\n"
        "      \"reason_code\": \"<short_reason>\"\n"
        "    }\n"
        "  ],\n"
        "  \"uncertain_candidate_ids\": []\n"
        "}"
    )

    # ── encode images (full frames first, then crops) ────────────────────────
    content_parts = [{"text": user_prompt}]
    total_bytes = 0
    dims = ""
    
    # 1. Add full frames
    content_parts.append({"text": "\n--- FULL FRAMES WITH LABELS ---"})
    for fi_idx, frame in enumerate(frames):
        h, w = frame.shape[:2]
        if fi_idx == 0:
            dims = f"{w}x{h}"
        scale = 640 / max(w, h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        _, png = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        b64 = base64.b64encode(png.tobytes()).decode("ascii")
        total_bytes += len(b64)
        content_parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

    # 2. Add crops
    content_parts.append({"text": "\n--- ZOOMED CROPS OF CANDIDATES ---"})
    for cid, crop in crops:
        ch, cw = crop.shape[:2]
        scale = 120 / max(cw, ch) if max(cw, ch) > 120 else 1.0
        if scale < 1.0:
            crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)))
        _, png_c = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        b64_c = base64.b64encode(png_c.tobytes()).decode("ascii")
        total_bytes += len(b64_c)
        content_parts.append({"text": f"Candidate ID: {cid}"})
        content_parts.append({"inline_data": {"mime_type": "image/png", "data": b64_c}})

    # ── API call ─────────────────────────────────────────────────────────────
    req = GeminiRequest(
        contents=[{"role": "user", "parts": content_parts}],
        model="gemini-3.1-flash-lite",
        generation_config={"temperature": 0.1, "maxOutputTokens": 2048},
        system_instruction=SYSTEM_V2,
        request_id=sid,
        source_kind="smoke_subtitle_verifier_v2",
        api_surface="benchmark",
    )

    lat_ms = 0.0
    attempt = 1
    http_status = 0
    api_status = "unknown"
    schema_valid = False
    cand_valid = False
    decisions = []
    error_msg = ""

    try:
        t0 = time.time()
        resp = client.generate_content(req, use_cache=False)
        lat_ms = round((time.time() - t0) * 1000)
        http_status = 200
        api_status = getattr(resp, "finish_reason", "STOP") or "unknown"
        schema_valid = True
        cand_valid = True
        text = resp.text if hasattr(resp, "text") else ""

        try:
            data = json.loads(text) if text.strip() else {}
        except Exception:
            data = {"decisions": []}
            error_msg = f"JSON parse failed: {text[:200]}"

        if isinstance(data, dict) and "decisions" in data:
            decisions = data["decisions"]
        elif isinstance(data, list):
            decisions = data
        else:
            decisions = []
            error_msg = f"Unexpected schema: {text[:200]}"
            schema_valid = False

        valid_ids = set(cand_ids)
        decisions = [d for d in decisions if d.get("candidate_id") in valid_ids]
        returned_ids = {d.get("candidate_id") for d in decisions}
        if returned_ids != valid_ids:
            cand_valid = False
            error_msg = f"Candidate IDs mismatch: expected {valid_ids}, got {returned_ids}"

        # Local validation & is_subtitle derivation
        for d in decisions:
            cat = d.get("category")
            if cat not in ALLOWED_CATS:
                d["category"] = "unknown"
                cat = "unknown"
            d["is_subtitle"] = cat in SUBTITLE_CATS

    except Exception as exc:
        error_msg = str(exc)[:300]
        lat_ms = 0
        http_status = 0

    span = scene["timestamps_ms"][-1] - scene["timestamps_ms"][0] if len(scene["timestamps_ms"]) > 1 else 33.3

    meta = RequestMetaV2(
        scene_id=sid, model="gemini-3.1-flash-lite",
        image_count=len(frames) + len(crops), image_dimensions=dims,
        total_image_payload_bytes=total_bytes,
        latency_ms=lat_ms, attempt_count=attempt,
        http_status=http_status, api_status=api_status,
        schema_valid=schema_valid, candidate_ids_valid=cand_valid,
        cache_hit=False,
        number_of_full_frames=len(frames),
        number_of_crops=len(crops),
        temporal_span_ms=span,
        candidate_count=len(cand_ids)
    )
    request_meta.append(asdict(meta))

    tp = fp = tn = fn = 0
    for cid in cand_ids:
        gt_r = gt.get(cid)
        if not gt_r:
            continue
        expected = gt_r["is_subtitle"]
        found = next((d for d in decisions if d.get("candidate_id") == cid), None)
        predicted = bool(found.get("is_subtitle")) if found else False
        
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    scene_results.append({
        "scene_id": sid, "pool": pool_name,
        "candidate_ids": cand_ids,
        "http_status": http_status, "api_status": api_status,
        "latency_ms": lat_ms, "attempt": attempt,
        "schema_valid": schema_valid, "candidate_ids_valid": cand_valid,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec,4), "recall": round(rec,4),
        "f1": round(f1,4), "specificity": round(spec,4),
        "gemini_decisions": [{"candidate_id": d["candidate_id"],
                              "is_subtitle": d.get("is_subtitle"),
                              "category": d.get("category"),
                              "confidence": d.get("confidence"),
                              "evidence": d.get("evidence"),
                              "reason_code": d.get("reason_code")}
                             for d in decisions],
        "ground_truth": {cid: {"is_subtitle": gt[cid]["is_subtitle"],
                               "category": gt[cid]["category"]} for cid in cand_ids if cid in gt},
        "error": error_msg
    })

    print(f"  -> HTTP {http_status} | {api_status} | {lat_ms:.0f}ms | TP={tp} FP={fp} TN={tn} FN={fn} P={prec:.2f} R={rec:.2f}")

# ── save outputs ─────────────────────────────────────────────────────────────
agg = {"tp":0,"fp":0,"tn":0,"fn":0}
for r in scene_results:
    agg["tp"] += r.get("tp",0)
    agg["fp"] += r.get("fp",0)
    agg["tn"] += r.get("tn",0)
    agg["fn"] += r.get("fn",0)
tp=agg["tp"]; fp=agg["fp"]; tn=agg["tn"]; fn=agg["fn"]
prec = tp/(tp+fp) if (tp+fp) else 0.0
rec  = tp/(tp+fn) if (tp+fn) else 0.0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
spec = tn/(tn+fp) if (tn+fp) else 0.0
lats = [r["latency_ms"] for r in scene_results if r.get("latency_ms",0)>0]
lats_sorted = sorted(lats)
p50 = lats_sorted[len(lats_sorted)//2] if lats_sorted else 0
p95 = lats_sorted[int(len(lats_sorted)*0.95)] if lats_sorted else 0

metrics = {
    "requests_total": len(selected),
    "requests_success": sum(1 for r in scene_results if r.get("http_status")==200),
    "requests_fail": sum(1 for r in scene_results if r.get("http_status")!=200),
    "responses_valid_schema": sum(1 for r in scene_results if r.get("schema_valid")),
    "responses_valid_candidates": sum(1 for r in scene_results if r.get("candidate_ids_valid")),
    "tp":tp,"fp":fp,"tn":tn,"fn":fn,
    "precision":round(prec,4),"recall":round(rec,4),
    "f1":round(f1,4),"specificity":round(spec,4),
    "latency_avg_ms": round(sum(lats)/len(lats),1) if lats else 0,
    "latency_p50_ms": round(p50,1),
    "latency_p95_ms": round(p95,1),
    "per_scene": [{"scene_id":r["scene_id"],"pool":r["pool"],
                   "tp":r["tp"],"fp":r["fp"],"tn":r["tn"],"fn":r["fn"],
                   "precision":r["precision"],"recall":r["recall"],
                   "f1":r["f1"],"latency_ms":r["latency_ms"],
                   "http_status":r["http_status"],"error":r.get("error","")}
                  for r in scene_results]
}

# smoke_v2_gemini_decisions.jsonl
dec_lines = [json.dumps({"scene_id":r["scene_id"],"decisions":r["gemini_decisions"]},ensure_ascii=False) for r in scene_results]
(OUT_DIR / "smoke_v2_gemini_decisions.jsonl").open("w",encoding="utf-8").write("\n".join(dec_lines)+"\n")

# smoke_v2_request_metadata_sanitized.jsonl
req_lines = [json.dumps(m, ensure_ascii=False) for m in request_meta]
(OUT_DIR / "smoke_v2_request_metadata_sanitized.jsonl").open("w",encoding="utf-8").write("\n".join(req_lines)+"\n")

# smoke_v2_metrics.json
json.dump(metrics, (OUT_DIR / "smoke_v2_metrics.json").open("w",encoding="utf-8"), indent=2, ensure_ascii=False)

# smoke_v2_comparison_contact_sheet.png
_build_contact_sheet(scene_results, gt_rows, OUT_DIR)

# Draw false positives and false negatives contact sheets
def _draw_error_sheet(results, gt_map, error_type, out_file):
    crops_to_draw = []
    for r in results:
        sid = r["scene_id"]
        gt = { (sid, cid): v for (sid_key, cid), v in gt_map.items() if sid_key == sid }
        decisions = {d["candidate_id"]: d for d in r["gemini_decisions"]}
        for cid in r["candidate_ids"]:
            gt_r = gt.get((sid, cid))
            gm_r = decisions.get(cid, {})
            if not gt_r:
                continue
            is_sub = gt_r.get("is_subtitle",False)
            pred_sub = gm_r.get("is_subtitle",False)
            
            is_error = False
            if error_type == "fp" and (not is_sub and pred_sub):
                is_error = True
            elif error_type == "fn" and (is_sub and not pred_sub):
                is_error = True
                
            if is_error:
                crop_path = CROP_DIR / f"{sid}_{cid}.jpg"
                if crop_path.exists():
                    crop = cv2.imread(str(crop_path))
                    # pad crop to 120x120
                    h,w = crop.shape[:2]
                    padded = np.zeros((120,240,3), dtype=np.uint8)
                    # center crop
                    offset_y = (120 - h) // 2
                    offset_x = (240 - w) // 2
                    padded[max(0,offset_y):min(120,offset_y+h), max(0,offset_x):min(240,offset_x+w)] = crop[0:min(h,120), 0:min(w,240)]
                    # draw text label
                    label = f"{sid}_{cid} GT={gt_r.get('category')} GM={gm_r.get('category')}"
                    cv2.putText(padded, label, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
                    crops_to_draw.append(padded)
    if crops_to_draw:
        # stack them vertically or in a grid
        sheet = np.vstack(crops_to_draw)
        cv2.imwrite(str(out_file), sheet)
    else:
        # write empty placeholder
        cv2.imwrite(str(out_file), np.zeros((120,240,3), dtype=np.uint8))

_draw_error_sheet(scene_results, gt_rows, "fp", OUT_DIR / "smoke_v2_false_positives.png")
_draw_error_sheet(scene_results, gt_rows, "fn", OUT_DIR / "smoke_v2_false_negatives.png")

print("\n=== SMOKE TEST V2 COMPLETE ===")
print(json.dumps(metrics, indent=2))
