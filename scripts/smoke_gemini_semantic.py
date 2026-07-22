"""Scoped live-Gemini semantic smoke test on 10 stratified scenes."""
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

# ── load manifest, GT, candidates ─────────────────────────────────────────────
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
client = runtime.get_client("gemini-3.5-flash-lite", api_key=api_key)

# ── stratified 10-scene selection ─────────────────────────────────────────────
scenes_by_sample = defaultdict(list)
for sc in manifest:
    scenes_by_sample[sc["video_id"]].append(sc)

# Build per-sample pools
pools = {
    "clear_subtitle":   [],   # 3 scenes: subtitle only, no significant distractors
    "mixed":            [],   # 3 scenes: subtitle + watermark/UI/decorative
    "no_subtitle":      [],   # 2 scenes: no subtitle, many candidates
    "difficult":        [],   # 2 scenes: transition/fragmented/blurry/stylized
}
for sc in manifest:
    reason = sc.get("selection_reason","")
    if "clear_bottom_long_track" in reason:
        pools["clear_subtitle"].append(sc)
    elif "bottom_track_with_concurrent_text" in reason:
        pools["mixed"].append(sc)
    elif "non_bottom_text_dense_scene" in reason:
        pools["no_subtitle"].append(sc)
    else:
        pools["difficult"].append(sc)

# Seed for reproducibility
random.seed(42)
selected = []
for pool_name, count in [("clear_subtitle",3),("mixed",3),("no_subtitle",2),("difficult",2)]:
    pool = pools[pool_name]
    # Prefer scenes with more candidates (richer signal)
    enriched = sorted(pool, key=lambda s: -len(s["candidate_ids"]))
    picked = random.sample(enriched, min(count, len(enriched)))
    for sc in picked:
        sc["_pool"] = pool_name
    selected.extend(picked)

assert len(selected) == 10, f"Selected {len(selected)} != 10"

# ── per-scene ground truth lookup ──────────────────────────────────────────────
def gt_for(scene_id):
    return {r["candidate_id"]: r for sid, r in gt_rows.items() if sid == scene_id}

# ── system prompt ──────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a vision-language classifier for on-screen text in video frames.\n"
    "Geometry is provided by a local detector. Do NOT re-evaluate or modify coordinates. "
    "Only classify each provided candidate ID as subtitle or non-subtitle.\n\n"
    "Subtitle: text added to the video to represent dialogue, narration, audio captions.\n"
    "NOT subtitle: scene text, product labels, signs, usernames, watermarks, logos, "
    "UI controls, title cards, decorative text.\n\n"
    "Allowed categories: dialogue_subtitle, narration_subtitle, sound_caption, "
    "karaoke_or_lyric_subtitle, scene_text, watermark, logo, ui_text, "
    "social_overlay, title_card, credits, decorative_text, unknown.\n\n"
    "Return JSON only. Do NOT invent candidate IDs."
)

@dataclass
class RequestMeta:
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

@dataclass
class SceneResult:
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

def _build_contact_sheet(results, gt_map, out_dir):
    """Build a contact sheet with GT labels and Gemini decisions side-by-side."""
    thumbs = []
    for r in results:
        sid = r["scene_id"]
        # get the first frame
        frame_files = sorted(CACHE_DIR.glob(f"{sid}_F*.jpg"))
        if not frame_files:
            continue
        frame = cv2.imread(str(frame_files[0]))
        h,w = frame.shape[:2]
        scale = 320 / max(w,h)
        if scale < 1:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        # draw GT labels on left, Gemini on right
        gt = { (sid, cid): v for (sid_key, cid), v in gt_map.items() if sid_key == sid }
        decisions = {d["candidate_id"]: d for d in r["gemini_decisions"]}
        overlay = frame.copy()
        for cid in r["candidate_ids"]:
            gt_r = gt.get((sid, cid))
            gm_r = decisions.get(cid, {})
            if not gt_r:
                continue
            is_sub = gt_r.get("is_subtitle",False)
            pred_sub = gm_r.get("is_subtitle", None)
            
            # draw a colored border based on correctness
            if is_sub and pred_sub:
                col = (0,220,0)
            elif not is_sub and not pred_sub:
                col = (0,180,180)
            else:
                col = (0,0,220)  # error - blue/red
            # just label in the corner
            label = f"{cid}:GT={is_sub}/GM={pred_sub}"
            cv2.putText(overlay, label, (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        # also add scene ID and metrics
        cv2.putText(overlay, f"{r['scene_id']} P={r['precision']} R={r['recall']}", (5,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        thumbs.append(overlay)
    if not thumbs:
        return
    rows2 = [thumbs[i:i+2] for i in range(0,len(thumbs),2)]
    pad_rows = []
    for row in rows2:
        row = list(row)
        while len(row) < 2:
            row.append(np.zeros_like(row[0]))
        pad_rows.append(np.hstack(row))
    sheet = np.vstack(pad_rows)
    cv2.imwrite(str(out_dir / "smoke_comparison_contact_sheet.png"), sheet,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])

for idx, scene in enumerate(selected, 1):
    sid = scene["scene_id"]
    pool_name = scene["_pool"]
    cand_ids = scene["candidate_ids"]
    gt = gt_for(sid)

    print(f"[{idx}/10] {sid} ({pool_name})")

    # ── build annotated frames ──────────────────────────────────────────────────
    frames = []
    for fi in scene["frame_indices"]:
        frame_path = CACHE_DIR / f"{sid}_F{fi:05d}.jpg"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            frames.append(img)

    if not frames:
        err = f"No frames found for {sid}"
        print(f"  SKIP: {err}")
        errors_log.append({"scene_id": sid, "error": err, "type": "missing_frames"})
        scene_results.append({"scene_id": sid, "pool": pool_name, "error": err})
        continue

    # ── build prompt ─────────────────────────────────────────────────────────
    user_prompt = (
        f"Classify candidate IDs in the following frames (scene {sid}).\n"
        f"Candidate IDs: {json.dumps(cand_ids)}\n"
        f"Return exactly: {{\"track_id\":\"{sid}\",\"decisions\":["
        + ",".join(
            f'{{"candidate_id":"{cid}","is_subtitle":true/false,"category":"<category>","confidence":0.0-1.0,"transcription":null,"reason_code":"<short>"}}'
            for cid in cand_ids
        )
        + f'],"uncertain":true/false}}'
    )

    # ── encode frames ────────────────────────────────────────────────────────
    content_parts = [{"text": user_prompt}]
    total_bytes = 0
    dims = ""
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

    # ── API call ─────────────────────────────────────────────────────────────
    req = GeminiRequest(
        contents=[{"role": "user", "parts": content_parts}],
        model="gemini-3.5-flash-lite",
        generation_config={"temperature": 0.1, "maxOutputTokens": 2048},
        system_instruction=SYSTEM,
        request_id=sid,
        source_kind="smoke_subtitle_verifier",
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

        for d in decisions:
            if d.get("category") not in ALLOWED_CATS:
                d["category"] = "unknown"

    except Exception as exc:
        error_msg = str(exc)[:300]
        lat_ms = 0
        http_status = 0

    meta = RequestMeta(
        scene_id=sid, model="gemini-3.5-flash-lite",
        image_count=len(frames), image_dimensions=dims,
        total_image_payload_bytes=total_bytes,
        latency_ms=lat_ms, attempt_count=attempt,
        http_status=http_status, api_status=api_status,
        schema_valid=schema_valid, candidate_ids_valid=cand_valid,
        cache_hit=False
    )
    request_meta.append(asdict(meta))

    tp = fp = tn = fn = 0
    for cid in cand_ids:
        gt_r = gt.get(cid)
        if not gt_r:
            continue
        expected = gt_r["is_subtitle"]
        found = next((d for d in decisions if d.get("candidate_id") == cid), None)
        if found is None:
            if expected:
                fn += 1
            else:
                tn += 1
        else:
            predicted = bool(found.get("is_subtitle"))
            if predicted == expected:
                if expected:
                    tp += 1
                else:
                    tn += 1
            else:
                if expected:
                    fn += 1
                else:
                    fp += 1

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
                              "confidence": d.get("confidence")}
                             for d in decisions],
        "ground_truth": {cid: {"is_subtitle": gt[cid]["is_subtitle"],
                               "category": gt[cid]["category"]} for cid in cand_ids if cid in gt},
        "error": error_msg
    })

    print(f"  -> HTTP {http_status} | {api_status} | {lat_ms:.0f}ms | TP={tp} FP={fp} TN={tn} FN={fn} P={prec:.2f} R={rec:.2f}")

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

# smoke_scene_manifest.json
json.dump({"smoke_test":True,"selection_method":"stratified_random_seed42",
           "scenes":[{**s, "_pool": s["_pool"]} for s in selected]},
          (OUT_DIR / "smoke_scene_manifest.json").open("w",encoding="utf-8"), indent=2, ensure_ascii=False)

# smoke_ground_truth.jsonl
gt_lines = []
for s in selected:
    for cid in s["candidate_ids"]:
        r = gt_rows.get((s["scene_id"], cid))
        if r:
            gt_lines.append(json.dumps(r, ensure_ascii=False))
(OUT_DIR / "smoke_ground_truth.jsonl").open("w",encoding="utf-8").write("\n".join(gt_lines)+"\n")

# smoke_gemini_decisions.jsonl
dec_lines = [json.dumps({"scene_id":r["scene_id"],"decisions":r["gemini_decisions"]},ensure_ascii=False) for r in scene_results]
(OUT_DIR / "smoke_gemini_decisions.jsonl").open("w",encoding="utf-8").write("\n".join(dec_lines)+"\n")

# smoke_request_metadata_sanitized.jsonl
req_lines = [json.dumps(m, ensure_ascii=False) for m in request_meta]
(OUT_DIR / "smoke_request_metadata_sanitized.jsonl").open("w",encoding="utf-8").write("\n".join(req_lines)+"\n")

# smoke_errors.jsonl
err_lines = [json.dumps(e, ensure_ascii=False) for e in errors_log]
(OUT_DIR / "smoke_errors.jsonl").open("w",encoding="utf-8").write("\n".join(err_lines)+"\n")

# smoke_metrics.json
json.dump(metrics, (OUT_DIR / "smoke_metrics.json").open("w",encoding="utf-8"), indent=2, ensure_ascii=False)

# smoke_comparison_contact_sheet.png
_build_contact_sheet(scene_results, gt_rows, OUT_DIR)

print("\n=== SMOKE TEST COMPLETE ===")
print(json.dumps(metrics, indent=2))
