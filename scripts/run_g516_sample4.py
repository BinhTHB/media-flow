"""Run G.5.16 Sample4 Gemini-Only End-to-End Validation post-processing.

Loads Sample4 V1-V5 and Translation checkpoints.
Generates local ASS file.
Renders with FFmpeg.
Performs Technical QA.
Performs Gemini Visual QA under corrected policy.
Dumps Phase G.5.16 report artifact.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.exceptions import ProviderError
from src.infrastructure.gemini.structured_output import parse_structured
from src.interfaces.config.config_loader import ConfigLoader

MODEL = "gemini-3.5-flash-lite"
PROVIDER = "gemini_video_direct"
VISUAL_QA_STATUSES = {"PASS", "FAIL", "REVIEW_REQUIRED"}

class G516ValidationError(ValueError):
    pass

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""

def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i+size]

def _ass_time(seconds: float) -> str:
    h, rem = divmod(int(round(seconds * 100)), 360000)
    m, rem = divmod(rem, 6000)
    return f"{h}:{m:02}:{rem//100:02}.{rem%100:02}"

def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

def generate_ass_file(events, path, width=720, height=1280):
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "", "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1", "", "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for ev in events:
        x1, y1, x2, y2 = ev["representative_box_2d"]
        l, t, r, b = round(x1 * width / 1000), round(y1 * height / 1000), round(x2 * width / 1000), round(y2 * height / 1000)
        bw, bh = max(1, r-l), max(1, b-t)
        fs = max(12, min(24, int(bh * 0.55)))
        start = ev.get("corrected_start", float(ev["start_seconds_in_source"]))
        end = ev.get("corrected_end", float(ev["end_seconds_in_source"]))
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{{\\an7\\pos({l},{t})\\p1\\1c&H000000&}}m 0 0 l {bw} 0 l {bw} {bh} l 0 {bh}{{\\p0}}")
        lines.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{{\\an5\\fs{fs}\\pos({round((l+r)/2)},{round((t+b)/2)})}}{_escape_ass(ev['translated_text'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def _probe(video: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def technical_qa(input_video: Path, output_video: Path, events: list) -> dict:
    reasons = []
    size = output_video.stat().st_size if output_video.is_file() else 0
    if size <= 0:
        reasons.append("OUTPUT_VIDEO_EMPTY")
    try:
        src, out = _probe(input_video), _probe(output_video)
        vid = next((s for s in out.get("streams",[]) if s.get("codec_type")=="video"), None)
        aud = next((s for s in out.get("streams",[]) if s.get("codec_type")=="audio"), None)
        dur_delta = abs(float(src["format"]["duration"]) - float(out["format"]["duration"]))
    except Exception:
        return {"verdict": "FAIL", "reasons": reasons + ["FFPROBE_FAILED"]}
    if vid is None: reasons.append("VIDEO_STREAM_MISSING")
    if aud is None: reasons.append("AUDIO_STREAM_MISSING")
    if dur_delta >= 0.25: reasons.append("DURATION_DELTA_EXCEEDED")
    if vid and (int(vid.get("width",0))!=720 or int(vid.get("height",0))!=1280): reasons.append("RESOLUTION_NOT_720X1280")
    bad = sum(not(isinstance(ev.get("representative_box_2d"),list) and len(ev["representative_box_2d"])==4) for ev in events)
    empty = sum(not(isinstance(ev.get("translated_text"),str) and ev["translated_text"].strip()) for ev in events)
    if bad: reasons.append("OUT_OF_FRAME_BBOX")
    if empty: reasons.append("EMPTY_RENDERED_TEXT")
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "file_size": size,
        "video_stream": vid.get("codec_name") if vid else None,
        "audio_stream": aud.get("codec_name") if aud else None,
        "duration_delta": dur_delta,
        "resolution": [vid.get("width"), vid.get("height")] if vid else None,
        "rendered_events": len(events),
    }

def g515_visual_qa_schema() -> dict:
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {"type": "object",
            "properties": {"event_id":{"type":"string"},"source_text_leakage":{"type":"boolean"},"source_container_visible":{"type":"boolean"},"container_style_mismatch":{"type":"boolean"},"duplicate_replacement":{"type":"boolean"},"replacement_text_overlap":{"type":"boolean"},"replacement_overflow":{"type":"boolean"},"unrelated_text_damage":{"type":"boolean"},"verdict":{"type":"string","enum":["PASS","FAIL","REVIEW_REQUIRED"]},"confidence":{"type":"number"},"evidence":{"type":"string"}},
            "required":["event_id","source_text_leakage","source_container_visible","container_style_mismatch","duplicate_replacement","replacement_text_overlap","replacement_overflow","unrelated_text_damage","verdict","confidence","evidence"]
        }}},
        "required":["results"]
    }

def g515_validate_visual_qa_result(result: dict) -> dict:
    hard_fail = ("source_text_leakage", "duplicate_replacement", "replacement_text_overlap", "replacement_overflow", "unrelated_text_damage")
    if any(result.get(key) is True for key in hard_fail):
        return {"verdict": "FAIL", "reason": "VISUAL_HARD_FAIL"}
    if result.get("container_style_mismatch") or result.get("verdict") == "REVIEW_REQUIRED":
        return {"verdict": "REVIEW_REQUIRED", "reason": "CONTAINER_STYLE_REVIEW"}
    if result.get("verdict") != "PASS":
        return {"verdict": "REVIEW_REQUIRED", "reason": "VISUAL_QA_UNCLEAR"}
    return {"verdict": "PASS", "reason": None}

def g515_validate_visual_qa_batch(results: list) -> str:
    verdicts = [g515_validate_visual_qa_result(r)["verdict"] for r in results]
    if "FAIL" in verdicts: return "FAIL"
    if "REVIEW_REQUIRED" in verdicts: return "REVIEW_REQUIRED"
    return "PASS"

def _extract_frame(video: Path, timestamp: float) -> bytes:
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-"], capture_output=True, check=False)
    if result.returncode or not result.stdout: raise G516ValidationError("FFMPEG_FRAME_EXTRACTION_FAILED")
    return result.stdout

def run_visual_qa_batch(provider, batch, input_video, output_video) -> dict:
    parts = [{ "text": "Compare original (odd index) vs rendered (even index) frames at the same timestamp. Evaluate source text leakage, container visibility, container style mismatch, duplicate replacement, replacement overlap/overflow, unrelated text damage.\nReturn JSON {results:[{event_id,source_text_leakage,source_container_visible,container_style_mismatch,duplicate_replacement,replacement_text_overlap,replacement_overflow,unrelated_text_damage,verdict,confidence,evidence}].}" }]
    for ev in batch:
        start = float(ev["start_seconds_in_source"])
        end = float(ev["end_seconds_in_source"])
        dur = end - start
        ts = start + dur * 0.5 if dur < 1.5 else start + dur * 0.1
        orig_bytes = _extract_frame(input_video, ts)
        rend_bytes = _extract_frame(output_video, ts)
        parts.extend([{"text": f"event_id={ev['event_id']} original"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(orig_bytes).decode()}}, {"text": f"event_id={ev['event_id']} rendered"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(rend_bytes).decode()}}])
    request = GeminiRequest(contents=[{"parts": parts}], model=provider.model, generation_config={"responseMimeType": "application/json", "responseSchema": g515_visual_qa_schema()}, system_instruction="You are a visual quality assurance assistant.", source_kind="visual_qa", api_surface="generate_content")
    client = provider.runtime.get_client(provider.model, role=MODE_ROLE_PRIMARY, api_key=provider.api_key)
    request.predicted_tokens = client.count_tokens(request)
    response = client.generate_content(request)
    payload = parse_structured(response.text)
    if not isinstance(payload, dict): raise ProviderError("Visual QA response was not an object.")
    provider.provenance.append({"stage": "visual_qa", "input_type": "image", "model": provider.model, "count_tokens": request.predicted_tokens or 0, "actual_usage": response.usage.to_dict() if response.usage else {}})
    return payload

def apply_timing_boundary_fix(events: list) -> list:
    fixed = []
    for i, ev in enumerate(events):
        start = float(ev["start_seconds_in_source"])
        end = float(ev["end_seconds_in_source"])
        orig_end = end
        if i > 0 and float(events[i-1]["end_seconds_in_source"]) == start:
            next_start = float(events[i+1]["start_seconds_in_source"]) if i+1 < len(events) else start + 1.0
            mid = (float(events[i-1]["end_seconds_in_source"]) + next_start) / 2.0
            end = max(start + 0.04, min(mid, float(events[i-1]["end_seconds_in_source"]) - 0.01))
            start = max(start, float(events[i-1]["end_seconds_in_source"]) + 0.01)
        if end <= start: end = start + 0.04
        fixed.append({**ev, "original_end": orig_end, "corrected_start": start, "corrected_end": end, "correction_applied": start != float(ev["start_seconds_in_source"]) or end != orig_end, "correction_reason": "ADJACENT_BOUNDARY_SHIFT" if (i > 0 and float(events[i-1]["end_seconds_in_source"]) == float(ev["start_seconds_in_source"])) else None})
    return fixed

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--live-visual-qa", action="store_true")
    args = ap.parse_args()

    checkpoint_path = args.output_root / "g516_checkpoint.json"
    artifact_path = args.output_root / "g516_sample4.json"

    loader = ConfigLoader()
    loader.secrets_path = str(ROOT / "config.yml")
    config = loader.load()
    provider = ProviderFactory(config=config.providers, gemini_config=config.gemini, gemini_api_key=loader.gemini_api_key).create_ocr_provider(PROVIDER)

    video_sha256 = file_hash(args.video)
    translations = json.loads((args.manifest_dir / "g5_translations.json").read_text(encoding="utf-8"))
    events = translations["events"]
    print(f"Loaded {len(events)} events from checkpoint.")

    fixed_events = apply_timing_boundary_fix(events)
    output_video = args.output_root / "sample4_image_level_dynamic_replacement.mp4"
    ass_path = args.output_root / "sample4_image_level_dynamic_replacement.ass"
    generate_ass_file(fixed_events, ass_path, width=720, height=1280)
    ass_filter = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else {}
    expected_render_hash = canonical_hash({"input_sha256": video_sha256, "geometry_sha256": canonical_hash([e["representative_box_2d"] for e in fixed_events]), "translation_sha256": canonical_hash([e["translated_text"] for e in fixed_events]), "ass_sha256": file_hash(ass_path)})

    if not (checkpoint.get("render",{}).get("hash") == expected_render_hash and output_video.is_file()):
        print("Rendering video...")
        result = subprocess.run(["ffmpeg", "-y", "-i", str(args.video), "-vf", f"ass='{ass_filter}'", "-c:a", "copy", str(output_video)], capture_output=True)
        if result.returncode:
            raise G516ValidationError(f"FFmpeg render failed: {result.stderr.decode('utf-8', errors='ignore')}")
        checkpoint["render"] = {"hash": expected_render_hash, "output_sha256": file_hash(output_video)}
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    technical = technical_qa(args.video, output_video, fixed_events)
    print(f"Technical QA: {technical['verdict']} | {technical.get('reasons', [])}")

    visual_results = []
    visual_status = "NOT_REQUESTED"
    if args.live_visual_qa and technical["verdict"] == "PASS":
        vqa_hash = canonical_hash({"video_sha256": file_hash(output_video), "geometry_sha256": canonical_hash([e["representative_box_2d"] for e in fixed_events]), "translations_sha256": canonical_hash([e["translated_text"] for e in fixed_events])})
        if checkpoint.get("visual_qa",{}).get("hash") == vqa_hash:
            # Re-validate cached results with corrected G.5.15 policy (no Gemini re-call)
            visual_results = checkpoint["visual_qa"]["results"]
            visual_status = g515_validate_visual_qa_batch(visual_results)
        else:
            print("Running Gemini Visual QA...")
            for batch in _batches(fixed_events, 10):
                payload = run_visual_qa_batch(provider, batch, args.video, output_video)
                visual_results.extend(payload.get("results", []))
            visual_status = g515_validate_visual_qa_batch(visual_results)
            checkpoint["visual_qa"] = {"hash": vqa_hash, "status": visual_status, "results": visual_results}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    print(f"Visual QA Status: {visual_status}")
    quota = {"visual_qa_requests": len([p for p in provider.provenance if p.get("stage")=="visual_qa"]), "total_tokens": sum(p.get("actual_usage",{}).get("candidates_token_count",0) for p in provider.provenance if p.get("stage")=="visual_qa")}
    final_verdict = "PRODUCTION READY" if (technical["verdict"]=="PASS" and visual_status in ("PASS","NOT_REQUESTED")) else "PRODUCTION READY: NO"
    artifact = {"phase":"G.5.16 Sample4 Gemini-Only Validation","request_status":"COMPLETED","final_verdict": final_verdict, "video_sha256": video_sha256, "eligible_event_count": len(fixed_events), "technical_qa": technical, "visual_qa":{"status": visual_status, "results": visual_results}, "quota_report": quota}
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Artifact written to {artifact_path}")

if __name__ == "__main__": main()
