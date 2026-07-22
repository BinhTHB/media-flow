import sys, os, base64, json, hashlib
sys.path.insert(0, os.getcwd())
from pathlib import Path

from src.infrastructure.gemini.client import GeminiClient
from src.infrastructure.gemini.contracts import GeminiRequest, MODE_ROLE_PRIMARY
from src.interfaces.config.config_loader import ConfigLoader
from src.infrastructure.factory.provider_factory import ProviderFactory

loader = ConfigLoader()
config = loader.load()
factory = ProviderFactory(config=config.providers, gemini_config=config.gemini, gemini_api_key=loader.gemini_api_key)
provider = factory.create_ocr_provider("gemini_video_direct")

# Initialize client using runtime to resolve api_key
client = provider.runtime.get_client("gemini-3.5-flash-lite")

OUT = Path("output/review/gemini_direct_video_g5/sample2/g59/g510_forensic/")

W, H = 720, 1280
start_s, end_s = 19.583, 21.833
ts = {"A_200ms_before": max(0.0,start_s-0.200), "B_mid": (start_s+end_s)/2, "C_200ms_before_end": max(start_s,end_s-0.200)}

# ── SOURCE FRAME REQUEST ──
src_parts = [{"text": (
    "The input frames are original 720x1280 portrait short-video frames at three timestamps "
    "for event merged_0:e_14. "
    "Inspect only the main dialogue subtitle associated with this event. "
    "Do not confuse it with warning labels, creator notes, logos, account labels, "
    "stickers, UI overlays, or newly rendered Vietnamese replacement text.\n\n"
    "Return normalized coordinates on a 0-1000 scale using exactly: [y_min, x_min, y_max, x_max].\n\n"
    "Identify separately:\n"
    "1. source_text_bbox\n"
    "2. source_container_bbox\n"
    "3. recommended_cover_bbox\n"
    "4. visible_source_text\n"
    "5. nearby_unrelated_text\n"
    "6. overlay_type\n"
    "7. confidence\n\n"
    "Do not estimate a nearly full-width banner unless the original source frame "
    "visibly contains a subtitle-specific background spanning that width."
)}]
for label in ["A_200ms_before", "B_mid", "C_200ms_before_end"]:
    p = OUT / f"source_{label}.png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    src_parts.append({"text": f"frame_timestamp={ts[label]:.3f}s_label={label}"})
    src_parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

src_resp = client.generate_content(GeminiRequest(contents=[{"parts": src_parts}], source_kind="image_forensic"), use_cache=False)
print("=== FORENSIC SOURCE RESPONSE ===")
print(src_resp.text)
(OUT / "gemini_forensic_source_response.json").write_text(json.dumps({"text": src_resp.text, "model": src_resp.model, "usage": src_resp.usage.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")

# ── RENDERED-FRAME REQUEST ──
rend_parts = [{"text": (
    "Compare each rendered 720x1280 portrait frame against the original source "
    "frame at the same timestamp.\n\n"
    "Identify every visible text group and classify each as:\n"
    "- original source subtitle\n"
    "- newly rendered Vietnamese replacement\n"
    "- unrelated original overlay\n"
    "- warning label\n"
    "- logo/account/UI\n"
    "- uncertain\n\n"
    "Determine whether any source subtitle glyph belonging to merged_0:e_14 "
    "remains visible outside the black replacement cover.\n\n"
    "Do not label Vietnamese text as source leakage merely because it appears "
    "outside the cover. Verify its position and visual appearance against the "
    "original frame.\n\n"
    "Return a JSON object with keys:\n"
    "source_leakage, leaked_text, leakage_bbox, duplicate_replacement,\n"
    "replacement_overflow, unrelated_overlay_confused_as_subtitle, evidence."
)}]
for label in ["A_200ms_before", "B_mid", "C_200ms_before_end"]:
    src = OUT / f"source_{label}.png"
    rend = OUT / f"rendered_{label}.png"
    for name, path in [("source", src), ("rendered", rend)]:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        rend_parts.append({"text": f"frame_timestamp={ts[label]:.3f}s_label={name}"})
        rend_parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

rend_resp = client.generate_content(GeminiRequest(contents=[{"parts": rend_parts}], source_kind="image_forensic"), use_cache=False)
print("\n=== FORENSIC RENDERED RESPONSE ===")
print(rend_resp.text)
(OUT / "gemini_forensic_rendered_response.json").write_text(json.dumps({"text": rend_resp.text, "model": rend_resp.model, "usage": rend_resp.usage.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nDONE")
