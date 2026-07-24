"""
Generate GT V2.3 from V2.2 + amendment.
5 mislabels become negative (keep entries, flip is_subtitle).
Output: artifacts/round28/gt_v2_3.json + sidecar SHA256 record.
"""
import json, hashlib, copy
from pathlib import Path

GT_V2_2 = Path("artifacts/round26/gt_v2_2.json")
AMENDMENT = Path("artifacts/round28/gt_amendment.json")
OUT = Path("artifacts/round28/gt_v2_3.json")
SIDECAR = Path("artifacts/round28/gt_v2_3_sidecar.json")

# Load original
gt = json.loads(GT_V2_2.read_text(encoding="utf-8"))
orig_sha = hashlib.sha256(GT_V2_2.read_bytes()).hexdigest()
print(f"GT V2.2 SHA256: {orig_sha}")

# Load amendment
amend = json.loads(AMENDMENT.read_text(encoding="utf-8"))
# Build set of mislabels
mislabels = {e["id"]: e for e in amend["entries"]}
print(f"Amendment entries ({len(amend['entries'])}): {list(mislabels.keys())}")

# Verify each mislabel target exists in original GT
gt_idx = {e["id"]: i for i, e in enumerate(gt["entries"])}
for eid in mislabels:
    assert eid in gt_idx, f"Mislabel entry {eid} not found in GT V2.2!"

# Build V2.3
v23 = copy.deepcopy(gt)
v23["version"] = "2.3"
v23["based_on"] = "2.2"
v23["based_on_sha256"] = orig_sha
v23["amendment"] = "artifacts/round28/gt_amendment.json"

changes = []
for eid, info in mislabels.items():
    idx = gt_idx[eid]
    old = v23["entries"][idx]
    old["is_subtitle"] = False
    old["_mislabel_reason"] = info["reason"]
    old["_mislabel_evidence"] = info["crop_ref"]
    changes.append({
        "id": eid,
        "was_subtitle": True,
        "now_subtitle": False,
        "reason": info["reason"],
        "crop_ref": info["crop_ref"]
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
v23_text = json.dumps(v23, indent=2, ensure_ascii=False)
OUT.write_text(v23_text, encoding="utf-8")
new_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()

sidecar = {
    "version": "2.3",
    "based_on": {"version": "2.2", "sha256": orig_sha, "path": str(GT_V2_2)},
    "sha256": new_sha,
    "amendment_path": str(AMENDMENT),
    "changes": changes,
    "total_entries": len(v23["entries"]),
    "positive_count": len([e for e in v23["entries"] if e["is_subtitle"]]),
    "negative_count": len([e for e in v23["entries"] if not e["is_subtitle"]]),
}
SIDECAR.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

print(f"GT V2.3 written: {OUT}")
print(f"  SHA256: {new_sha}")
print(f"  Positives: {sidecar['positive_count']}")
print(f"  Negatives: {sidecar['negative_count']}")
print(f"  Changes applied: {len(changes)}")
for c in changes:
    print(f"    {c['id']}: {c['reason']}")
