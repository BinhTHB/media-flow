path = "E:\\projects\\media-flow\\scripts\\build_round26_evidence.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
old = "ledger_findings.get('crop_classifier_ledger')"
new = "ledger_findings.get('crop_classifier_ledger.jsonl')"
content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed:", content.count(new), "occurrences")
