import subprocess, json, sys
from pathlib import Path

ROOT = Path(".").resolve()

for name in ["sweep_r1.py", "sweep_r2.py", "sweep_r3.py", "sweep_step7.py"]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name)],
        capture_output=True, text=True, timeout=30,
        cwd=str(ROOT),
    )
    raw = result.stdout
    idx = raw.rfind("\n{")
    if idx >= 0:
        json_str = raw[idx:].strip()
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            parsed = "JSON_ERROR"
    else:
        parsed = "NO_DICT_FOUND"

    print(f"=== {name} ===")
    print(f"  idx={idx}, type={type(parsed).__name__}")
    if isinstance(parsed, dict):
        print(f"  status={parsed.get('status')}")
    elif isinstance(parsed, (int, float, str)):
        print(f"  value={parsed}")
    else:
        print(f"  parsed={parsed}")
    print()
