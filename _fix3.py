import pathlib

p = pathlib.Path("tests/test_sweep_scripts.py")
src = p.read_text(encoding="utf-8")

old = """    # Parse last JSON line from stdout
    for line in reversed(result.stdout.strip().split("\\n")):
        line = line.strip()
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"status": "NO_OUTPUT", "error": result.stderr[:200] if result.stderr else ""}"""

new = """    # Find last JSON dict in stdout (skip progress lines)
    idx = result.stdout.rfind("\\n{")
    if idx >= 0:
        try:
            return json.loads(result.stdout[idx:])
        except json.JSONDecodeError:
            pass
    return {"status": "NO_OUTPUT", "error": result.stderr[:200] if result.stderr else ""}"""

src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("test_sweep_scripts.py parser fixed")
