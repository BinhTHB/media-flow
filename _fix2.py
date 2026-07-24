import pathlib

Q4 = chr(34) * 4
Q3 = chr(34) * 3

# Fix test_sweep_scripts.py
p = pathlib.Path("tests/test_sweep_scripts.py")
src = p.read_text(encoding="utf-8")
src = src.replace("without data." + Q4, "without data." + Q3)
src = src.replace("output." + Q4, "output." + Q3)
p.write_text(src, encoding="utf-8")
print("test_sweep_scripts.py fixed")

# Fix sweep_r3.py
p = pathlib.Path("scripts/sweep_r3.py")
src = p.read_text(encoding="utf-8")
src = src.replace("without frames." + Q4, "without frames." + Q3)
p.write_text(src, encoding="utf-8")
print("sweep_r3.py fixed")
