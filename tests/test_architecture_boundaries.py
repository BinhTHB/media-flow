import ast
import os
import pytest

FORBIDDEN_INFRASTRUCTURE_PACKAGES = {
    "whisper",
    "faster_whisper",
    "paddleocr",
    "openai",
    "google",
    "ffmpeg",
    "moviepy",
    "cv2"
}

def get_all_python_files(dir_path):
    py_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def check_imports_in_file(file_path, forbidden_prefixes, forbidden_external):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return [f"Syntax error in file: {file_path}"]

    violations = []
    for node in ast.walk(tree):
        imported_modules = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)

        for mod in imported_modules:
            # Check for local forbidden layers
            for prefix in forbidden_prefixes:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"Forbidden local import '{mod}'")

            # Check for forbidden external packages
            mod_parts = mod.split(".")
            root_mod = mod_parts[0]
            if root_mod in forbidden_external:
                violations.append(f"Forbidden infrastructure import '{mod}'")

    return violations

def test_domain_boundary():
    domain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/domain"))
    py_files = get_all_python_files(domain_dir)

    forbidden_local = ["src.application", "src.infrastructure", "src.interfaces"]
    all_violations = {}
    for f in py_files:
        violations = check_imports_in_file(f, forbidden_local, FORBIDDEN_INFRASTRUCTURE_PACKAGES)
        if violations:
            all_violations[os.path.relpath(f)] = violations

    assert not all_violations, f"Architecture boundary violations in domain layer: {all_violations}"

def test_application_boundary():
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/application"))
    py_files = get_all_python_files(app_dir)

    forbidden_local = ["src.infrastructure", "src.interfaces"]
    all_violations = {}
    for f in py_files:
        violations = check_imports_in_file(f, forbidden_local, FORBIDDEN_INFRASTRUCTURE_PACKAGES)
        if violations:
            all_violations[os.path.relpath(f)] = violations

    assert not all_violations, f"Architecture boundary violations in application layer: {all_violations}"
