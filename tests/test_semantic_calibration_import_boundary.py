from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _forbidden_imports(package: str, forbidden: str) -> list[str]:
    violations: list[str] = []
    for path in sorted((ROOT / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == forbidden or module.startswith(f"{forbidden}."):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                is_dynamic = (
                    isinstance(function, ast.Name) and function.id == "__import__"
                ) or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                )
                first = node.args[0]
                if (
                    is_dynamic
                    and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and (first.value == forbidden or first.value.startswith(f"{forbidden}."))
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return violations


def test_semantic_and_warn_packages_have_zero_runtime_imports_between_them() -> None:
    assert (ROOT / "semantic_calibration").is_dir()
    assert _forbidden_imports("semantic_calibration", "human_qc") == []
    assert _forbidden_imports("human_qc", "semantic_calibration") == []


def test_semantic_only_modules_are_not_left_as_human_compatibility_shims() -> None:
    old = {
        "contracts.py",
        "hdf5_commit.py",
        "semantic_service.py",
        "source_adapters.py",
        "timeline.py",
    }
    assert old.isdisjoint({path.name for path in (ROOT / "human_qc").glob("*.py")})

    source = (ROOT / "human_qc" / "__init__.py").read_text(encoding="utf-8")
    for symbol in (
        "SemanticCalibrationService",
        "SharedBoundaryTimeline",
        "Hdf5ScalarJsonSubtaskAdapter",
        "BoundaryEditRequest",
    ):
        assert symbol not in source


def test_formal_human_server_and_launcher_are_warn_only() -> None:
    source = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "human_qc/http_server.py",
            "human_qc/workbench_service.py",
            "tools/serve_human_qc_workbench.py",
        )
    )
    assert "SemanticCalibrationService" not in source
    assert "semantic_service" not in source
    assert "/semantic/" not in source


def test_semantic_launcher_imports_only_semantic_and_neutral_packages() -> None:
    launcher = ROOT / "tools" / "serve_semantic_calibration.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"), filename=str(launcher))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert any(
        module == "semantic_calibration" or module.startswith("semantic_calibration.")
        for module in imports
    )
    assert not any(module == "human_qc" or module.startswith("human_qc.") for module in imports)
