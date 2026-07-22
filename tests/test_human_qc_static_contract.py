from __future__ import annotations

from pathlib import Path
import re
import subprocess


STATIC = Path(__file__).parents[1] / "human_qc" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_human_static_bundle_is_warn_only() -> None:
    source = "\n".join(_read(name) for name in ("app.js", "index.html", "warn_adapter.js"))
    assert "semantic_adapter" not in source
    assert "SemanticCalibration" not in source
    assert "completeSemantic" not in source
    assert "语义" not in source
    assert "共享边界" not in source
    assert "相邻任务之间的" not in source
    assert 'data-action="verdict-pass"' in _read("warn_adapter.js")
    assert 'data-action="verdict-fail"' in _read("warn_adapter.js")


def test_machine_issue_fields_remain_read_only() -> None:
    warn = _read("warn_adapter.js")
    assert "data-machine-reason" in warn
    assert "data-machine-metrics" in warn
    assert "data-machine-threshold" in warn
    assert not re.search(r"<(?:input|textarea)[^>]+data-machine-", warn)
    assert "issue.severity =" not in warn
    assert "issue.observed_value =" not in warn


def test_warn_node_contracts_pass() -> None:
    result = subprocess.run(
        ["node", "--test", *sorted(str(path) for path in STATIC.glob("*.test.mjs"))],
        cwd=STATIC.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
