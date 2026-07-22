from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "semantic_calibration" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_semantic_static_bundle_contains_no_warn_or_evidence_domain() -> None:
    source = "\n".join(_read(name) for name in ("index.html", "app.js", "semantic_adapter.js"))
    for forbidden in ("warn_adapter", "WarnReview", "verdict-pass", "verdict-fail", "SAM3", "/evidence/"):
        assert forbidden not in source
    assert "/api/semantic/assets" in _read("app.js")
    assert "semantic_adapter.js" in _read("index.html")


def test_semantic_page_contains_reviewer_video_timeline_and_text_calibration_ui() -> None:
    html = _read("index.html")
    adapter = _read("semantic_adapter.js")
    app = _read("app.js")
    assert "data-reviewer-input" in html
    assert "data-action=\"start-calibration\"" in html
    assert "<video" in html and "data-semantic-video" in html
    assert "data-current-frame" in html
    assert "timeline-track" in adapter
    assert "semantic-text-slot" in adapter
    assert "lease/acquire" in app
    assert "lease/renew" in app
    assert "lease/release" in app


def test_semantic_node_contracts_pass() -> None:
    result = subprocess.run(
        ["node", "--test", *sorted(str(path) for path in STATIC.glob("*.test.mjs"))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
