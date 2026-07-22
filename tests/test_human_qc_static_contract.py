from __future__ import annotations

from pathlib import Path
import subprocess


STATIC = Path(__file__).parents[1] / "human_qc" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_human_static_bundle_is_canonical_warn_only() -> None:
    assert not (STATIC / "warn_adapter.js").exists()
    source = "\n".join(
        _read(name)
        for name in (
            "app.js",
            "index.html",
            "review_panel.js",
            "warning_timeline.js",
            "video_controller.js",
        )
    )
    for forbidden in (
        "semantic_adapter",
        "SemanticCalibration",
        "completeSemantic",
        "/api/assets/",
        "acquire-lease",
        "获取编辑锁",
        "warn_adapter",
        "data-action=\"step-back\"",
        "data-action=\"step-forward\"",
    ):
        assert forbidden not in source
    assert "/api/warn/assets" in _read("app.js")
    assert "Warn 复核" in _read("index.html")


def test_warn_layout_has_one_current_frame_and_required_bottom_action_order() -> None:
    html = _read("index.html")
    assert html.count("data-current-frame") == 1
    assert html.index("data-video-root") < html.index("data-warning-timeline")
    assert html.index("data-warning-timeline") < html.index("data-review-panel")
    assert html.index("data-review-panel") < html.index("data-bottom-navigation")
    assert "data-action=\"rate-decrease\"" in html
    assert "data-action=\"rate-increase\"" in html
    assert "data-overlay-video" in html
    assert "aria-hidden=\"true\"" in html

    panel = _read("review_panel.js")
    assert panel.index('data-action="verdict-pass"') < panel.index('data-action="complete-review"')
    assert "data-action=\"verdict-fail\"" in panel
    assert "data-reason-other" in panel
    assert "required" in panel
    assert "data-threshold-tooltip" in panel
    assert "data-passed-marker" in panel
    assert "aria-pressed" in panel
    assert "aria-describedby" in panel
    assert "canSubmitIssue" in panel

    app = _read("app.js")
    assert "canSubmitIssue" in app
    assert "overlay-status" not in app
    assert "setInterval" not in app


def test_static_contracts_pass() -> None:
    result = subprocess.run(
        ["node", "--test", *sorted(str(path) for path in STATIC.glob("*.test.mjs"))],
        cwd=STATIC.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
