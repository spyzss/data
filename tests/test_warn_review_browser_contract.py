"""Real-browser contract for the canonical Warn review page.

The existing Node suites deliberately use a small fake DOM.  This file covers
the browser-only seams: real media decode, pointer hit testing, hover corridor,
native focus/keyboard delivery and continuous-overlay visibility.  It keeps
the report/service state real and does not mutate client state directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import pytest

from human_qc.http_server import create_http_server
from human_qc.warn_workbench_service import (
    FrameRangeDto,
    OverlayHandle,
    OverlaySegmentHandle,
    WarnWorkbenchService,
)
from qc_common.report import load_asset_qc_report, write_asset_qc_report
from qc_pipeline.context import AssetContext
from tests.qc_report_fixtures import make_v2_report


CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FPS = 30
TOTAL_FRAMES = 1800


class _QuietHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib hook.
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _run_ffmpeg(*args: str) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stderr


def _write_h264_video(path: Path, *, frames: int, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=96x64:r={FPS}",
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    )


def _issue(
    issue_id: str,
    display_name: str,
    start: int,
    end: int,
    *,
    continuous_sam3: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "issue_id": issue_id,
        "code": issue_id.replace("-", "_"),
        "display_name": display_name,
        "severity": "warn",
        "module": "sam3_containment" if continuous_sam3 else "video_quality",
        "issue_type": "metric_threshold",
        "metric": "fixture_metric",
        "operator": ">",
        "boundary_value": 0.5,
        "needs_manual_review": True,
        "start_frame": start,
        "end_frame_exclusive": end,
        "default_reason": issue_id,
    }
    if continuous_sam3:
        result["evidence_type"] = "sam3_continuous"
    return result


def _write_report(root: Path, asset_id: str, *, issues: list[dict[str, object]]) -> AssetContext:
    video = root / "video" / f"{asset_id}.mp4"
    _write_h264_video(video, frames=TOTAL_FRAMES, color="navy")
    report = make_v2_report(status="awaiting_external")
    report["asset_id"] = asset_id
    report["execution"]["profile"] = "supplier_evaluation"
    report["pipeline_state"] = {
        "status": "awaiting_external",
        "last_completed_module": "sam3_containment",
        "next_module": "manual_review",
        "stop_reason": None,
    }
    report["source_files"] = {"video": {"path": "video/%s.mp4" % asset_id}}
    report["issues"] = issues
    selected = [str(issue["issue_id"]) for issue in issues]
    report["manual_review"] = {
        "required": True,
        "state": "queued",
        "candidate_issue_ids": selected,
        "failures_for_batch_stats_issue_ids": [],
        "selected_issue_ids": selected,
        "selected_issue_id": selected[0],
        "issue_reviews": {},
        "completed_at": None,
    }
    report_path = root / "quality_archive" / f"{asset_id}.json"
    write_asset_qc_report(
        report_path, report, expected_revision=0, profile="supplier_evaluation"
    )
    return AssetContext(
        asset_id=asset_id,
        batch_root=root,
        report_path=report_path,
        source_files={"video": {"path": video.relative_to(root).as_posix()}},
        source_range=(0, TOTAL_FRAMES),
        metadata={"fps": FPS},
    )


class _SwitchingOverlayProvider:
    """A public provider seam, not a browser-side task mutation.

    The first task projection exposes ``generating``.  The first formal
    status poll promotes the same approved segment to ``ready`` so the browser
    proves its no-reload unlock path against an actual MP4.
    """

    def __init__(self, overlay_path: Path) -> None:
        self.overlay_path = overlay_path
        self.calls = 0

    def get_asset_overlays(self, asset_id: str, selected: tuple[Any, ...]):
        del asset_id
        self.calls += 1
        ready = self.calls >= 2
        segment = OverlaySegmentHandle(
            FrameRangeDto(142, 182),
            "ready" if ready else "generating",
            "fixture-overlay" if ready else None,
            self.overlay_path if ready else None,
        )
        return {
            item.issue_id: OverlayHandle(
                "ready" if ready else "generating", segments=(segment,)
            )
            for item in selected
        }

    def retry_asset_overlays(self, asset_id: str, selected: tuple[Any, ...]):
        return self.get_asset_overlays(asset_id, selected)


@pytest.fixture
def browser_workbench(tmp_path: Path):
    root = tmp_path / "browser-fixture"
    root.mkdir()
    overlap = _write_report(
        root,
        "asset-overlap",
        issues=[
            _issue("exposure", "曝光异常", 120, 169),
            _issue("shake-a", "画面抖动", 142, 182, continuous_sam3=True),
            _issue("shake-b", "画面抖动", 390, 427),
        ],
    )
    # Kept in the real queue to prove the page can navigate after a durable
    # explicit completion, while the browser scenario itself stays focused on
    # the overlap fixture.
    next_context = _write_report(
        root,
        "asset-z-next",
        issues=[_issue("next-warn", "曝光异常", 300, 340)],
    )
    overlay = root / "overlays" / "fixture-overlay.mp4"
    _write_h264_video(overlay, frames=40, color="orange")
    provider = _SwitchingOverlayProvider(overlay)
    service = WarnWorkbenchService(
        reviewer="browser-reviewer",
        asset_contexts={overlap.asset_id: overlap, next_context.asset_id: next_context},
        overlay_provider=provider,
        profile="supplier_evaluation",
    )
    server = create_http_server("127.0.0.1", 0, service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{server.server_port}/",
            "report_path": overlap.report_path,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_browser_scenario(base_url: str, tmp_path: Path) -> dict[str, object]:
    completed = _run_browser_process(base_url, tmp_path / "chrome")
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


class BrowserDriverTimeout(TimeoutError):
    def __init__(
        self,
        *,
        cleaned_after_sigterm: bool,
        returncode: int | None,
        stdout: str,
        stderr: str,
    ) -> None:
        state = "after SIGTERM cleanup" if cleaned_after_sigterm else "after SIGKILL escalation"
        super().__init__(f"browser driver timed out and exited {state}")
        self.cleaned_after_sigterm = cleaned_after_sigterm
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_browser_process(
    base_url: str,
    profile_dir: Path,
    *,
    timeout_seconds: float = 40,
    terminate_grace_seconds: float = 5,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Node without bypassing its SIGTERM child-reaping handler on timeout."""

    driver = Path(__file__).with_name("browser") / "cdp_driver.mjs"
    process = subprocess.Popen(
        ["node", str(driver), base_url, str(profile_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy() if environment is None else environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=terminate_grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise BrowserDriverTimeout(
                cleaned_after_sigterm=False,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            ) from None
        raise BrowserDriverTimeout(
            cleaned_after_sigterm=True,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        ) from None
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def test_outer_python_timeout_terminates_driver_before_killing_and_reaps_chrome_and_server(
    tmp_path: Path,
) -> None:
    """A parent timeout gives Node's SIGTERM cleanup a chance to reap Chrome."""

    sandbox = tmp_path / "outer-timeout"
    sandbox.mkdir()
    pid_path = sandbox / "fake-chrome.pid"
    fake_chrome = sandbox / "fake-chrome.py"
    fake_chrome.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import os, signal, sys, time",
                f"open({str(pid_path)!r}, 'w', encoding='utf-8').write(str(os.getpid()))",
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))",
                "while True: time.sleep(1)",
            )
        ),
        encoding="utf-8",
    )
    fake_chrome.chmod(0o755)
    server = HTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(BrowserDriverTimeout) as raised:
            _run_browser_process(
                f"http://127.0.0.1:{server.server_port}/",
                sandbox / "profile",
                timeout_seconds=1,
                terminate_grace_seconds=3,
                environment={**os.environ, "CHROME_BIN": str(fake_chrome)},
            )
        assert raised.value.cleaned_after_sigterm is True
        assert raised.value.returncode == 143
        chrome_pid = int(pid_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(chrome_pid, 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert thread.is_alive() is False


def test_real_browser_preserves_overlap_geometry_focus_and_overlay_boundaries(
    browser_workbench: dict[str, object], tmp_path: Path
) -> None:
    """Exercise actual CSS hit targets, native video and formal HTTP routes."""

    result = _run_browser_scenario(str(browser_workbench["base_url"]), tmp_path)

    assert result["asset_id"] == "asset-overlap"
    assert result["video_ready"] is True
    assert result["timeline_blocks"] == [
        {"start": 120, "end": 182, "label": False},
        {"start": 390, "end": 427, "label": False},
    ]
    assert result["popover_frames"] == [120, 142]
    assert result["popover_seek_frame"] == 142
    assert result["drag_frame"] == 390
    assert result["keyboard_frames"] == [143, 142, 142]
    assert result["blank_other_submission_blocked"] is True
    assert result["first_pass_issue"] == "exposure"
    assert result["saved_pass_marker"] is True
    assert result["overlay_boundaries"] == {
        "141": False,
        "142": True,
        "181": True,
        "182": False,
    }
    assert result["overlay_sync"] is True
    assert result["uncaught"] == []
    assert result["console_errors"] == []
    assert result["failed_network"] == []

    report = load_asset_qc_report(Path(browser_workbench["report_path"]))
    assert report is not None
    assert set(report["manual_review"]["issue_reviews"]) == {"exposure"}
    assert report["manual_review"]["issue_reviews"]["exposure"]["verdict"] == "pass"
    assert "shake-a" not in report["manual_review"]["issue_reviews"]
