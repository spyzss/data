# Task 11 browser contract report

## Scope

Added the real-browser acceptance contract for the canonical Warn review UI,
then corrected only browser-observed integration seams in the static client.

## RED to GREEN evidence

The new contract uses local Chrome through CDP, a real 30 fps / 1,800-frame
H.264 source MP4, formal Warn HTTP routes, and a real report/service fixture.
Its initial run exposed four browser-only failures:

1. An overlap popover covered the draggable playhead and a row pointer event
   also started a free-track drag.
2. The review shell clipped the popover while moving the pointer from the block
   into the chooser.
3. Native video focus superseded the root after pointerdown, so arrows did not
   receive the declared video-region keyboard ownership.
4. Browser-normalized absolute `HTMLVideoElement.src` did not equal the
   canonical relative overlay route, so a ready overlay stayed hidden.

The contract is now GREEN: it hovers the overlap block, crosses into the
popover, chooses the second warning and seeks exactly to frame 142; drags the
playhead exactly to frame 390; checks video-only arrow stepping and input
non-stepping; validates blank Other fail locally without a mutation request;
persists the earliest active warning as Pass; and verifies continuous SAM3
overlay visibility at `[142, 182)` plus playback-rate synchronization.

## Minimal integration corrections

- Place overlap popovers under their source track with a hover corridor;
  preserve the freely draggable playhead and prevent chooser pointerdown from
  becoming a track drag.
- Keep the timeline panel visible above its surrounding clipped shell.
- Reassert the video-root focus on pointerup after native video focus default.
- Compare overlay routes by normalized URL rather than raw relative-vs-absolute
  source strings.
- Use a data favicon to avoid a real `/favicon.ico` 404 in the browser run.

## Durable/recovery coverage

- Warn facade covers Pass -> modified Fail -> early-fail and verifies only the
  reviewed warning is written, the manual reason (including Other) persists,
  audit says `resubmitted`, semantic is skipped, and the pipeline stops.
- Restarting the real Warn facade restores saved review state.  Stale revision
  and expired-lease writes both leave report bytes unchanged.
- Active release config is verified to order `sam3_containment`,
  `manual_review`, then `semantic_consistency`, with the active config's
  immutable snapshot validation performed by the config loader.

## Verification

```text
CHROME_BIN='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  .venv/bin/python -m pytest -q \
  tests/test_warn_review_browser_contract.py \
  tests/test_human_qc_end_to_end.py \
  tests/test_human_qc_profile_routing.py \
  tests/test_human_qc_recovery.py \
  tests/test_human_qc_workbench.py
# 64 passed in 5.84s

node --test human_qc/static/*.test.mjs
# 61 passed

git diff --check
# clean
```

## Final review hardening

The browser driver now gives every CDP command a configurable 5--10 second
deadline (`HUMAN_QC_CDP_TIMEOUT_MS`, default 8 seconds), awaits WebSocket
close, and supervises Chrome through SIGTERM grace followed by SIGKILL when it
has not actually exited.  SIGTERM/SIGINT invoke the same idempotent cleanup
before the Node process exits.  `CHROME_BIN` is preferred; absent that, the
driver diagnoses the macOS and common Linux Chrome/Chromium locations it
checked.  Network health also records `Network.loadingFailed` through a
request-id-to-URL map: canceled media/navigation are reported separately while
actual transport or blocked failures fail the browser health assertion.

New `tests/browser/cdp_driver.test.mjs` exercises deadline cleanup with
controlled Chrome/server children, TERM-to-KILL escalation, environment
override/discovery, and failure/cancellation classification.  It also starts
the real driver with a controlled executable browser and sends a POSIX SIGTERM
from its parent; the driver exits only after the launched browser PID is gone.

```text
Chrome browser contract, run 1: 1 passed in 2.29s
Chrome browser contract, run 2: 1 passed in 1.77s
Focused Python suite: 64 passed in 3.55s
Node static + driver suite: 67 passed
git diff --check: clean
```

## Active-CDP shutdown repair

The signal path now calls cleanup with `skipCdpClose`: it reaps only the
isolated Chrome child and exits, instead of waiting for a potentially stalled
DevTools WebSocket close.  Normal cleanup starts child reaping and CDP close
in parallel, so an ordinary slow close cannot defer ownership cleanup either.
The outer Python grace remains bounded at five seconds; it is no longer asked
to cover the 8--10 second CDP close budget.

The active-CDP regression launches an isolated real Chrome, confirms a live
CDP connection, injects a ten-second close stall, and then triggers a
five-second outer Python timeout.  It proves the driver still exits after
SIGTERM cleanup (143), the isolated Chrome PID is gone, and the HTTP server
thread is stopped.  Unit coverage additionally verifies that normal cleanup
reaps before a stalled close settles and that the signal variant never calls
close.

```text
Chrome contract including both outer-timeout regressions, run 1: 3 passed in 11.03s
Chrome contract including both outer-timeout regressions, run 2: 3 passed in 10.91s
Focused Python suite: 66 passed in 12.57s
Node static + driver suite: 68 passed
git diff --check: clean
```

## Outer Python timeout repair

The browser contract no longer uses `subprocess.run(..., timeout=...)`, whose
POSIX timeout path kills the Node driver before its `finally` and SIGTERM
handler can reap Chrome.  `_run_browser_process` now uses `Popen` plus
`communicate`: on timeout it sends Node SIGTERM and waits a bounded grace
period for the driver's cleanup; only an unresponsive driver is SIGKILLed.

The regression test starts an actual driver with a controlled executable fake
Chrome and a live HTTP server thread.  It triggers the outer Python timeout,
asserts the driver exits after SIGTERM cleanup (not parent SIGKILL), verifies
the fake Chrome PID no longer exists, then shuts down and joins the server
thread.

```text
Chrome contract including the outer-timeout regression, run 1: 2 passed in 3.71s
Chrome contract including the outer-timeout regression, run 2: 2 passed in 3.47s
Focused Python suite: 65 passed in 5.25s
Node static + driver suite: 67 passed
git diff --check: clean
```
