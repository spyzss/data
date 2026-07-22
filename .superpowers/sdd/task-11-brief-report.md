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
