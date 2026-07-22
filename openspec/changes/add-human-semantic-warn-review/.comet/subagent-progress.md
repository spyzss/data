# Comet Subagent Progress

## Completed

- Task 1: report schema, migration compatibility, write boundary, and count projection complete in `57f8ab1..b01bcf1`; progress commit `133701c`.
- Task 2: Warn verdict, failure-reason audit, and all-reviewed/early-fail state machine complete in `d62a443..192625e`; progress commit `a0471a7`.
- Task 3: persisted manual eligibility, manual-before-semantic routing, profile compatibility, and bypass closure complete in `427d834..13d11fe`; progress commit `775309c`.
- Task 4: independent semantic service split, Warn-only Human cleanup, guarded video/range support, queue/lease hardening, and final route semantics complete in `82b2299..365e755`; final targeted re-review approved.
- OpenSpec checked off: 7.1, 7.2, 7.4, 8.1, 8.2, and 8.4. OpenSpec 7.3 remains pending for aggregator integration.

## Current Task

- Plan task: `Task 4A: 完成 early-fail 未查看 Warn 的正式批次投影`
- OpenSpec mapping: `7.3 更新批次投影，仅统计实际 issue review，并单独统计 early-fail 后未查看 Warn`
- Stage: `implementing`
- Review mode: `thorough`
- Review/fix round: `0/2`
- Implementer commits: pending
- Changed files: pending — formal `qc_reporting` projection, aggregate, export, and their focused tests only.
- RED evidence: pending
- GREEN evidence: pending
- Review result: pending
- Binding downstream decisions:
  - Count only actual `issue_reviews`; early-fail unviewed selected Warns are a separate terminal metric, never implied Pass/Fail verdicts.
  - Derive the metric from persisted selected IDs, completion mode, current report revision, terminal pipeline state, and current machine-Warn issues.
  - Keep all formal aggregate formats and actual CLI snapshot output consistent; no Feishu/Lark integration is in scope.
