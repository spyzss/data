# Comet Subagent Progress

- Change: `add-human-semantic-warn-review`
- Plan task: `Task 1: 扩展人工复核报告合同和批次统计`
- OpenSpec mapping:
  - `7.1 扩展 QC JSON Schema，加入 completion_mode、资产级 failure_reason、未查看 selected issue 兼容和审计约束`
  - `7.3 更新批次投影，仅统计实际 issue review，并单独统计 early-fail 后未查看 Warn`
  - `7.4 为旧报告添加兼容读取与显式迁移测试，已完成历史报告保持只读`
- Stage: `done`
- Review mode: `thorough`
- Review/fix round: `extra 3/2 (explicitly authorized by user on 2026-07-22)`
- Implementer commits: `3a52716bda630f5e2740dba80b19aaafba8f8654`, `fe3651a51c8150b9e77cc88d52d4c3e827a529ad`, `23549ea4380c3d3c43893ec8e59c2206c0690692`, `b01bcf1bc3a5c3f92cb98be8e5dc2994338bfde1`
- Changed files: `qc_common/schema.py`, `qc_common/report_migration.py`, `qc_common/projection.py`, and three focused test files
- RED evidence: import failure for missing projection, then `2 failed, 46 passed`, then focused migration failure
- GREEN evidence: extra bypass regressions `2 passed`; schema/migration focus `51 passed`; write-path regression `21 passed`
- Review result: Task 1 accepted after controller adjudication. The user-authorized schema bypass fix is verified. Publisher `reviews[]` legality remains owned and enforced by `lerobot_v3_publisher.prerequisites._validate_manual_review()` before publication; Warn service canonical completion writes are Task 2. Aggregator wiring remains pending under OpenSpec 7.3, so only 7.4 is checked off now.
