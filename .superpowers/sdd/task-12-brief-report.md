# Task 12 文档契约与完整验证报告

## 范围

- 同步正式文档为 `自动/SAM3 → Warn 人工复核 → 语义`：
  `docs/PRD-asset-qc-pipeline.md`、`docs/asset-qc-json-format.md`、
  `docs/reviewer-guide.md`、`ACCEPTANCE.md` 与 `WORKFLOW_INTERFACE.md`。
- 将 Warn 操作员 CLI 帮助明确为独立 8897 服务：`--batch-root`、
  `--quality-archive` 与 `--reviewer` 必填，`--sam3-model` 可选。
- 新增文档契约测试，覆盖 completion mode、未查看 Warn、Publisher 与 Warn 审计结构
  分离、全视频时间轴、半开区间、SAM3 局部 overlay、六档倍速、人工原因及旧指引移除。
- 本任务未修改任何产品运行逻辑或 OpenSpec task 勾选状态。

## TDD 证据

先将 `tests/test_qc_docs_contract.py` 扩展为 Warn-first 工作流与 Warn CLI 帮助契约。
初始运行预期 RED：正式文档不存在 `自动/SAM3 → Warn 人工复核 → 语义`，CLI 帮助也
未说明 Warn 人工复核与操作员输入。同步文档与 argparse help 后转为 GREEN。

## 发布的操作员合同

- `not_required` 无需正常 Pass 抽检；全部 Pass 以 `all_reviewed` 完成；Fail 只在
  “完成复核”提交 `early_fail`，未查看 Warn 保留原状并不进入语义。
- Warn 结果写入 `manual_review.issue_reviews[issue_id]`；Publisher audit 仅读取独立的
  `manual_review.reviews[]`。
- Warn 区间为 `[start_frame, end_frame_exclusive)`，显示结束帧为
  `end_frame_exclusive - 1`，不改变 legacy freeze；播放器使用整条视频。
- 时间轴按真实帧宽度绘制、重叠进入可交互 popover、点击跳转起始帧、播放针可自由拖动。
  窄色块仅显示颜色且不堆叠；同一区间的问题卡会逐条列出。
- SAM3 仅在本 Warn 的区间实时 overlay；`pending/failed` 仅锁定该 Warn，可轮询重试。
  视频聚焦后左右键逐帧，速率为 0.25×/0.5×/1×/1.5×/2×/3× 并保留上次设置。
- 人工原因可以预选、多选与取消；Other 必填，人工原因覆盖默认原因，选择原因本身不改变
  verdict。问题卡正文不展示检测分数；阈值只经 Warn 名称后的“？”提示展示。

## 验证

```text
.venv/bin/python -m pytest -q tests/test_qc_docs_contract.py tests/test_human_qc_static_contract.py
# 7 passed in 0.45s

.venv/bin/python tools/serve_human_qc_workbench.py --help
.venv/bin/python tools/serve_semantic_calibration.py --help
# Warn help 显示 8897、必填 batch/archive/reviewer 和可选 --sam3-model；语义帮助可执行。

openspec validate add-human-semantic-warn-review --strict
# Change 'add-human-semantic-warn-review' is valid

node --test human_qc/static/*.test.mjs semantic_calibration/static/*.test.mjs
# 79 passed

.venv/bin/python -m pytest -q
# 1898 passed, 1 skipped in 133.52s

if rg -n "获取编辑锁|语义与 Warn 复核|[+-]1 帧|machine score" \
  human_qc/static docs/reviewer-guide.md; then exit 1; fi
# success

git diff --check
# success
```

## 已隔离的后续项

计划中的跨运行时名称扫描仍失败，原因不在本任务允许的文档/CLI-help 范围内：

```text
human_qc/legacy_import.py:593  semantic_calibration
human_qc/warn_service.py:293  semantic_calibration
human_qc/warn_service.py:295  semantic_calibration
```

这些既有生产源码引用已报告给主任务，须以独立最小化运行时修复和审查处理；本提交不通过
改动产品逻辑来掩盖该验证失败。
