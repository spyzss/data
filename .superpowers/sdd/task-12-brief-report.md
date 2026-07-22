# Task 12 文档契约与完整验证报告

## 范围

- 同步正式文档为 `自动/SAM3 → Warn 人工复核 → 语义`：
  `docs/PRD-asset-qc-pipeline.md`、`docs/asset-qc-json-format.md`、
  `docs/reviewer-guide.md`、`ACCEPTANCE.md` 与 `WORKFLOW_INTERFACE.md`。
- 同步 `docs/qc-dataflow-migration.md` 的旧语义优先表述，避免正式迁移文档与 Warn-first
  路由相互矛盾。
- 将 Warn 操作员 CLI 帮助明确为独立 8897 服务：`--batch-root`、
  `--quality-archive` 与 `--reviewer` 必填，`--sam3-model` 可选。
- 新增文档契约测试，覆盖 completion mode、未查看 Warn、Publisher 与 Warn 审计结构
  分离、全视频时间轴、半开区间、SAM3 局部 overlay、六档倍速、人工原因及旧指引移除。
- 本任务未修改任何产品运行逻辑或 OpenSpec task 勾选状态。

## TDD 证据

先将 `tests/test_qc_docs_contract.py` 扩展为 Warn-first 工作流与 Warn CLI 帮助契约。
后续终审再先把测试改为逐份读取关键文档的精确路由/Fail 断言；该 RED 明确暴露五份
文档缺少 `not_required` 语义 ready、`all_reviewed` 语义 ready、`early_fail` 停止/跳过，
且仍错误宣称 Fail 只能在完成复核时写入。同步文字后转为 GREEN。

## 发布的操作员合同

- `not_required` 是不写 `completion_mode` 的语义 ready Gate，不能写成 completed 或
  completion-mode pass；全部 Pass 以 `all_reviewed` 完成并使语义 ready。`early_fail`
  才停止资产并将语义标记 skipped，未查看 Warn 保留原状。
- 点击 Fail 立即保存当前 issue 的 fail verdict；人工原因的预选、多选、取消及 Other
  输入仅是本地草稿。只有点击“完成复核”才写资产级 `early_fail`、停止资产并自动跳转
  下一条。
- Warn 结果写入 `manual_review.issue_reviews[issue_id]`；Publisher audit 仅读取独立的
  `manual_review.reviews[]`。
- Warn 区间为 `[start_frame, end_frame_exclusive)`，显示结束帧为
  `end_frame_exclusive - 1`，不改变 legacy freeze；播放器使用整条视频。
- 时间轴按真实帧宽度绘制、重叠进入可交互 popover、点击跳转起始帧、播放针可自由拖动。
  窄色块仅显示颜色且不堆叠；同一区间的问题卡会逐条列出。
- SAM3 仅在本 Warn 的区间实时 overlay；`pending/failed` 仅锁定该 Warn，可轮询重试。
  视频聚焦后左右键逐帧，速率为 0.25×/0.5×/1×/1.5×/2×/3× 并保留上次设置。
- 人工原因可以预选、多选与取消；Other 必填，人工原因覆盖默认原因，选择原因本身不改变
  verdict。问题卡正文不展示机器指标；阈值只经 Warn 名称后的“？”提示展示。

## 验证

```text
.venv/bin/python -m pytest -q tests/test_qc_docs_contract.py tests/test_human_qc_static_contract.py
# 17 passed in 0.48s（含逐份关键文档和迁移文档的精确路由断言）

.venv/bin/python tools/serve_human_qc_workbench.py --help
.venv/bin/python tools/serve_semantic_calibration.py --help
# Warn help 显示 8897、必填 batch/archive/reviewer 和可选 --sam3-model；语义帮助可执行。

openspec validate add-human-semantic-warn-review --strict
# Change 'add-human-semantic-warn-review' is valid

node --test human_qc/static/*.test.mjs semantic_calibration/static/*.test.mjs
# 79 passed

.venv/bin/python -m pytest -q
# 1903 passed, 1 skipped in 122.53s

if rg -n "获取编辑锁|语义与 Warn 复核|[+-]1 帧|machine score" \
  human_qc/static docs/reviewer-guide.md; then exit 1; fi
# success

git diff --check
# success
```

`.venv/bin/python -m pytest -q tests/test_semantic_calibration_import_boundary.py` 通过 5 项；
随后对 `human_qc` 与 `semantic_calibration` 的 Python imports 执行 AST 双向边界检查，
结果无跨运行时 import。终审 route/UI/文档旧顺序扫描也通过；`/evidence/` 的唯一文本
命中见本报告末尾的 false-positive 说明。

## 终审勘误范围

本轮仅更正文档、docs contract tests 和本报告。跨运行时边界以 AST import 检查为准；
不为通过文档验收而改动生产模块，也不勾选 OpenSpec 11.2/11.3。

预检的 `/evidence/` 文字扫描在 `human_qc/http_server.py:261` 命中的是未知请求的静态
文件 fallback 排除条件，并非暴露的旧 evidence route；经路由分支核验后按 false positive
记录，不改动产品源码。

## 最终窄修复

最终复审发现两处“仅 `all_reviewed`”的遗漏：PRD 的 `supplier_evaluation` profile 表和
操作员指南的 8898 启动说明。先为两份文档加入精确正向/负向断言并观察 2 项 RED；随后
两处统一为 `all_reviewed` **或** `not_required` 后进入语义。该修改不触及产品逻辑、CLI
帮助或协调器的 OpenSpec 计划文件。
