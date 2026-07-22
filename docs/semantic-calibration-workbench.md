# 独立语义校准工作台使用说明

## 1. 模块边界

语义校准是独立应用，不与 Warn 人工质检共用前端、后端 facade、HTTP 路由或
静态资源：

- Python 包：`semantic_calibration/`
- 启动入口：`tools/serve_semantic_calibration.py`
- 浏览器入口：服务根地址，默认 `http://127.0.0.1:8898/`
- API 命名空间：`/api/semantic/...`
- 正式状态源：`quality_archive/<asset_id>.json`

`human_qc` 不托管语义页面或语义写 API；语义模块也不导入 Warn、Evidence 或
人工质检服务。两个模块只通过 `asset_qc_report.v2` 状态合同衔接。

## 2. 启动

批次根目录内必须包含 HDF5、可选视频和 `quality_archive/*.json`。报告的
`source_files.hdf5.path` 与 `source_files.video.path` 必须位于批次根目录内。

```bash
.venv/bin/python tools/serve_semantic_calibration.py \
  --batch-root /data/qc-batch \
  --quality-archive quality_archive \
  --host 127.0.0.1 \
  --port 8898 \
  --profile acceptance
```

可选参数：

- `--dataset-path`：HDF5 subtask dataset，默认 `/label/subtask_label`。
- `--lease-ttl-seconds`：任务占用时间，默认 900 秒。
- `--profile`：必须与报告执行 profile 一致。

直接打开根地址会自动加载第一条可编辑语义任务；没有待办时显示空队列状态。
使用 `/?asset_id=<asset_id>` 可以深链到指定资产。深链不能绕过服务端门禁。

## 3. 前置门禁

正式顺序为：

```text
自动 QC -> manual_review -> semantic_consistency -> 后续模块/完成
```

语义服务只列出以下资产：

- `manual_review.state=not_required`：没有人工候选，沿用现有跳过规则；
- `manual_review.state=completed` 且 `completion_mode=all_reviewed`：人工候选全部
  判定 Pass。

以下资产不可读取或修改语义任务：

- `not_evaluated`、`queued`、`in_progress`：人工质检尚未完成；
- `completion_mode=early_fail`：人工确认 Fail，流水线已经终止；
- `skipped_due_to_fail` 或 `error`：流程已跳过或出错；
- pipeline cursor 不是 `awaiting_external/semantic_consistency`。

门禁由服务端同时校验 QC JSON 和 pipeline cursor。客户端参数、直接 API 请求和
历史 URL 都不能绕过。

## 4. 页面操作

1. 填写 reviewer，点击“开始校准”取得单资产 lease。
2. 播放视频并选择 subtask。
3. 只拖动相邻 subtask 之间的内部边界手柄，或修改当前 subtask 文字。
4. 每次修改立即进入 pending 状态；确认或取消前不能继续其他编辑或完成资产。
5. 检查 before/after 后确认，或取消并恢复原值。
6. 全部修改完成后提交语义校准。

时间轴内部使用半开区间 `[start, end)`。拖动一个内部共享边界会同时改变左侧
subtask 的结束帧和右侧 subtask 的起始帧，两段属于同一个不可拆分事务；确认一次
只增加一次 `timeline_edit_count`。首段最左边界、末段最右边界和整个 subtask
色块不可拖动。

## 5. 写入与恢复

- 所有写请求携带 lease token 和 `expected_revision`。
- 409 `stale_revision`：重新加载最新报告，不能覆盖新 revision。
- 423 `lease_held` / `lease_invalid`：重新取得任务占用后继续。
- pending edit 在服务端持久化，刷新不依赖 localStorage。
- 完成时先写同目录临时 HDF5，重新打开并验证结构、内容与共享边界，然后
  `fsync` + `os.replace`；失败时原 HDF5 保持不变。
- `finalizing` 事务在进程重启后根据报告事务记录和 HDF5 hash 恢复。

## 6. API

语义服务只暴露以下正式路由：

```text
GET  /api/semantic/assets
GET  /api/semantic/assets/{asset_id}/task
GET  /api/semantic/assets/{asset_id}/video
POST /api/semantic/assets/{asset_id}/lease/acquire
POST /api/semantic/assets/{asset_id}/lease/renew
POST /api/semantic/assets/{asset_id}/lease/release
POST /api/semantic/assets/{asset_id}/boundary/pending
POST /api/semantic/assets/{asset_id}/text/pending
POST /api/semantic/assets/{asset_id}/pending/confirm
POST /api/semantic/assets/{asset_id}/pending/cancel
POST /api/semantic/assets/{asset_id}/complete
```

该 server 不提供 Warn task、SAM3 evidence 或人工 Pass/Fail 路由。错误响应只返回
稳定错误码和安全提示，不暴露绝对路径、命令行或 Python 异常原文。

## 7. 验证命令

```bash
.venv/bin/python -m pytest -q \
  tests/test_semantic_service.py \
  tests/test_semantic_calibration_application.py \
  tests/test_semantic_calibration_http_server.py \
  tests/test_semantic_calibration_import_boundary.py \
  tests/test_semantic_calibration_static_contract.py \
  tests/test_hdf5_semantic_commit.py \
  tests/test_shared_timeline.py \
  tests/test_subtask_source_adapter.py

node --test \
  semantic_calibration/static/app.test.mjs \
  semantic_calibration/static/semantic_adapter.test.mjs

openspec validate add-human-semantic-warn-review --strict
```
