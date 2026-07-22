# Task 5 实施报告：Warn-only DTO、媒体 API 与自动 Lease

日期：2026-07-22  
目标提交：`feat(human-qc): expose warn-only full-video API`

## 1. 结果

- 新增 `WarnWorkbenchService` 与 frozen DTO，根对象和 issue 均采用显式白名单投影，不再经过旧 `WorkbenchService`/`jsonable`。
- task GET 只读持久化报告和媒体 probe cache；不调用 `EvidenceService`、ffmpeg 或 overlay renderer。
- launcher 强制 `--reviewer <id>`，首次 task GET 自动 acquire；`X-Reviewer-Lease` reload 自动 renew；token 不进入 URL/query。
- HTTP 只暴露 canonical Warn 路由；旧 `/api/assets`、`/api/semantic`、路径型 `/evidence` 返回稳定 404。
- 原视频和 opaque overlay 通过 per-asset catalog 提供；单 Range 严格解析，媒体以 64 KiB 上限分块传输。
- stale revision 返回脱敏 409；lease 写冲突返回脱敏 423；task load lease 占用返回 200 read-only task 且不含 token/持有人。
- Task 10 的 overlay status/retry 路由未提前实现；静态 UI 未修改。

## 2. TDD 证据

### 首轮 RED

命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_http_server.py \
  tests/test_human_qc_workbench.py \
  tests/test_review_evidence.py \
  tests/test_human_qc_media.py \
  tests/test_human_qc_launcher.py
```

结果：`28 failed, 11 passed`。失败原因符合预期：缺少 `human_qc.media`/`WarnWorkbenchService`，仍是旧 shared route，launcher 未要求 reviewer 且丢弃 source hash 元数据。

### 分组 GREEN

- media Range/catalog：`14 passed`
- Warn DTO/auto lease：`6 passed`
- Warn-only HTTP/media/error：`7 passed`
- launcher：`3 passed`

### 第二轮 RED → GREEN

RED：`3 failed`，分别证明 raw `message` 可泄漏路径、缺少 immutable `OverlayHandle`、正式包仍导出旧 facade。  
GREEN：`3 passed`；默认原因只消费显式安全字段，overlay provider 返回 opaque handle，`human_qc` 正式导出切换为 Warn facade。

## 3. API 合同

```text
GET  /api/warn/assets
GET  /api/warn/assets/{asset_id}/task
POST /api/warn/assets/{asset_id}/lease/acquire
POST /api/warn/assets/{asset_id}/lease/renew
POST /api/warn/assets/{asset_id}/lease/release
POST /api/warn/assets/{asset_id}/issues/{issue_id}/verdict
POST /api/warn/assets/{asset_id}/complete
GET  /media/assets/{asset_id}/source
GET  /media/assets/{asset_id}/overlays/{overlay_id}
```

## 4. 边界确认

未修改：

- `human_qc/warn_service.py`
- `qc_common/manual_review.py`
- pipeline/orchestrator/config
- `semantic_calibration/**`
- `human_qc/static/**`
- overlay worker、status/retry API

## 5. 验证

聚焦套件（含 launcher/media）：`41 passed`。  
聚焦与受影响面合并回归：`125 passed`。  
全量 Python suite：`1757 passed, 1 skipped`。  
`compileall`、`openspec validate add-human-semantic-warn-review --strict` 和 `git diff --check` 均通过。

## 6. Review round 1 修复（2026-07-22）

修复提交：`3f70d299eb40f8bc0e2481b4d41c1867bc738070`（`fix(human-qc): harden warn workbench transport`）。

### 6.1 逐项修复结果

1. DTO fail-closed：`manual_review_state`、`completion_mode`、review verdict、audit action、overlay state 和 threshold operator 均使用闭集；路径、命令、ffmpeg、traceback 等不可信展示文本不再进入 DTO，展示名/默认原因回退到稳定 issue code，threshold 仅接受有限 operator 与数值型值。
2. HEAD/405：媒体 HEAD 支持完整、单 Range 和 416，均不返回 body 且保留对应 headers；已知路由的任意不支持 method 统一返回脱敏 JSON 405 与 `Allow`，未知路由仍为 JSON 404。
3. 安全 reload/recovery：HTTP server 使用随机 opaque、`HttpOnly`、`SameSite=Strict` session cookie 在服务端关联每资产 lease token；cookie 不含 reviewer/token，未知客户端预置 session id 不会被采用；真实其他 reviewer 持有 lease 时仍返回 read-only。
4. hash/probe single-flight：同一 source identity/hash 的并发 hash 与 probe 共用 in-flight 结果；失败会共享给等待者并移除 flight，后续请求可重新尝试；probe 期间身份变化不会污染旧 hash cache。
5. 媒体一致性：catalog 在 hash/probe 前后校验 `(device, inode, size, mtime_ns)`；HTTP 在发送 headers 前打开并核对同一身份，随后从该已验证 descriptor 流式发送。确定性替换竞态返回稳定 `source_video_unavailable`，不会组合旧 ETag/size/fps 与新路径字节。
6. aligned minor：新增最小 `review_audit` 投影（仅 action、selected issue id、合法 timestamp）及服务端派生 `can_complete`，不返回 previous/raw audit 数据。

保持不变：canonical Warn routes、GET 200/206/416 Range、64 KiB 分块、opaque overlay allowlist 与 containment、verdict/complete 的 failure reason/completion mode 转发、revision/lease CAS、脱敏 409/423、early-fail；未新增 Task 10 status/retry、UI、semantic/pipeline/manual state-machine 改动。

### 6.2 RED 证据

首组（5 个 Important 与 aligned minor 的直接复现）：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py::test_warn_task_rejects_unknown_public_state_values \
  tests/test_human_qc_workbench.py::test_warn_task_replaces_untrusted_issue_text_with_stable_codes \
  tests/test_human_qc_workbench.py::test_warn_task_projects_minimal_audit_and_server_completion_gate \
  tests/test_human_qc_http_server.py::test_source_media_head_is_bodyless_for_full_range_and_416 \
  tests/test_human_qc_http_server.py::test_known_route_unsupported_method_is_json_405_and_unknown_is_404 \
  tests/test_human_qc_http_server.py::test_hard_reload_recovers_fixed_reviewer_lease_via_opaque_session \
  tests/test_human_qc_media.py::test_source_probe_failure_is_single_flight_and_later_request_retries \
  tests/test_human_qc_http_server.py::test_source_replacement_between_catalog_and_stream_is_rejected
```

结果：`8 failed in 4.40s`。失败分别为状态未拒绝、恶意字段原样泄漏、缺少 audit/gate、HEAD 501、PUT 501 HTML、无 reload session、并发 probe 未等待同一 flight、文件替换后发送新字节。

hash 层补充 RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_media.py::test_source_hashing_is_single_flight_for_one_file_identity
```

结果：`1 failed in 0.29s`，证明同一 identity 的并发首次 hash 被执行两次。

安全边缘补充 RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_http_server.py::test_known_route_unsupported_method_is_json_405_and_unknown_is_404 \
  tests/test_human_qc_http_server.py::test_hard_reload_recovers_fixed_reviewer_lease_via_opaque_session \
  tests/test_human_qc_media.py::test_probe_result_is_not_cached_when_source_identity_changes
```

结果：`3 failed in 1.37s`，证明未知 method 仍为 501 HTML、客户端可固定 session id、probe 竞态会污染旧 hash cache。

### 6.3 GREEN 与最终验证

对应 GREEN：首组 `8 passed in 2.38s`；hash 层 `1 passed in 0.26s`；安全边缘 `3 passed in 1.31s`。

最终聚焦命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_http_server.py \
  tests/test_human_qc_workbench.py \
  tests/test_review_evidence.py \
  tests/test_human_qc_media.py \
  tests/test_human_qc_launcher.py
```

结果：`51 passed in 5.48s`。

完整回归命令：

```bash
.venv/bin/python -m pytest -q
```

结果：`1767 passed, 1 skipped in 116.93s`。

静态验证命令：

```bash
.venv/bin/python -m compileall -q human_qc tools/serve_human_qc_workbench.py
git diff --check
```

结果：两项退出码均为 0，无输出。

### 6.4 变更文件与 concerns

- `human_qc/warn_workbench_service.py`
- `human_qc/media.py`
- `human_qc/http_server.py`
- `human_qc/__init__.py`
- `tests/test_human_qc_workbench.py`
- `tests/test_human_qc_media.py`
- `tests/test_human_qc_http_server.py`

Concerns：无阻断项。session 与 lease 按既有本地 launcher 边界保持 process-local；不可信展示文本有意回退到稳定 issue code，后续若需要更丰富文案，应从正式受信配置映射提供，而不是重新透传报告原文。
