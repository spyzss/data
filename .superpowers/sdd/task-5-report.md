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

## 7. Review round 2 修复与中断恢复（2026-07-22）

业务与测试提交：`113b9273cd310152999133fc4d6f052c54a94712`（`fix(human-qc): close warn transport race gaps`）。

### 7.1 恢复与 RED 证据

- 从 `.superpowers/sdd/task-5-round2-recovery.md` checkpoint 接管中断工作树；先读取 round-2 review、recovery、既有 Task 5 报告和完整 diff，保留所有未提交改动，未执行 reset/checkout。
- checkpoint 记录前一修复代理已在业务改动前观察到四个 RED：不安全 selected issue ID 泄漏到 task DTO、泄漏到 review audit、公开 ID 写回时未反映射到持久化 raw ID，以及同 inode/同 size 原地覆盖后出现旧 ETag 配新 body。恢复时未回退现有业务改动来重复 RED，避免破坏中断工作树。
- 恢复设计审查另发现：opaque ID 若为 `command` 等稳定 code 形式且 issue 缺少可信展示字段，旧实现仍会把 raw ID 用作 `display_name/default_reason` 回退。新增直接回归后观察到预期 RED：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py::test_opaque_issue_id_is_not_reused_as_display_fallback
```

结果：`1 failed in 0.38s`，实际 `display_name` 为原始 `command`，证明 public-ID 判定与展示回退判定不一致。

### 7.2 修复设计与范围

1. 对 selected issue ID 建立闭集 public 表示：合法且不含危险标记的 opaque component 保持兼容；其余值映射为固定 `issue-<sha256>`。task issue 与 audit 只投影 public ID，缺失/冲突错误不含 raw 值。
2. 每次 verdict 只在当前报告的 selected 集合中将 public ID 反映射为 raw ID，再调用既有 domain write；未知 public ID fail-closed，不改变 revision/lease/CAS 语义。
3. opaque 映射后的 raw ID 不再参与展示 code 回退；可信显式 code 缺失时统一回退 `issue`，避免 raw `command/ffmpeg/traceback` 经展示字段旁路泄漏。
4. source media 在发送 headers 前，从已核验 descriptor 以 64 KiB 分块复制到临时文件快照，并复核最终 `(device, inode, size, mtime_ns)`、长度和 SHA-256/ETag。成功 full/Range/HEAD 响应只读取该一致快照；源文件随后同 inode/同 size 原地改写不会改变已响应 body，下一请求会按新 identity/hash 生成新 ETag。

仅修改 `human_qc/media.py`、`human_qc/warn_workbench_service.py` 及两份对应测试；未实现 Tasks 6–10，未修改 plan/OpenSpec/progress。恢复前已存在的 `.comet/subagent-progress.md` 未提交改动保持原样且未暂存。

### 7.3 GREEN 与最终验证

round-2 四类直接回归（task/audit 节点含 4 个参数案例）：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py::test_untrusted_selected_issue_id_is_opaque_in_task_and_audit \
  tests/test_human_qc_workbench.py::test_opaque_selected_issue_id_preserves_internal_write_semantics \
  tests/test_human_qc_http_server.py::test_source_in_place_rewrite_after_open_serves_one_consistent_snapshot
```

结果：`6 passed in 0.82s`。

补充展示回退与安全错误路径 GREEN（连同上述直接回归）：`8 passed in 0.83s`。

完整 Task 5 聚焦套件：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_http_server.py \
  tests/test_human_qc_workbench.py \
  tests/test_review_evidence.py \
  tests/test_human_qc_media.py \
  tests/test_human_qc_launcher.py
```

结果：`59 passed in 6.00s`。

全量 Python 回归：`1775 passed, 1 skipped in 119.80s`。

静态验证：

```bash
.venv/bin/python -m compileall -q human_qc tools/serve_human_qc_workbench.py
git diff --check -- \
  human_qc/media.py \
  human_qc/warn_workbench_service.py \
  tests/test_human_qc_http_server.py \
  tests/test_human_qc_workbench.py
```

结果：两项退出码均为 0，无输出。

### 7.4 Concerns

无阻断项。source 成功响应现在会在发送 headers 前完整读取并写入临时快照，以磁盘 I/O 和临时空间换取强一致性及有界内存；超大视频或高并发部署应预留相应临时存储容量并监控首字节延迟。

## 8. Review round 3 修复：service exception 脱敏（2026-07-22）

业务与测试提交：`d8bc05956fdf734246e2fb3e05b7812008f4aebd`（`fix(human-qc): sanitize raw warn state errors`）。

### 8.1 Root cause 与 RED

`WarnWorkbenchService.warn_verdict()` 会将安全公开 issue ID 反查为原始持久化 ID，以保持既有 domain 写回语义。若该 issue 缺少机器 verdict，底层 `WarnReviewService.submit_verdict()` 会抛出含 raw ID 的 `WarnStateError`；HTTP 边界虽会脱敏，但直接 service 调用仍泄漏该值。

新增最小回归后先执行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py::test_opaque_selected_issue_without_machine_verdict_has_sanitized_service_error
```

结果：`1 failed in 0.29s`。实际异常为 `issue /private/traceback-command=ffmpeg has no machine verdict`，证明泄漏来自 domain-to-workbench exception 边界，而非 DTO 或 HTTP 序列化。

### 8.2 修复与 GREEN

仅在 `WarnWorkbenchService.warn_verdict()` 的 domain 调用处处理这一条精确的 expected state error：当且仅当错误文本是当前已反查 raw issue 的“缺少机器 verdict”状态时，使用 `from None` 转换为稳定的 `selected issue has no machine verdict`。其余 `WarnStateError` 原样抛出；未修改 domain service、公开到 raw 的写回映射、revision/lease/CAS、HTTP 脱敏或媒体行为。

定点兼容回归：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_workbench.py::test_opaque_selected_issue_without_machine_verdict_has_sanitized_service_error \
  tests/test_human_qc_workbench.py::test_opaque_selected_issue_id_preserves_internal_write_semantics \
  tests/test_human_qc_workbench.py::test_untrusted_selected_issue_id_is_opaque_in_task_and_audit
```

结果：`6 passed in 0.14s`。

Task 5 聚焦套件：

```bash
.venv/bin/python -m pytest -q \
  tests/test_human_qc_http_server.py \
  tests/test_human_qc_workbench.py \
  tests/test_review_evidence.py \
  tests/test_human_qc_media.py \
  tests/test_human_qc_launcher.py
```

结果：`60 passed in 5.85s`。

`git diff --check` 退出码 0，无输出。

### 8.3 Concerns

无阻断项。该转换有意只匹配此 domain service 的既有、精确错误文本；若未来 domain error contract 改动，需要同步更新这个 facade 回归测试，而不能扩展为对所有 `WarnStateError` 的吞没或重写。
