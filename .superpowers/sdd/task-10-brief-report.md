# Task 10：同步播放 Overlay 与局部就绪门禁

## 范围

- 新增连续 SAM3 overlay 的安全 status/retry API、同 asset union retry seam，及一个浏览器侧单 overlay video 同步控制器。
- 修改 Warn 复核面板和 App 的局部门禁、轮询、retry 接线；保留 Task 7 的视频键盘、逐帧、倍速和 localStorage 真值。
- 没有修改 worker/cache、runner/recipe、timeline 实现、Task 11+ 文档或 OpenSpec 任务文件。

## TDD

### RED

先新增并运行：

```text
node --test human_qc/static/overlay_controller.test.mjs \
  human_qc/static/video_controller.test.mjs human_qc/static/workbench.test.mjs
.venv/bin/python -m pytest -q tests/test_human_qc_http_server.py
```

结果为预期失败：`OverlayController` 模块不存在；`VideoController.subscribe`、
`ReviewPanel.canSubmit`、App poll scheduler 尚不存在；status route 返回 404。随后
`test_production_provider_retries_the_same_asset_union_request_once` 也先失败，因
`ProductionWorkerOverlayProvider.retry_asset_overlays` 不存在。

### GREEN

- `GET /api/warn/assets/{asset}/overlays/{issue}/status` 只返回该 selected continuous-SAM3 issue 的安全投影；pending/generating 带 `Retry-After`。
- `POST .../retry` 要求 revision 与 lease，HTTP handler 只调用非阻塞 facade/worker seam；failed retryable 进入 pending/generating 时返回 202，不改 report revision；非 retryable、lease/revision、unknown issue 均保持稳定安全错误。
- `ProductionWorkerOverlayProvider` retry 重新使用同一 trusted asset-union request，并且仅调用 `worker.retry()`；不按 issue 重建或等待渲染。
- `OverlayController` 只复用一张 muted/no-controls overlay `<video>`，按半开区间映射 base time，镜像 seek/play/pause/rate，正常播放仅在超过一帧漂移时校正；source-scoped listeners 防止旧 segment 的迟到媒体事件污染新 segment。
- `ReviewPanel` 仅锁定 `overlay != null && status != ready`（或本地媒体未加载）的目标 issue；不重排最早 pending Pass 规则。失败且 retryable 的 issue 有单一“重试 Overlay”操作。
- App 使用一个 timeout scheduler（1/2/4/5 秒上限+jitter）、abort/generation 防旧 response、局部替换 issue overlay，不 reload task、不清空原因草稿。

## 验证

```text
node --test human_qc/static/*.test.mjs
# 54 passed

.venv/bin/python -m pytest -q \
  tests/test_human_qc_http_server.py \
  tests/test_human_qc_workbench.py \
  tests/test_review_evidence.py \
  tests/test_sam3_overlay_renderer.py \
  tests/test_human_qc_static_contract.py \
  tests/test_sam3_overlay_worker.py
# 134 passed

.venv/bin/python -m compileall -q \
  human_qc/warn_workbench_service.py human_qc/http_server.py \
  human_qc/sam3_overlay_renderer.py
node --check human_qc/static/app.js human_qc/static/overlay_controller.js \
  human_qc/static/review_panel.js
git diff --check
# success
```

## 终审修复（二）：轮询单飞

- status poll 显式保存单个 in-flight promise；任意 retry 或下一轮 intent 在前一轮结束前仅标记 queued，不会启动同 generation 的并发 status fetch。
- queued retry 保留 reset intent，当前 poll 结束后才按既有 scheduler 安排一次后续轮询；原有 1/2/4/5 秒退避、503 `Retry-After` 和本地草稿语义不变。
- asset change 和 destroy 会同时清空 timer、queued intent 和 in-flight 引用；迟到的旧请求不能为新 asset 重新安排轮询。

```text
node --test human_qc/static/*.test.mjs
# 59 passed

.venv/bin/python -m pytest -q \
  tests/test_human_qc_static_contract.py \
  tests/test_human_qc_workbench.py \
  tests/test_human_qc_http_server.py
# 34 passed

node --check human_qc/static/app.js
git diff --check
# success
```

## 终审修复（最小范围）

- 同一 issue 的 `generating` 状态仅在 overlay 安全投影实际变化时重置轮询；连续同值响应按 `1000 → 2000 → 4000 → 5000ms` 退避。
- retry 返回 `503` 时只读取 `Retry-After` 响应头这一安全 metadata；本地仍保留 failed/busy 视图，冷却期内保持单飞，期满后继续 status 轮询并解锁。
- status 网络错误或 `5xx` 不暴露响应 body，只展示“状态暂时无法更新，正在重试”，且不会 reload task 或清空人工原因草稿。

```text
node --test human_qc/static/*.test.mjs
# 57 passed

.venv/bin/python -m pytest -q \
  tests/test_human_qc_static_contract.py \
  tests/test_human_qc_workbench.py \
  tests/test_human_qc_http_server.py
# 34 passed

git diff --check
# success
```
