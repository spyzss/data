# 批次抽样拉取模块

## 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 本地批次配置

```yaml
manifest: XJGT_20260616.xlsx
readme: README.txt
output: sampled/XJGT_20260616
workers: 8
seed: 20260701
sample_ratio: 0.01
hdf5:
  kind: local
  root: hdf5
video:
  kind: local
  root: video
```

运行：

```bash
python run_acceptance_pull.py --config pull.yaml
```

## oss-browser2 本地测试配置

```yaml
batch_uri: oss://xingjiguitu/egodata/XJGT_20260616
region: beijing
output: sampled/XJGT_20260616
workers: 8
sample_ratio: 0.01
```

当前本地 OSS 测试优先复用本机 `oss-browser2` 登录上下文。运行前先打开 `oss-browser2` 并完成登录。模块会读取 `~/Library/Application Support/oss-browser2/config.json`，解密 `currentSession`，并只在内存中使用 access key 信息；日志和报告不得输出登录态、token 或 access key。后续切换正式登录方式时，配置文件仍保持 `batch_uri` 和 `region` 这种业务输入形式。

OSS 批次地址默认包含以下结构：

```text
oss://bucket/path/to/batch/
  README.txt
  *.xlsx
  hdf5/
  video/
```

程序会自动将 `region: beijing` 转成 `https://oss-cn-beijing.aliyuncs.com`，将 `batch_uri` 拆成 bucket 和批次根 prefix，并推导：

```text
hdf5 prefix: <batch prefix>/hdf5
video prefix: <batch prefix>/video
```

## 需要人工填写的信息

- `output`：本地输出目录，程序会在其中创建 `hdf5/`、`video/`、`reports/`。
- `sample_ratio`：抽样比例，小数形式；默认 `0.01` 表示 1%，后续可改为 `0.02`、`0.005` 等。
- `workers`：并发拉取线程数，默认 `8`。
- `seed`：随机种子；不填时按运行当天日期生成，例如 2026-07-01 生成 `20260701`。需要复现历史抽样时可以手工填写。
- 本地来源需要填写 `manifest`、`readme`、`hdf5.root`、`video.root`。
- OSS 来源只需要填写 `batch_uri` 和 `region`；当前国内节点支持用 `beijing` 这类短名，程序自动生成 endpoint，并从批次根目录自动下载 `README.txt` 和 `.xlsx` 清单。
- OSS 登录态不写入 YAML；本地测试前需要手工打开 `oss-browser2` 并完成登录。

## 输出

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  reports/
    id_consistency.csv
    sample_manifest.csv
    pull_report.csv
    summary.json
```

## 抽样口径

- 有效 ID 为清单、hdf5、video 三方都存在的交集。
- 最低抽样量为 `ceil(valid_id_count * sample_ratio)`，默认 `sample_ratio: 0.01`。
- `scene` 必须全覆盖；当 scene 数量超过最低抽样量时，实际抽样量允许超过最低抽样量。
- `task` 不要求全覆盖，只在剩余名额中尽量分散。
- 三方 ID 不一致时输出报告，并继续在有效 ID 内抽样。

## 后续视频质量检测

抽样拉取完成后，可以对输出批次运行无参考视频质量检测：

```bash
python run_acceptance_video_quality.py --batch sampled/XJGT_20260616
```

可选使用 YAML 覆盖视频预筛参数：

```yaml
decode:
  max_sample_frames: 300
hdf5_alignment:
  mode: fail
resolution:
  min_short_side_fail: 720
  min_long_side_fail: 1280
exposure:
  black:
    max_frame_count_fail: 10
    ratio_pass: 0.01
    ratio_warn: 0.90
  over_dark:
    ratio_pass: 0.05
    ratio_warn: 0.90
  over_exposed:
    ratio_pass: 0.05
    ratio_warn: 0.90
sharpness_global:
  target_short_side: 720
  laplacian_p10_pass: 15
  laplacian_p10_warn: 0
  laplacian_median_pass: 20
  laplacian_median_warn: 0
  laplacian_under_100_ratio_pass: 1.00
  laplacian_under_100_ratio_warn: 1.00
  tenengrad_p10_pass: 6
  tenengrad_p10_warn: 4
  tenengrad_median_pass: 7
  tenengrad_median_warn: 4
freeze:
  adjacent_near_duplicate_ratio_warn: 0.90
  freeze_candidate_window_sec: 0.5
  confirmed_freeze_window_sec: 1.0
  frozen_frame_ratio_pass: 0.05
  frozen_frame_ratio_warn: 0.10
  min_interval_frames: 6
  min_interval_duration_ms: 100
  ssim_min: 0.995
  phash_hamming_max: 4
  motion_conflict_enabled: true
  critical_window_enabled: true
  video_state_conflict_noncritical_duration_ms_fail: 1000
  video_state_conflict_critical_duration_ms_fail: 500
defects:
  max_duration_ratio_fail: 0.10
  duration_ratio_warn: 0.05
hand_roi:
  enabled: false
  mode: warn_except_severe_fail
```

运行后输出：

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  quality_archive/
    <asset_id>.json
```

`quality_archive/<asset_id>.json` 是单条数据的全流程 QC 档案，和 `hdf5/`、`video/` 同级，格式见 `docs/asset-qc-json-format.md`。上游建档模块应在拉取完成后先创建这个文件；视频质量检测后续只更新其中的 `video_quality`、`hdf5_text_info`、`reference_quality` 等 block。后续批次 summary、表格或完整报告都可以直接从这些 `<asset_id>.json` 聚合生成。

当前视频模块是 `video_prefilter_v0.3.2` 低成本预筛，不使用 VMAF、CAMBI 或标准对照视频；它只计算基础可用性、fps、分辨率、时间轴连续性、抽样解码、黑帧/过暗/过曝、全帧清晰度、冻结帧、丢帧、HDF5 帧数对齐、瑕疵时长合计比例，以及连续冻帧区间。掉帧检测优先使用 `ffprobe` 每帧真实 PTS，其次 PyAV；OpenCV `CAP_PROP_POS_MSEC` 只作为 fallback，且写入 `drop_detection_reliable: false`。`drop_frame_ratio` 按 `estimated_missing_frames` 估算，不再按异常间隔次数计数。`adjacent_near_duplicate_ratio` 只作为低运动量指标，不作为拒收条件，默认 warn 线保持 `0.90`；相隔 0.5s 的两帧仍近重复才记为 `freeze_candidate`，相隔 1.0s 的两帧仍近重复才记为 confirmed freeze 并进入 `frozen_intervals`。如果 confirmed freeze 期间 HDF5 hand keypoints / action / cam_pose / 4x4 transform 仍有明显变化，则写入 `video_state_conflict`：非关键窗口 >=1.0s hard reject，grasp / place / contact / hand-object interaction 关键窗口 >=0.5s hard reject。默认不再运行 hand ROI，因为粗 bbox 对预筛 gating 噪声较大。该版本按人工复核反馈进一步放宽清晰度线：能看清边缘、分清物体的视频不应仅因低纹理或低锐化响应而 warn/fail；清晰度 hard fail 只保留给极端模糊风险。hard fail 更关注视频打不开、解码失败、黑屏超过 10 帧、接近全段过暗/过曝、confirmed freeze / 丢帧，以及所有瑕疵时长合计超过 10%。输出 JSON 里必须保存 `decision: pass|warn|fail` 和 `should_run_mask_qc`：只有 `decision == "fail"` 时后续 mask / 骨骼点比对 / 语义一致性等高成本 QC 才应跳过。`reasons` / `warn_reasons` 只保存稳定原因码；报告展示和人工排查必须读取 `reason_details` / `warn_reason_details`，其中包含触发指标、实际值、阈值和比较方向。
