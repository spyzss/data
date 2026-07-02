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

可选使用 YAML 覆盖阈值：

```yaml
sample_count: 10
alignment_mode: warn
thresholds:
  min_fps: 1
  min_width: 1
  min_height: 1
  min_sample_decode_ratio: 1.0
  max_mean_over_dark_ratio: 0.10
  max_mean_over_exposed_ratio: 0.05
  max_black_frame_ratio: 0.05
  max_frozen_frame_ratio: 0.8
```

运行后输出：

```text
sampled/XJGT_20260616/
  hdf5/
  video/
  quality_archive/
    <asset_id>.json
```

`quality_archive/<asset_id>.json` 是单条数据的全流程 QC 档案，和 `hdf5/`、`video/` 同级，格式见 `docs/asset-qc-json-format.md`。上游建档模块应在拉取完成后先创建这个文件；视频质量检测后续只更新其中的 `video_quality`、`hdf5_text_info`、`reference_quality` 等 block。后续批次 summary、表格或完整报告都可以直接从这些 `<asset_id>.json` 聚合生成。该检测不使用 VMAF、CAMBI 或标准对照视频；它只计算可解码性、fps、分辨率、抽样帧亮度/过暗/过曝/模糊代理、黑屏、冻结帧，以及可选 HDF5 帧数对齐。
