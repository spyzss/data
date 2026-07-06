# 单条数据质量档案 JSON 规范 v1

本文档给上游建档同事使用：每条数据在拉取完成后必须先生成一个长期伴随全流程的质量档案文件。后续视频质量、HDF5 对齐、标注一致性、手部/物体检测等 QC 模块都只在这个文件里追加或更新自己的 block。

## 文件位置

```text
<batch>/
  hdf5/
  video/
  quality_archive/
    <asset_id>.json
```

要求：

- 质量档案目录固定为 `<batch>/quality_archive/`，与 `hdf5/`、`video/` 同级。
- 文件名固定为 `<asset_id>.json`，例如 `408817.json`。
- 不要把单条数据质量档案放在 `reports/` 下；`reports/` 后续只适合放由质量档案聚合出来的批次报告。

## asset_id 规则

`asset_id` 必须和同一条数据的 HDF5/video ID 一致。

推荐命名映射：

```text
video/408817_video.mp4       -> asset_id = 408817
hdf5/408817_hdf5.hdf5        -> asset_id = 408817
quality_archive/408817.json  -> asset_id = 408817
```

如果供应商文件名没有 `_video`、`_hdf5` 后缀，上游建档模块必须先确定唯一稳定 ID，并在 `source_files` 中记录原始文件名。

## 初始最小 JSON

上游建档模块在拉取完成后至少写出下面结构：

```json
{
  "schema_version": "asset_qc_report.v1",
  "asset_id": "408817",
  "qc_summary": {
    "status": "pending",
    "passed": null,
    "completed_modules": [],
    "failed_modules": [],
    "reasons": [],
    "warn_reasons": [],
    "should_run_mask_qc": null
  },
  "source_files": {
    "video": {
      "path": "video/408817_video.mp4",
      "filename": "408817_video.mp4",
      "extension": ".mp4",
      "exists": true
    },
    "hdf5": {
      "path": "hdf5/408817_hdf5.hdf5",
      "filename": "408817_hdf5.hdf5",
      "extension": ".hdf5",
      "exists": true
    }
  },
  "hdf5_text_info": null,
  "video_quality": null,
  "reference_quality": {
    "mode": "none",
    "reference_video_path": null,
    "vmaf": null,
    "note": "当前无标准对照视频，未计算 VMAF。"
  }
}
```

写法要求：

- `schema_version` 当前固定为 `asset_qc_report.v1`。
- `path` 使用相对 `<batch>` 根目录的路径，不写本机绝对路径。
- 尚未执行的模块写 `null`，不要写空对象伪装完成。
- `qc_summary.status` 初始为 `pending`。
- `qc_summary.passed` 初始为 `null`，等至少一个 QC 模块完成后再改为 boolean。
- `qc_summary.should_run_mask_qc` 初始为 `null`；视频预筛完成后必须写 boolean，供后续 pipeline 判断是否继续跑 mask / 骨骼点比对 / 语义一致性。

## 顶层字段

| 字段 | 类型 | 创建方 | 说明 |
|---|---|---|---|
| `schema_version` | string | 建档模块 | 当前固定为 `asset_qc_report.v1`。 |
| `asset_id` | string | 建档模块 | 单条数据稳定 ID。 |
| `qc_summary` | object | 建档模块初始化，QC 模块更新 | 全流程 QC 汇总状态。 |
| `source_files` | object | 建档模块 | 该数据关联的原始文件。 |
| `hdf5_text_info` | object/null | HDF5/QC 模块 | HDF5 文本与帧数对齐信息。 |
| `video_quality` | object/null | 视频质量模块 | 视频质量检测结果。 |
| `reference_quality` | object | 建档模块初始化，后续可更新 | 标准对照质量指标预留位。 |

## qc_summary

初始状态：

```json
{
  "status": "pending",
  "passed": null,
  "completed_modules": [],
  "failed_modules": [],
  "reasons": [],
  "warn_reasons": [],
  "should_run_mask_qc": null
}
```

模块执行后示例：

```json
{
  "status": "pass",
  "passed": true,
  "completed_modules": ["video_quality"],
  "failed_modules": [],
  "reasons": [],
  "warn_reasons": [],
  "should_run_mask_qc": true
}
```

字段规则：

| 字段 | 类型 | 规则 |
|---|---|---|
| `status` | string | `pending`、`pass`、`warn`、`fail` 四选一。 |
| `passed` | boolean/null | `pending` 时为 `null`；`pass/warn` 为 `true`；`fail` 为 `false`。 |
| `completed_modules` | string[] | 已完成并写入结果的模块名。 |
| `failed_modules` | string[] | 有失败结论的模块名。 |
| `reasons` | string[] | 全局 hard fail 原因，格式建议为稳定机器可读枚举。 |
| `warn_reasons` | string[] | 全局 warning 原因，不阻断后续高成本 QC。 |
| `should_run_mask_qc` | boolean/null | 视频预筛完成后必须写入；规则是 `status != "fail"`。 |

## source_files

```json
{
  "video": {
    "path": "video/408817_video.mp4",
    "filename": "408817_video.mp4",
    "extension": ".mp4",
    "exists": true
  },
  "hdf5": {
    "path": "hdf5/408817_hdf5.hdf5",
    "filename": "408817_hdf5.hdf5",
    "extension": ".hdf5",
    "exists": true
  }
}
```

规则：

- `path` 必须是批次根目录相对路径。
- 对缺失文件也要保留对应 key，并写 `exists: false`。
- 如果同一条数据有多路视频，`video` 可以扩展为对象集合，例如 `videos.cam_left`、`videos.cam_right`；不要把多个文件塞进一个字符串。

## hdf5_text_info

该 block 由 HDF5/QC 模块写入。未执行前为 `null`。

```json
{
  "alignment": {
    "status": "matched",
    "mode": "fail",
    "video_frame_count": 913,
    "hdf5_frame_count": 913,
    "frame_count_delta": 0,
    "frame_count_delta_ratio": 0.0,
    "frame_count_match": true,
    "reason": null
  },
  "text_fields": {
    "attributes": {},
    "datasets": {
      "/label/text_label": {
        "data_cn": {
          "scene": "家庭",
          "events": []
        }
      }
    }
  }
}
```

`alignment.status` 可选值：

| status | 含义 |
|---|---|
| `matched` | 找到 HDF5，且帧数与视频帧数一致。 |
| `mismatch` | 找到 HDF5，但帧数不一致。 |
| `missing` | 未找到对应 HDF5。 |
| `unreadable` | HDF5 存在但不可读，或缺少约定数据集。 |

`text_fields` 只收集字符串型 HDF5 attribute 和 dataset，不收集图像、关键点、大数组等数值数据。如果字符串内容本身是 JSON object 或 array，应解析成嵌套 JSON；普通文本保持 string。

## video_quality

该 block 由视频预筛模块写入。未执行前为 `null`。本阶段定位是低成本视频预筛，只判断这条视频是否值得继续跑 mask、骨骼点比对、语义一致性等高成本 QC；不做 21 点精度验收、手物 mask IoU、轨迹跳变或 subtask 验收。

```json
{
  "stage": "video_prefilter",
  "threshold_version": "video_prefilter_v0.2.1",
  "evaluation": {
    "decision": "pass",
    "passed": true,
    "reasons": [],
    "warn_reasons": [],
    "should_run_mask_qc": true
  },
  "metadata": {
    "opened": true,
    "frame_count": 913,
    "fps": 29.987814371159146,
    "duration_seconds": 30.4457,
    "width": 1280,
    "height": 720,
    "short_side": 720,
    "long_side": 1280
  },
  "sampling": {
    "sample_count_configured": 30,
    "sampled_frame_count": 30,
    "decoded_sample_count": 30,
    "sample_decode_ratio": 1.0
  },
  "metrics": {
    "video_basic": {
      "video_open_ok": true,
      "video_stream_present": true,
      "codec_readable": true,
      "metadata_read_ok": true,
      "fps": 29.97,
      "short_side": 720,
      "long_side": 1280
    },
    "timeline_metrics": {
      "pts_monotonic_valid": true,
      "drop_frame_ratio": 0.0,
      "frame_interval_p99_ms": 35.1,
      "max_frame_gap_ms": 38.4
    },
    "decode_metrics": {
      "sample_decode_ratio": 1.0
    },
    "exposure_metrics": {
      "black_frame_ratio": 0.0,
      "mean_over_dark_ratio": 0.0,
      "mean_over_exposed_ratio": 0.0
    },
    "sharpness_global": {
      "sharpness_scale_short_side": 720,
      "laplacian_p10": 420.5,
      "laplacian_median": 650.2,
      "laplacian_under_100_ratio": 0.0,
      "tenengrad_p10": 34.1,
      "tenengrad_median": 42.7
    },
    "freeze_metrics": {
      "frozen_frame_ratio": 0.0,
      "max_consecutive_frozen_sec": 0.0
    },
    "hdf5_alignment": {
      "enabled": true,
      "mode": "fail",
      "video_frame_count": 913,
      "hdf5_frame_count": 913,
      "frame_count_delta": 0,
      "frame_count_delta_ratio": 0.0
    },
    "hand_roi_metrics": {
      "enabled": true,
      "source": "hdf5_keypoints_bbox",
      "hand_roi_available_ratio": 0.86,
      "hand_roi_laplacian_p10": 185.3,
      "hand_roi_laplacian_median": 280.4,
      "hand_roi_tenengrad_p10": 20.2,
      "hand_roi_tenengrad_median": 26.8,
      "hand_roi_blur_bad_frame_ratio": 0.08
    }
  },
  "thresholds": {
    "threshold_version": "video_prefilter_v0.2.1",
    "fps": {
      "expected_fps": null,
      "min_fps_pass": 24,
      "min_fps_warn": 20,
      "min_fps_fail": 20
    },
    "resolution": {
      "min_short_side_fail": 720,
      "min_long_side_fail": 1280
    },
    "sharpness_global": {
      "target_short_side": 720,
      "laplacian_p10_pass": 150,
      "laplacian_p10_warn": 80,
      "laplacian_median_pass": 220,
      "laplacian_median_warn": 120,
      "laplacian_under_100_ratio_pass": 0.05,
      "laplacian_under_100_ratio_warn": 0.50,
      "tenengrad_p10_pass": 25,
      "tenengrad_p10_warn": 15,
      "tenengrad_median_pass": 30,
      "tenengrad_median_warn": 18
    }
  },
  "errors": []
}
```

`video_quality.evaluation.decision` 与 `qc_summary.status` 同步，取值为 `pass | warn | fail`。`should_run_mask_qc` 是 pipeline 调度 flag，规则固定为：

```text
should_run_mask_qc = decision != "fail"
```

`video_quality.evaluation.reasons` 和 `warn_reasons` 使用稳定机器可读枚举，例如：

| reason | 含义 |
|---|---|
| `cannot_open_video` | 视频文件无法打开。 |
| `video_not_opened` | OpenCV 未能打开视频流。 |
| `fps_below_min` | FPS 低于阈值。 |
| `short_side_below_min` | 短边分辨率低于阈值。 |
| `long_side_below_min` | 长边分辨率低于阈值。 |
| `drop_frame_ratio_above_max` | 时间轴疑似丢帧比例超过 hard fail 阈值。 |
| `max_frame_gap_ms_above_max` | 最大帧间隔超过 hard fail 阈值。 |
| `sample_decode_ratio_below_min` | 抽样帧解码率低于阈值。 |
| `mean_over_dark_ratio_above_max` | 过暗帧比例高于阈值。 |
| `mean_over_exposed_ratio_above_max` | 过曝帧比例高于阈值。 |
| `laplacian_p10_below_min` | Laplacian 方差 p10 低于清晰度硬筛阈值。 |
| `laplacian_median_below_min` | Laplacian 方差中位数低于清晰度硬筛阈值。 |
| `laplacian_under_100_ratio_above_max` | Laplacian 方差低于 100 的帧比例高于阈值。 |
| `tenengrad_p10_below_min` | Tenengrad p10 低于清晰度硬筛阈值。 |
| `tenengrad_median_below_min` | Tenengrad 中位数低于清晰度硬筛阈值。 |
| `black_frame_ratio_above_max` | 黑帧率高于阈值。 |
| `frozen_frame_ratio_above_max` | 冻帧率高于阈值。 |
| `max_consecutive_frozen_sec_above_max` | 连续冻结时长超过阈值。 |
| `hdf5_frame_count_mismatch` | HDF5 帧数与视频帧数不一致。 |
| `hdf5_missing` | HDF5 缺失，且配置要求失败。 |
| `hdf5_unreadable` | HDF5 不可读，且配置要求失败。 |
| `hand_roi_severe_blur` | HDF5 21 点粗 bbox ROI 严重模糊。 |
| `hand_roi_blur_bad_frame_ratio_above_max` | hand ROI 模糊坏帧比例超过 hard fail 阈值。 |

## reference_quality

当前没有标准对照视频，建档时写：

```json
{
  "mode": "none",
  "reference_video_path": null,
  "vmaf": null,
  "note": "当前无标准对照视频，未计算 VMAF。"
}
```

如果未来有同内容标准对照视频，再把 `mode` 改成 `full_reference` 并补充 VMAF 等指标。

## 后续模块写入规则

- 后续模块只能更新自己负责的 block，不要重写整份 JSON。
- 更新前先读取已有 JSON，保留未知字段，避免覆盖其他模块结果。
- 模块完成后必须更新 `qc_summary.completed_modules`。
- 模块失败后必须更新 `qc_summary.failed_modules` 和 `qc_summary.reasons`。
- 不要在 JSON 里写入大体积数组、图像二进制、视频帧或 token。
- 需要生成批次 summary、表格、HTML/PDF 报告时，从 `quality_archive/*.json` 聚合生成。
