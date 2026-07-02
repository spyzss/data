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
    "reasons": []
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
  "reasons": []
}
```

模块执行后示例：

```json
{
  "status": "passed",
  "passed": true,
  "completed_modules": ["video_quality"],
  "failed_modules": [],
  "reasons": []
}
```

字段规则：

| 字段 | 类型 | 规则 |
|---|---|---|
| `status` | string | `pending`、`passed`、`failed` 三选一。 |
| `passed` | boolean/null | `pending` 时为 `null`；有 QC 结论后为 boolean。 |
| `completed_modules` | string[] | 已完成并写入结果的模块名。 |
| `failed_modules` | string[] | 有失败结论的模块名。 |
| `reasons` | string[] | 全局失败原因，格式建议为 `<module>.<reason>`。 |

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
    "frame_count_source": "label/quality_hand",
    "frame_count": 913,
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

该 block 由视频质量模块写入。未执行前为 `null`。

```json
{
  "evaluation": {
    "passed": true,
    "reasons": []
  },
  "metadata": {
    "opened": true,
    "frame_count": 913,
    "fps": 29.987814371159146,
    "duration_seconds": 30.4457,
    "width": 1280,
    "height": 720
  },
  "sampling": {
    "sample_count_configured": 10,
    "sampled_frame_count": 10,
    "decoded_sample_count": 10,
    "sample_decode_ratio": 1.0
  },
  "metrics": {
    "brightness": {
      "mean": 141.94834483506946,
      "black_frame_ratio": 0.0,
      "black_frame_brightness_threshold": 16
    },
    "exposure": {
      "mean_over_dark_ratio": 0.00017957899305555555,
      "mean_over_exposed_ratio": 0.00016731770833333328,
      "dark_pixel_threshold": 16,
      "over_exposed_pixel_threshold": 245
    },
    "sharpness": {
      "mean_blur_laplacian_var": 69.71875739224541
    },
    "temporal": {
      "frozen_frame_ratio": 0.0,
      "frozen_frame_mean_abs_diff_threshold": 1.0
    }
  },
  "thresholds": {
    "min_fps": 1.0,
    "min_width": 1,
    "min_height": 1,
    "min_sample_decode_ratio": 1.0,
    "max_mean_over_dark_ratio": 0.1,
    "max_mean_over_exposed_ratio": 0.05,
    "min_mean_blur_laplacian_var": 1.0,
    "max_black_frame_ratio": 0.05,
    "max_frozen_frame_ratio": 0.8,
    "fail_on_hdf5_frame_mismatch": true
  },
  "errors": []
}
```

`video_quality.evaluation.reasons` 使用稳定机器可读枚举，例如：

| reason | 含义 |
|---|---|
| `cannot_open_video` | 视频文件无法打开。 |
| `video_not_opened` | OpenCV 未能打开视频流。 |
| `fps_below_min` | FPS 低于阈值。 |
| `width_below_min` | 宽度低于阈值。 |
| `height_below_min` | 高度低于阈值。 |
| `sample_decode_ratio_below_min` | 抽样帧解码率低于阈值。 |
| `mean_over_dark_ratio_above_max` | 平均过暗像素占比高于阈值。 |
| `mean_over_exposed_ratio_above_max` | 平均过曝像素占比高于阈值。 |
| `mean_blur_laplacian_var_below_min` | 模糊代理指标低于阈值。 |
| `black_frame_ratio_above_max` | 黑帧率高于阈值。 |
| `frozen_frame_ratio_above_max` | 冻帧率高于阈值。 |
| `hdf5_frame_count_mismatch` | HDF5 帧数与视频帧数不一致。 |
| `hdf5_missing` | HDF5 缺失，且配置要求失败。 |
| `hdf5_unreadable` | HDF5 不可读，且配置要求失败。 |

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
