# 视频质量单视频 JSON 格式 v1

CLI 运行后，每个视频输出一个对应 ID 的 JSON 文件：

```text
<batch>/reports/video_quality/<asset_id>.json
```

其中 `<asset_id>` 来自视频文件名：`408817_video.mp4` 对应 `408817.json`。

## 顶层结构

```json
{
  "schema_version": "video_quality_report.v1",
  "asset_id": "408817",
  "source_files": {},
  "evaluation": {},
  "hdf5_text_info": {},
  "video_quality": {},
  "reference_quality": {}
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | string | 当前固定为 `video_quality_report.v1`。后续破坏性变更必须升版本。 |
| `asset_id` | string | 单条数据 ID。 |
| `source_files` | object | 输入文件路径信息。 |
| `evaluation` | object | 最终是否通过与失败原因。 |
| `hdf5_text_info` | object | HDF5 对齐状态和文本字段。 |
| `video_quality` | object | 无标准对照的视频质量检测结果。 |
| `reference_quality` | object | 标准对照质量指标预留位。当前固定为无对照。 |

## source_files

```json
{
  "video": {
    "path": "/abs/path/to/408817_video.mp4",
    "filename": "408817_video.mp4",
    "extension": ".mp4"
  },
  "hdf5": {
    "path": "/abs/path/to/408817_hdf5.hdf5",
    "exists": true
  }
}
```

## evaluation

```json
{
  "passed": true,
  "reasons": []
}
```

`reasons` 使用稳定机器可读枚举，例如：

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

## hdf5_text_info

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
    "attributes": {
      "/": {
        "task": "pick up red cup"
      },
      "/meta": {
        "scene": "kitchen"
      }
    },
    "datasets": {
      "/meta/instruction": "move the cup to the tray"
    }
  }
}
```

`alignment.status` 可选值：

| status | 含义 |
|---|---|
| `matched` | 找到 HDF5，且 `label/quality_hand` 第一维帧数与视频帧数一致。 |
| `mismatch` | 找到 HDF5，但帧数不一致。 |
| `missing` | 未找到对应 HDF5。 |
| `unreadable` | HDF5 存在但不可读，或缺少 `label/quality_hand`。 |

`text_fields` 只收集字符串型 HDF5 attribute 和 dataset，不收集图像、数组、关键点等大体积数值数据。如果字符串内容本身是 JSON object 或 array，会解析成嵌套 JSON；普通文本保持 string。

## video_quality

```json
{
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

当前质量检测是无标准对照检测：不依赖 VMAF，也不要求同内容参考视频。

## reference_quality

```json
{
  "mode": "none",
  "reference_video_path": null,
  "vmaf": null,
  "note": "当前无标准对照视频，未计算 VMAF。"
}
```

该块用于未来接入标准对照视频时扩展。当前 `mode` 固定为 `none`。
