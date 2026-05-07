# DOM-only 主模型推理前最终方案

## 一、方案目标

本方案只服务于 **DOM-only 主线**，目标是在主模型推理前建立一套稳定、分层、可扩展的输入处理与策略编译机制，使系统能够对**任意尺寸、任意非整数倍宽高、不同质量风险、不同局部复杂度**的 DOM 做差异化处理。

最终不是把整张 DOM 直接丢给模型，也不是把所有 tile 都当成完全独立的小图做重分析，而是采用：

```text
一级：全局输入契约
二级：block 级局部分析与默认策略
三级：tile 级轻量检查与局部修正
```

核心思想：

```text
全局统一约束；
block 差异化策略；
tile 局部异常修正。
```

------

## 二、三层总体架构

### 1. 一级：`dom_input_contract`

一级是**全局输入契约层**，作用不是分析影像内容，而是保证：

```text
DOM 能不能进系统；
怎么进系统；
怎么稳定生成 block / tile 任务；
怎么保证输出能准确回写到原图。
```

一级对整张 DOM 只生成一份，全局统一，供所有 block 继承。

------

### 2. 二级：`processing_block_profile`

二级是 **block 级局部分析与默认策略层**。
block 不是模型输入，而是：

```text
IO 读取单位；
缓存单位；
局部分析单位；
断点续跑单位；
默认策略生成单位。
```

每个 block 都要根据自身影像内容单独计算二级信息，因此不同 block 的结果通常不同。

二级的工程意义必须落到“改策略”上，否则没有必要每个 block 都计算一次。

------

### 3. 三级：`TileRunContext`

三级是 **tile 级执行上下文层**。
tile 才是模型真实输入窗口，例如 `1024×1024`。

三级不做完整重分析，只做轻量快速检查，并在必要时对 block 默认策略做有限覆盖。

原则：

```text
同一 block 内的大多数 tile：
直接继承 block 默认策略

同一 block 内的异常 tile：
只做轻量覆盖，不重复完整分析
```

------

# 三、默认工程参数

建议把参数划分成三类：**全局契约参数、block 策略参数、tile 轻量修正参数**。

------

## 3.1 全局契约参数

这些参数对同一张 DOM 全局统一，不按 block 改。

```yaml
gsd_policy:
  recommended_gsd_m: 0.02
  acceptable_gsd_range_m: [0.015, 0.05]
  resample_if_finer_than_m: 0.015
  warn_if_coarser_than_m: 0.05

processing_block_policy:
  block_px: 5632
  block_stride_px: 5120
  block_overlap_px: 512
  edge_absorb_px: 512
  block_min_preferred_px: 5120
  block_max_preferred_px: 6144

tile_execution_contract:
  tile_px: 1024
  overlap_px: 256
  stride_px: 768
  allow_elastic_model_input: false
  pad_if_smaller_than_model_input: true
  snap_last_tile_to_edge: true
  discard_padding_output: true

stage2_fixed_contract:
  bsize: 256
```

### 说明

#### `block_px = 5632`

表示 block 是分析和调度尺寸，不是模型输入尺寸。

#### `tile_px = 1024`

表示 tile 才是模型真实输入尺寸。

#### `bsize = 256`

固定，不参与动态调整。

------

## 3.2 block 级动态核心参数

这些参数**应该根据每个 block 的二级分析结果动态生成**：

```yaml
block_dynamic_core_params:
  - diam_list
  - augment
  - iou_merge_thr
```

其中：

- `diam_list`：最重要，控制尺度适配
- `augment`：控制稳健性与时间开销
- `iou_merge_thr`：控制合并行为，建议小范围动态调整

------

## 3.3 tile 级有限覆盖参数

这些参数由 tile 快速检查决定，可在 block 默认策略基础上轻量覆盖：

```yaml
tile_light_override_params:
  - skip
  - skip_reason
  - augment
  - diam_list
  - fusion_priority
  - export_sample_flag
  - finetune_candidate_flag
```

不建议 tile 级动态调整：

```yaml
tile_do_not_override:
  - bsize
  - tile_px
  - overlap_px
  - stride_px
```

------

# 四、三层信息流

最终推荐的信息流公式：

```text
FinalTilePolicy =
    GlobalInputContract
  + BlockDynamicPolicy
  + TileFastDelta
```

解释：

### `GlobalInputContract`

整张 DOM 统一继承，保证稳定执行和正确回写。

### `BlockDynamicPolicy`

每个 block 根据二级分析结果动态生成的默认推理策略。

### `TileFastDelta`

tile 级快速检查发现局部偏离时，对默认策略做轻量修正。

------

# 五、一级方案：`dom_input_contract`

## 5.1 一级定位

一级只做“全局输入契约”，不做复杂内容分析。

一级主要回答：

```text
文件可读吗？
空间参考完整吗？
GSD 是否合理？
波段如何映射？
nodata / valid mask 怎么处理？
block / tile 怎么生成？
输出怎么裁回原图？
```

------

## 5.2 一级硬验证

以下失败则停止：

```text
文件可读
width > 0
height > 0
CRS 存在
transform 存在
bounds 合法
GSD 可计算
至少可构造 RGB
dtype 可归一化
nodata / valid mask 可识别
```

------

## 5.3 一级软验证

以下只记 warning，不停止：

```text
GSD 过细
GSD 过粗
大面积 nodata / 黑边
band_count 异常
dtype 需要特殊归一化
transform 有旋转
DOM 面积过大
```

------

## 5.4 一级预处理原则

```text
原始 DOM 不改；
生成 working DOM / VRT；
生成 valid mask；
保留 working_to_original_transform；
任何输出都必须能回写到原图。
```

------

## 5.5 一级输出 JSONC 模板：`dom_input_contract.jsonc`

```jsonc
{
  // =========================
  // 一、DOM 身份与来源信息
  // =========================

  "dom_id": "dom_001", // 当前 DOM 的唯一标识，用于贯穿 block、tile、推理结果、记忆与评估
  "source_path": "/data/dom/dom_001.tif", // 原始输入 DOM 文件路径
  "working_dom_path": "/workspace/dom/dom_001_working.vrt", // 预处理后供系统读取的工作 DOM 路径，可为 VRT 或 TIF
  "input_type": "geotiff", // 输入类型，如 geotiff、vrt、directory_manifest
  "mainline_profile": "A_DOM_ONLY", // 当前主线模式，这里固定为 DOM-only

  // =========================
  // 二、空间与尺寸信息
  // =========================

  "width": 50000, // 原始 DOM 宽度，单位为像素
  "height": 48000, // 原始 DOM 高度，单位为像素
  "pixel_count": 2400000000, // 总像素数，等于 width × height
  "bounds": [100.0, 30.0, 101.0, 31.0], // DOM 空间范围，格式一般为 [minx, miny, maxx, maxy]
  "crs": "EPSG:4547", // 坐标参考系统
  "transform": [0.02, 0.0, 100.0, 0.0, -0.02, 31.0], // 仿射变换参数，用于像素坐标与地理坐标转换
  "working_to_original_transform": null, // 若 working DOM 与原图有重采样或旋转修正，则记录其到原图的映射关系

  // =========================
  // 三、分辨率与 GSD 信息
  // =========================

  "gsd_x_m": 0.02, // X 方向地面分辨率，单位米/像素
  "gsd_y_m": 0.02, // Y 方向地面分辨率，单位米/像素
  "recommended_gsd_m": 0.02, // 系统推荐 GSD
  "acceptable_gsd_range_m": [0.015, 0.05], // 系统接受的 GSD 范围
  "gsd_status": "acceptable", // GSD 状态，可取 acceptable / too_fine / too_coarse / unknown

  // =========================
  // 四、波段与像元类型
  // =========================

  "band_count": 3, // 波段数量
  "dtype": "uint8", // 栅格数据类型，如 uint8、uint16、float32
  "band_mapping": {
    "red": 1, // 红波段对应原始栅格中的波段索引
    "green": 2, // 绿波段对应原始栅格中的波段索引
    "blue": 3 // 蓝波段对应原始栅格中的波段索引
  },
  "normalization_policy": "uint8_passthrough", // 归一化策略，如 uint8_passthrough / uint16_to_uint8 / percentile_clip_then_scale

  // =========================
  // 五、有效区域与 nodata
  // =========================

  "nodata": 0, // 原始 nodata 值
  "nodata_policy": "use_valid_mask", // nodata 处理策略，如 use_valid_mask / direct_nodata / infer_from_border
  "valid_mask_path": "/workspace/dom/dom_001_valid_mask.tif", // 有效区域掩膜路径
  "global_valid_pixel_ratio_estimate": 0.96, // 全图有效像元占比估计，用于快速预判空白程度

  // =========================
  // 六、全局执行契约
  // =========================

  "processing_block_px": 5632, // block 分析与调度尺寸，不是模型输入尺寸
  "processing_block_stride_px": 5120, // block 滑动步长
  "processing_block_overlap_px": 512, // 相邻 block 重叠像素
  "processing_edge_absorb_px": 512, // 边缘不足时的吸收阈值
  "processing_block_min_preferred_px": 5120, // 边缘 block 的推荐最小尺寸
  "processing_block_max_preferred_px": 6144, // 边缘 block 的最大允许尺寸

  "tile_px": 1024, // 模型真实输入 tile 尺寸
  "tile_overlap_px": 256, // 相邻 tile 重叠像素
  "tile_stride_px": 768, // tile 步长，通常等于 tile_px - tile_overlap_px
  "allow_elastic_model_input": false, // 是否允许模型输入尺寸弹性变化，这里固定为 false
  "pad_if_smaller_than_model_input": true, // 边缘不足 1024 时是否 padding
  "snap_last_tile_to_edge": true, // 最后一个 tile 是否贴边，避免漏覆盖
  "discard_padding_output": true, // padding 区域的输出是否丢弃不写回

  "bsize": 256, // Stage2 固定内部处理尺寸，属于稳定执行约束，不动态调整

  // =========================
  // 七、执行模式与资源预估
  // =========================

  "processing_mode": "block_then_sliding_window", // 执行模式，如 single_tile / sliding_window_only / block_then_sliding_window
  "estimated_block_count": 81, // 预计 block 数量
  "estimated_tile_count": 3969, // 预计 tile 数量
  "output_clip_policy": "clip_to_original_bounds", // 结果输出裁剪策略，确保不写出原图范围

  // =========================
  // 八、状态与告警
  // =========================

  "warnings": [
    "large_dom_enable_resume"
  ], // 全局 warning 列表
  "status": "ready" // 当前输入契约状态，如 ready / warning / invalid
}
```

------

# 六、二级方案：`processing_block_profile`

## 6.1 二级定位

每个 block 要自己计算局部信息，并生成自己的默认策略。

二级必须回答：

```text
这个 block 的影像条件如何？
这个 block 是否复杂、阴影重、模糊、低纹理、过曝？
这个 block 内部是否异质性很强？
这个 block 默认应该怎么推理？
```

------

## 6.2 二级重分析内容

建议每个 block 计算：

```text
valid_pixel_ratio
brightness_mean/std
shadow_ratio
overexposed_ratio
underexposed_ratio
blur_score
gradient_mean/std
texture_complexity_score
block_heterogeneity_score
risk_tags
expected_failure_modes
```

------

## 6.3 二级异质性检测

block 内部不一定均匀，因此除了均值，还要计算异质性。

推荐 coarse grid：

```text
7×7 或 8×8 的轻量子网格
```

每个 coarse cell 只算轻量指标，不做完整重分析。

最后给出：

```text
block_heterogeneity_score
block_heterogeneity_level
localized_risk_tags
```

------

## 6.4 二级动态默认策略

这是二级最关键的产物。

每个 block 根据分析结果生成：

```text
diam_list
augment
iou_merge_thr
enable_tile_fast_check
fusion_priority
expert_model_candidates
```

建议从有限模板中选，而不是完全自由生成。

------

## 6.5 二级输出 JSONC 模板：`processing_block_profile.jsonc`

> 这里给的是**单个 block 的对象模板**。实际工程中可保存为：
>
> - `processing_block_profile.jsonl`：每行一个 block
> - 或 `processing_block_profile/xxx.json`

```jsonc
{
  // =========================
  // 一、block 身份信息
  // =========================

  "block_id": "dom_001_b_0001", // block 唯一标识
  "dom_id": "dom_001", // 所属 DOM 的唯一标识
  "block_index": 1, // block 顺序编号

  // =========================
  // 二、block 空间与窗口信息
  // =========================

  "block_window": [0, 0, 5632, 5632], // block 在 DOM 像素坐标中的窗口，格式为 [x, y, width, height]
  "block_geo_bounds": [100.0, 30.8, 100.11264, 30.91264], // block 对应的地理范围
  "width": 5632, // block 实际宽度
  "height": 5632, // block 实际高度
  "edge_block_flag": false, // 是否为边缘 block
  "overlap_with_neighbors_px": 512, // 与相邻 block 的重叠像素

  // =========================
  // 三、block 有效区域信息
  // =========================

  "valid_pixel_ratio": 0.91, // 当前 block 的有效像元占比
  "skip_block_candidate": false, // 是否可作为跳过候选 block，例如几乎全空白时为 true
  "low_valid_area_flag": false, // 是否属于低有效区域 block

  // =========================
  // 四、质量指标
  // =========================

  "brightness_mean": 122.5, // block 平均亮度
  "brightness_std": 31.2, // block 亮度标准差
  "shadow_ratio_estimate": 0.18, // 阴影区域占比估计
  "overexposed_ratio": 0.02, // 过曝像元占比
  "underexposed_ratio": 0.04, // 欠曝像元占比
  "laplacian_variance": 145.0, // 拉普拉斯方差，用于衡量清晰度
  "tenengrad": 0.73, // 梯度清晰度指标
  "blur_score": 0.27, // 模糊风险分数，越高表示越模糊
  "stripe_noise_score": 0.08, // 条带噪声评分
  "stripe_noise_direction": "none", // 条带噪声方向，可为 row / col / none
  "color_cast_score": 0.12, // 色偏评分

  // =========================
  // 五、纹理与结构指标
  // =========================

  "gradient_mean": 0.36, // 平均梯度强度
  "gradient_std": 0.14, // 梯度波动强度
  "texture_entropy": 5.2, // 纹理熵，反映复杂度
  "texture_contrast": 0.41, // 纹理对比度
  "texture_homogeneity": 0.52, // 纹理同质性
  "texture_complexity_score": 0.81, // 综合纹理复杂度评分
  "low_texture_flag": false, // 是否属于低纹理 block
  "dense_texture_flag": true, // 是否属于高纹理或密集纹理 block

  // =========================
  // 六、block 内异质性信息
  // =========================

  "heterogeneity_coarse_grid": [7, 7], // 用于评估 block 内异质性的轻量网格尺寸
  "brightness_variance_across_cells": 0.42, // 子网格之间的亮度方差
  "shadow_spatial_variance": 0.55, // 子网格之间阴影分布差异
  "gradient_variance_across_cells": 0.37, // 子网格之间梯度差异
  "valid_ratio_variance_across_cells": 0.08, // 子网格之间有效像元占比差异
  "block_heterogeneity_score": 0.78, // block 综合异质性分数
  "block_heterogeneity_level": "high", // 异质性等级，可为 low / medium / high

  // =========================
  // 七、风险标签与失败预期
  // =========================

  "risk_tags": [
    "dense_texture",
    "moderate_shadow"
  ], // block 全局风险标签

  "localized_risk_tags": [
    "local_shadow_patch"
  ], // block 内局部风险标签，不代表整个 block 都如此

  "quality_class": "medium_risk", // block 质量类别
  "priority_score": 0.72, // block 调度优先级分数
  "expected_failure_modes": [
    "crown_merge",
    "under_segmentation"
  ], // 预期失败模式

  // =========================
  // 八、block 默认推理策略
  // =========================

  "policy_template_name": "dense_small_crown", // block 选用的策略模板名称
  "diam_list": "64,96,160", // 当前 block 默认尺度候选
  "augment": true, // 当前 block 默认是否开启增强推理
  "iou_merge_thr": 0.30, // 当前 block 默认合并阈值
  "enable_tile_fast_check": true, // 当前 block 内 tile 是否启用快速轻量检查
  "fusion_priority": "normal", // 当前 block 默认融合优先级

  // =========================
  // 九、专家模型与样本策略
  // =========================

  "expert_model_candidates": [
    "dense_crown_expert"
  ], // block 候选专家模型列表，现阶段可只保留接口
  "memory_candidate_policy": "high_risk_only", // 记忆候选策略
  "finetune_candidate_policy": "failed_or_corrected_only", // 微调候选策略

  // =========================
  // 十、tile 级统计摘要
  // =========================

  "expected_tile_count": 49, // 当前 block 内预计 tile 数量
  "empty_tile_estimate": 3, // 预计空白 tile 数量
  "high_risk_tile_estimate": 8, // 预计高风险 tile 数量

  // =========================
  // 十一、状态信息
  // =========================

  "status": "ready" // 当前 block 状态，可为 ready / skip / warning / failed
}
```

------

# 七、三级方案：`TileRunContext`

## 7.1 三级定位

三级不是完整 profile，而是**当前 tile 的执行上下文**。

主要回答：

```text
这个 tile 从哪里读？
这个 tile 是否 padding？
这个 tile 输出写回哪里？
这个 tile 是否为空白？
这个 tile 是否偏离 block 默认判断？
最终用什么参数推理？
```

------

## 7.2 三级轻量检查内容

默认只做：

```text
valid_pixel_ratio
empty_tile_flag
padding_ratio
edge_tile_flag
```

在 block 开启 `enable_tile_fast_check = true` 时，再增加：

```text
brightness_proxy
shadow_proxy
gradient_proxy
local_texture_proxy
```

------

## 7.3 三级覆盖原则

### 差异不大

直接继承 block 默认策略。

### 差异明显

只做轻量覆盖，例如：

```text
skip
augment
diam_list
fusion_priority
export_sample_flag
finetune_candidate_flag
```

------

## 7.4 三级输出 JSONC 模板：`TileRunContext.jsonc`

> 这里给的是**单个 tile 的对象模板**。普通 tile 不一定要完整落盘，建议只对异常 tile、失败 tile、记忆/微调候选 tile 保存。

```jsonc
{
  // =========================
  // 一、tile 身份信息
  // =========================

  "tile_id": "dom_001_b_0001_t_0032", // tile 唯一标识
  "dom_id": "dom_001", // 所属 DOM 标识
  "block_id": "dom_001_b_0001", // 所属 block 标识
  "tile_index": 32, // tile 在 block 内的顺序编号

  // =========================
  // 二、tile 空间执行契约
  // =========================

  "read_window": [2304, 1536, 1024, 1024], // 从 working DOM 读取的像素窗口 [x, y, width, height]
  "model_window": [0, 0, 1024, 1024], // 模型输入窗口，通常固定为 [0, 0, 1024, 1024]
  "valid_write_window": [2304, 1536, 768, 768], // 模型输出最终允许写回原图的有效区域窗口
  "pad_left": 0, // 左侧 padding 像素
  "pad_top": 0, // 上侧 padding 像素
  "pad_right": 0, // 右侧 padding 像素
  "pad_bottom": 0, // 下侧 padding 像素
  "padding_ratio": 0.0, // 当前 tile 中 padding 区域占比
  "edge_tile_flag": false, // 是否属于边缘 tile
  "clip_to_valid_write_window": true, // 输出是否裁到 valid_write_window
  "discard_padding_output": true, // padding 区域输出是否丢弃

  // =========================
  // 三、从一级统一继承的信息
  // =========================

  "working_dom_path": "/workspace/dom/dom_001_working.vrt", // 供读取的工作 DOM 路径
  "valid_mask_path": "/workspace/dom/dom_001_valid_mask.tif", // 有效区域掩膜路径
  "crs": "EPSG:4547", // 坐标参考系统
  "transform_ref": "original_transform", // 当前 tile 写回时使用的 transform 参考
  "gsd_m": 0.02, // 当前 DOM 的地面分辨率
  "gsd_status": "acceptable", // GSD 状态
  "band_mapping": {
    "red": 1, // 红波段映射
    "green": 2, // 绿波段映射
    "blue": 3 // 蓝波段映射
  },
  "normalization_policy": "uint8_passthrough", // 像元归一化策略
  "nodata_policy": "use_valid_mask", // nodata 处理策略

  // =========================
  // 四、从二级 block 默认继承的信息
  // =========================

  "inherited_risk_tags": [
    "dense_texture",
    "moderate_shadow"
  ], // 从 block 继承的默认风险标签

  "inherited_quality_class": "medium_risk", // 从 block 继承的质量类别
  "inherited_priority_score": 0.72, // 从 block 继承的优先级分数
  "inherited_block_heterogeneity_level": "high", // 从 block 继承的异质性等级
  "inherited_expected_failure_modes": [
    "crown_merge",
    "under_segmentation"
  ], // 从 block 继承的预期失败模式

  "inherited_diam_list": "64,96,160", // 从 block 默认继承的 diam_list
  "inherited_augment": true, // 从 block 默认继承的 augment
  "inherited_iou_merge_thr": 0.30, // 从 block 默认继承的合并阈值
  "inherited_fusion_priority": "normal", // 从 block 默认继承的融合优先级
  "enable_tile_fast_check": true, // 当前 tile 是否启用快速轻量检查

  // =========================
  // 五、tile 级快速轻量特征
  // =========================

  "valid_pixel_ratio": 0.96, // 当前 tile 的有效像元占比
  "empty_tile_flag": false, // 是否为空白 tile
  "brightness_proxy": 0.58, // tile 级亮度代理特征，仅用于快速判断
  "shadow_proxy": 0.21, // tile 级阴影代理特征，仅用于快速判断
  "gradient_proxy": 0.74, // tile 级梯度代理特征，仅用于快速判断
  "local_texture_proxy": 0.69, // tile 级局部纹理代理特征，仅用于快速判断

  // =========================
  // 六、tile 局部修正结果
  // =========================

  "tile_delta_detected": false, // 是否检测到当前 tile 与 block 默认判断有明显偏差
  "tile_delta_reason": [], // 触发局部修正的原因列表
  "tile_risk_tags": [
    "dense_texture",
    "moderate_shadow"
  ], // 当前 tile 最终风险标签，可能等于 inherited_risk_tags，也可能在此基础上增加局部标签

  // =========================
  // 七、最终推理策略
  // =========================

  "skip": false, // 当前 tile 是否跳过推理
  "skip_reason": null, // 跳过原因，例如 low_valid_area / fully_empty

  "final_diam_list": "64,96,160", // 当前 tile 最终用于推理的 diam_list
  "final_augment": true, // 当前 tile 最终是否开启增强推理
  "final_iou_merge_thr": 0.30, // 当前 tile 最终使用的合并阈值
  "final_bsize": 256, // 当前 tile 最终使用的 bsize，固定不变
  "final_fusion_priority": "normal", // 当前 tile 最终融合优先级

  // =========================
  // 八、样本与专家路由
  // =========================

  "expert_model_name": null, // 当前 tile 是否路由到专家模型，未启用时为 null
  "export_sample_flag": false, // 是否导出该 tile 作为样本
  "memory_candidate_flag": false, // 是否进入记忆候选
  "finetune_candidate_flag": false, // 是否进入微调候选

  // =========================
  // 九、状态信息
  // =========================

  "status": "ready" // 当前 tile 执行状态，可为 ready / skipped / failed / expert_routed
}
```

------

# 八、block 默认策略如何动态生成

建议采用**有限模板选择**，不要完全自由生成。

例如：

```yaml
policy_templates:
  default:
    diam_list: "96,192,320"
    augment: false
    iou_merge_thr: 0.28
    enable_tile_fast_check: false

  dense_small_crown:
    diam_list: "64,96,160"
    augment: true
    iou_merge_thr: 0.30
    enable_tile_fast_check: true

  large_sparse_crown:
    diam_list: "128,256,320"
    augment: false
    iou_merge_thr: 0.24
    enable_tile_fast_check: false

  shadow_weak_boundary:
    diam_list: "96,192,320"
    augment: true
    iou_merge_thr: 0.28
    enable_tile_fast_check: true

  high_heterogeneity:
    diam_list: "64,128,256,320"
    augment: true
    iou_merge_thr: 0.28
    enable_tile_fast_check: true
```

------

## 8.1 二级到策略模板的映射

### 情况 1：密集小冠幅

适合：

```text
dense_texture 高
gradient 高
局部冠层粘连风险高
```

输出：

```text
dense_small_crown
```

------

### 情况 2：稀疏大冠幅

适合：

```text
纹理不密
目标尺度偏大
大冠幅区域明显
```

输出：

```text
large_sparse_crown
```

------

### 情况 3：阴影重、弱边界

适合：

```text
shadow_ratio 高
low_texture
blur 或 weak_boundary 风险明显
```

输出：

```text
shadow_weak_boundary
```

------

### 情况 4：异质性高

适合：

```text
block_heterogeneity_level = high
局部差异明显
```

输出：

```text
high_heterogeneity
```

------

# 九、tile 级轻量覆盖规则

tile 级只在局部偏差明显时覆盖 block 默认策略。

------

## 9.1 跳过规则

```text
if valid_pixel_ratio < 0.05:
    skip = true
```

------

## 9.2 阴影局部覆盖

```text
if shadow_proxy 显著高于 block 均值:
    final_augment = true
    tile_risk_tags += ["local_shadow"]
```

------

## 9.3 小冠幅局部覆盖

```text
if local_texture_proxy 很高 且 gradient_proxy 很高:
    final_diam_list = "64,96,160"
```

------

## 9.4 大冠幅局部覆盖

```text
if local_texture_proxy 较低 且结构尺度偏大:
    final_diam_list = "128,256,320"
```

------

## 9.5 边缘 tile 融合降权

```text
if padding_ratio > 0.3:
    final_fusion_priority = "low"
```

------

# 十、落盘建议

## 10.1 必须落盘

```text
dom_input_contract.json
processing_block_profile.jsonl 或分 block json
inference_tile_plan.csv
preflight_report.json
```

------

## 10.2 非必须全量落盘

`TileRunContext` 不建议所有 tile 都保存。

建议只保存：

```text
skip tile
edge tile
high-risk tile
tile_delta_detected = true 的 tile
推理失败 tile
记忆候选 tile
微调候选 tile
专家模型接管 tile
```

------

# 十一、最终执行流程

```text
1. 读取 DOM
2. 生成 dom_input_contract
3. 生成 block plan
4. 遍历每个 block
5. 计算 processing_block_profile
6. 生成该 block 的默认策略
7. 在 block 内生成 tile plan
8. 遍历每个 tile
9. 生成 TileRunContext
10. 若 tile 与 block 默认判断差异不大，则直接继承
11. 若差异明显，则轻量覆盖部分参数
12. 调用主模型推理
13. 裁掉 padding 输出
14. 写回 valid_write_window
15. 在 block 内融合
16. 拼回全图
17. 记录高风险 / 失败 / 候选样本
```

------

# 十二、最终结论

这版最终方案的核心不是“多做一级、二级、三级信息”，而是把三层信息各自的职责钉死：

```text
一级负责统一契约；
二级负责 block 差异化默认策略；
三级负责 tile 异常轻量修正。
```

最终你要的就是这一句：

```text
让同一张 DOM 的不同区域拥有不同 block 级默认推理策略；
让同一 block 内的异常 tile 有机会局部修正；
但不让每个 tile 都重复完整分析。
```

如果你下一步要，我可以继续把这版方案直接整理成：
**“适合放进你项目文档的正式版 Markdown”**，或者进一步整理成 **字段表格版**。









------

# 十三、【新增】关于 block 切分方式、二级完整信息提取与主模型推理前读取逻辑的补充说明

## 13.1 【新增】block 的默认实现方式：逻辑切分，不默认物理裁剪为 block.tif

本方案中的 `block_px = 5632` 默认应理解为：

```text
逻辑 block / 调度 block / 分析 block / 融合 block
```

而不是：

```text
必须先把整张 DOM 物理裁剪成一批 5632×5632 的 block.tif 文件
```

也就是说，系统在默认工程实现中应优先采用：

```text
原始 DOM 保持不动
→ 生成 working DOM / working VRT
→ 生成 logical block plan
→ 每个 block 只保存 block_window / block_geo_bounds / block_id / profile / policy
→ 真正需要分析或推理时，再按 window 从 working DOM / VRT 中读取对应范围数据
```

因此，block 仍然保留其原有职责：

```text
IO 读取单位；
缓存单位；
局部分析单位；
断点续跑单位；
默认策略生成单位；
局部融合单位。
```

但这里的 “IO 读取单位” 指的是：

```text
按 block_window 从原图 / VRT 中读取该范围数据
```

而不是必须先导出独立 block 文件。

---

## 13.2 【新增】为什么采用逻辑切分

默认采用逻辑切分的原因是：

```text
1. 避免先切 block.tif 再重复读取带来的额外磁盘 IO；
2. 避免产生大量中间小文件，减少工程管理负担；
3. 保留 block 级分析、策略生成、断点续跑和局部融合能力；
4. 更适合大范围 DOM（尤其是几十平方公里级）的稳定执行。
```

因此，本方案推荐的默认理解应为：

```text
不做 block 级默认物理落盘；
只做 block 级逻辑切分与窗口读取。
```

但在以下场景下，允许按需导出 block 文件：

```text
调试失败 block；
失败复现；
样本导出；
微调候选沉淀；
分布式调度 / 多机分发。
```

---

## 13.3 【新增】逻辑 block 的最小数据结构

推荐每个 block 至少保存以下字段：

```jsonc
{
  "block_id": "dom_001_b_0001",
  "block_index": 1,
  "block_window": [0, 0, 5632, 5632],
  "block_geo_bounds": [100.0, 30.8, 100.11264, 30.91264],
  "width": 5632,
  "height": 5632,
  "edge_block_flag": false,
  "status": "ready"
}
```

其中最核心的是：

```text
block_window
block_geo_bounds
```

因为它们决定了：
- 二级信息从哪里读取
- block 内 tile 任务如何生成
- 推理结果最终写回哪里
- block 间融合范围如何确定

---

## 13.4 【新增】二级完整信息提取的原则：不落盘 block.tif，但必须完整读取 block 范围数据

如果要严格按照本文件中的二级方案，完整提取以下信息：

```text
valid_pixel_ratio
brightness_mean / std
shadow_ratio_estimate
overexposed_ratio
underexposed_ratio
laplacian_variance
tenengrad
blur_score
gradient_mean / std
texture_entropy
texture_contrast
texture_homogeneity
texture_complexity_score
block_heterogeneity_score
block_heterogeneity_level
risk_tags
localized_risk_tags
expected_failure_modes
policy_template_name
diam_list
augment
iou_merge_thr
```

则逻辑切分方案下必须满足：

```text
对每个 block，不需要先生成 block.tif；
但必须根据 block_window，将该 block 范围内的影像数据完整读取出来，再在内存中完成二级分析。
```

也就是说：

```text
逻辑切分 ≠ 不读取 block 数据
逻辑切分 = 不预先落盘 block.tif，但按 window 完整读取该 block 对应范围的数据
```

---

## 13.5 【新增】二级完整信息提取的标准实现方式

推荐实现方式如下：

```python
with rasterio.open(working_dom_path) as src:
    window = Window(block_x, block_y, block_w, block_h)
    block_rgb = src.read(
        indexes=[1, 2, 3],
        window=window
    )  # shape: (3, block_h, block_w)
```

如果 valid mask 独立保存，则同时读取：

```python
with rasterio.open(valid_mask_path) as msk:
    valid_mask = msk.read(
        1,
        window=window
    )  # shape: (block_h, block_w)
```

之后在 `block_rgb` 与 `valid_mask` 上，完整计算本文件定义的全部二级指标。

---

## 13.6 【新增】逻辑切分下，二级信息与物理 block.tif 提取结果的等价条件

逻辑切分下提取的二级信息，与先裁出物理 `block.tif` 再提取二级信息，在以下条件成立时，可视为等价：

```text
1. 读取的空间范围一致；
2. 数据源一致（均来自同一原始 DOM / working VRT）；
3. 未发生额外重采样；
4. 归一化策略一致；
5. 波段顺序一致；
6. nodata / valid mask 处理一致；
7. 边缘 block 的处理方式一致。
```

更准确地说：

```text
二级指标依赖的是该 block 对应范围内的像素内容；
并不依赖于这一范围是否被提前保存为一个 block.tif 文件。
```

因此，只要逻辑读取与物理切块读取的输入一致，则：

```text
影像质量特征；
纹理特征；
光谱与亮度特征；
异质性特征；
风险标签与默认策略结果
```

均可做到与物理 block.tif 路线一致或等价。

---

## 13.7 【新增】二级分析时的边界处理原则

二级分析阶段，推荐使用：

```text
真实有效 block 数据
```

而不是为了凑满固定尺寸，先对 block 做 padding 再分析。

原因是：

```text
padding 更适用于三级 tile 推理输入契约；
不适合作为二级 block 质量与纹理分析输入。
```

因此，对边缘 block：

```text
block_window 的 width / height 可以小于 5632；
二级信息提取应基于真实读取到的有效 block 数据进行。
```

---

## 13.8 【新增】主模型推理前的 block 内 tile 读取逻辑

主模型推理阶段，不需要先导出 `block.tif` 再在其中滑窗读取 tile。

推荐默认流程为：

```text
1. 已知某个 block 的 block_window = [bx, by, bw, bh]
2. 在该 block 内生成 tile local window
3. 将 tile local window 换算为整图上的 global read_window
4. 直接从 working DOM / VRT 中读取 tile
5. 送入主模型推理
6. 将输出按 valid_write_window 写回全图
```

换算关系如下：

```text
global_tile_x = block_x + local_tile_x
global_tile_y = block_y + local_tile_y
```

---

## 13.9 【新增】主模型推理前 tile 的标准读取方式

对于 block 内某个 tile，推荐使用：

```python
with rasterio.open(working_dom_path) as src:
    tile = src.read(
        indexes=[1, 2, 3],
        window=Window(global_tx, global_ty, 1024, 1024),
        boundless=True,
        fill_value=0
    )
```

这里：

- `global_tx / global_ty` 是 tile 在整张 working DOM 上的全局像素坐标；
- `1024 × 1024` 是模型真实输入尺寸；
- `boundless=True` 用于边缘 tile 的补齐读取；
- `fill_value=0` 表示超出边界部分填充为 0。

如需送入常见深度学习模型，还需按模型输入要求转为：

```text
H × W × C
```

或进一步转为张量格式。

---

## 13.10 【新增】逻辑读取 tile 与物理 tile.tif 推理输入的等价条件

对于主模型推理，真正决定结果是否一致的是：

```text
模型最终接收到的 tile 像素数组是否一致
```

因此，在以下条件成立时：

```text
1. tile 对应的空间窗口一致；
2. 数据源一致；
3. 波段顺序一致；
4. 无额外重采样；
5. 无额外拉伸 / dtype 变化；
6. 预处理一致；
7. 边缘 tile 的 padding 规则一致；
```

则以下两种路线对主模型输入可视为等价：

```text
路线 A：
原图 → 物理裁剪 tile.tif → 读取 tile.tif → 推理

路线 B：
原图 / working VRT → 直接按 window 读取 tile → 推理
```

对于非边缘 tile，二者通常可以认为是相同输入；

对于边缘 tile，只有当物理裁剪和逻辑读取采用相同的 padding 策略时，才能视为等价。

---

## 13.11 【新增】三级 TileRunContext 与逻辑窗口读取的对应关系

在逻辑切分方案下，三级 `TileRunContext` 中的关键字段应这样理解：

- `read_window`：从 working DOM / VRT 中实际读取 tile 的全局窗口
- `model_window`：送入模型的输入坐标系，通常固定为 `[0, 0, 1024, 1024]`
- `valid_write_window`：推理结果真正允许写回原图的有效区域
- `padding_ratio`：边缘 tile 中补齐区域占比
- `edge_tile_flag`：是否为边缘 tile
- `discard_padding_output`：padding 区域输出是否丢弃

因此，三级方案与逻辑切分不是冲突关系，而是天然匹配关系。

---

## 13.12 【新增】推荐的默认工程实现原则

推荐将本方案中的 block 切分、二级信息提取与主模型推理前读取逻辑统一为以下工程原则：

```text
1. 原始 DOM 保持不动；
2. 只生成 working DOM / working VRT；
3. 5632 仅作为逻辑 block window；
4. block 默认不物理裁剪为 block.tif；
5. 二级信息完整提取时，按 block_window 从 working DOM / VRT 读取完整 block 数据；
6. 1024 作为模型真实输入 tile；
7. tile 按 read_window 直接从 working DOM / VRT 读取；
8. 每个 block 在局部融合完成后及时落盘结果；
9. 全图仅保留必要索引和最终结果，不长期保留所有中间数组；
10. 只有在调试、失败复现、样本导出、微调沉淀、分布式调度时，才按需导出物理 block / tile 文件。
```

---

## 13.13 【新增】对“block_then_sliding_window”执行模式的修正解释

对于本文件中的：

```jsonc
"processing_mode": "block_then_sliding_window"
```

推荐在工程解释中统一理解为：

```text
logical_block_then_windowed_sliding
```

也就是：

```text
先生成逻辑 block；
再在每个 block 内做 tile 滑窗读取与推理；
而不是默认先物理切出 block 文件再做滑窗。
```

---

## 13.14 【新增】最终执行流程（修正版）

在保持本文件原有三级结构不变的前提下，推荐将执行流程明确为：

```text
1. 读取原始 DOM
2. 生成 dom_input_contract
3. 生成 working DOM / working VRT
4. 生成 logical block plan
5. 遍历每个 block
6. 根据 block_window 从 working DOM / VRT 中读取完整 block 数据
7. 计算 processing_block_profile（完整二级信息）
8. 生成该 block 的默认策略
9. 在 block 内生成 tile plan
10. 对每个 tile 生成全局 read_window
11. 按 read_window 从 working DOM / VRT 中读取 tile
12. 生成 TileRunContext
13. 若 tile 与 block 默认判断差异不大，则直接继承
14. 若差异明显，则做有限覆盖
15. 调用主模型推理
16. 裁掉 padding 输出
17. 写回 valid_write_window
18. 在 block 内融合
19. block 结果及时落盘
20. 全图级拼接与融合
21. 记录高风险 / 失败 / 候选样本
```

---

## 13.15 【新增】本补充部分的最终结论

关于本文件中的 block 切分、二级信息提取与主模型推理前读取逻辑，最终应明确以下结论：

```text
1. block 默认是逻辑切分，不是默认物理切块；
2. 一级信息不受影响；
3. 二级信息可以完整提取，但前提是按 block_window 完整读取该 block 范围数据；
4. 三级 tile 读取与逻辑窗口方案天然兼容；
5. 主模型推理时，直接从 working DOM / VRT 按 tile window 读取即可；
6. 只要输入窗口、预处理、mask 与 padding 规则一致，逻辑读取与物理 block / tile 文件读取在分析与推理上可视为等价。
```
