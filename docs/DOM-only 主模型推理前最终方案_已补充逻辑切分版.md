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

**【新增】2.1 `block` 默认采用逻辑切分，不预先物理裁剪为独立影像文件**

这里需要进一步明确：

```text
block 默认是逻辑 block（logical block），不是预先落盘的 block.tif。
```

也就是说，`5632×5632` 在本方案中优先表示：

```text
局部分析范围；
局部策略生成范围；
断点续跑范围；
局部融合范围；
tile 生成范围。
```

而不是：

```text
必须先把原始 DOM 物理裁剪成大量 5632×5632 的 block.tif 文件。
```

推荐默认工程实现：

```text
原始 DOM 保持不动；
只生成 working DOM / working VRT；
系统根据全图尺寸生成 logical block plan；
每个 block 仅保存 block_window、block_geo_bounds、block_id、状态与策略信息；
只有在调试、失败复现、样本导出、微调候选沉淀或分布式调度时，才按需将 block 或 tile 物理落盘。
```

因此，本方案中的 block 应统一理解为：

```text
逻辑切分单元，而非默认的物理切图文件。
```

**【新增】2.2 逻辑切分不影响一级、二级、三级信息提取**

采用 logical block 后，不改变三级结构本身，只改变 block 数据的获取方式：

```text
一级：
仍然基于整张 DOM / working VRT 提取全局输入契约

二级：
仍然对每个 block 提取完整 block 级信息
只是从“先读取 block.tif”改成“按 block_window 直接从 working VRT 读取这一块影像”

三级：
仍然对每个 tile 构建 TileRunContext
只是 tile 数据也改为按 read_window 从 working VRT 直接读取
```

因此：

```text
逻辑切分 ≠ 不读取 block 内容
逻辑切分 = 不预先裁成 block.tif，但在需要分析或推理时，仍然完整读取该 block 对应范围的数据
```


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



## 5.4.1 【新增】一级只负责生成全局契约，不负责预先导出 block.tif

一级阶段建议明确采用如下原则：

```text
原始 DOM 不改；
生成 working DOM / VRT；
生成 valid mask；
生成全图 block plan 与 tile execution contract；
不在一级阶段预先把所有 block 裁成独立影像文件。
```

一级阶段生成的 block plan 建议只包含：

```text
block_id
block_window
block_geo_bounds
edge_block_flag
expected_tile_count
status
```

也就是说，一级阶段输出的是：

```text
逻辑 block 索引
```

而不是：

```text
一个个真实存在的 block_0001.tif、block_0002.tif 文件
```


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



## 6.1.1 【新增】二级完整信息提取的默认读取方式

在采用 logical block 的前提下，二级信息仍然要求**完整提取**，并且默认读取方式统一为：

```text
根据 block_window
直接从 working DOM / working VRT 中按窗口读取该 block 的完整影像数据
```

推荐理解为：

```text
物理切块方案：
原图 → 裁出 block.tif → 读取 block.tif → 提取二级信息

逻辑切分方案：
原图 / working VRT → 根据 block_window 直接读取对应像素范围 → 提取同样的二级信息
```

只要满足以下条件一致：

```text
读取范围一致；
数据源一致；
波段顺序一致；
归一化方式一致；
nodata / valid mask 处理一致；
边缘处理一致；
```

那么逻辑读取方式提取出的以下二级信息，可以与物理 block.tif 路线做到等价：

```text
valid_pixel_ratio
brightness_mean/std
shadow_ratio
overexposed_ratio
underexposed_ratio
laplacian_variance
tenengrad
blur_score
gradient_mean/std
texture_entropy
texture_contrast
texture_homogeneity
texture_complexity_score
heterogeneity_coarse_grid 相关统计
risk_tags
localized_risk_tags
quality_class
priority_score
expected_failure_modes
policy_template_name
diam_list / augment / iou_merge_thr 等默认策略
```

## 6.1.2 【新增】二级 block 数据读取与 profile 提取逻辑

对于每个 block，应先得到：

```text
block_window = [x, y, width, height]
```

然后按如下逻辑读取该 block 的完整影像内容：

```python
with rasterio.open(working_dom_path) as src:
    block_arr = src.read(
        indexes=[1, 2, 3],
        window=Window(x, y, width, height)
    )
```

如需 valid mask，同样按相同窗口读取：

```python
with rasterio.open(valid_mask_path) as msk:
    valid_mask = msk.read(
        1,
        window=Window(x, y, width, height)
    )
```

随后在 `block_arr + valid_mask` 上完整提取二级指标。

这里需要特别强调：

```text
二级信息提取阶段，
应基于该 block 的真实有效数据进行分析，
而不是为了凑满尺寸先 pad 再分析。
```

也就是说：

```text
二级 block profile 的输入优先是真实 block 数据；
padding 主要服务于三级 tile 模型输入，不应优先用于二级质量与纹理分析。
```

## 6.1.3 【新增】二级信息与物理 block.tif 的等价性说明

如果 physical block.tif 本来就是由同一原图、同一范围、同一处理规则裁出的，那么：

```text
直接从 working VRT 按 block_window 读取 block 数据
```

与：

```text
先生成 block.tif 再读取 block.tif
```

在理论上可以得到相同或等价的 block 像素矩阵。

因此，在实现正确时，使用 logical block 读取所提取的：

```text
影像质量特征
纹理特征
光谱统计特征
异质性特征
风险标签
默认策略
```

可以与物理切分 block.tif 路线保持一致。

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



## 7.1.1 【新增】三级主模型推理前的数据读取逻辑

三级阶段的 tile 是模型真实输入窗口，因此在 logical block 方案下，模型推理前的数据读取逻辑统一为：

```text
先进入某个 block；
在 block 内生成 tile plan；
把 tile 的局部坐标换算为全图坐标；
然后直接从 working DOM / working VRT 中按全局 read_window 读取该 tile；
将其送入主模型推理。
```

推荐理解为：

```text
物理 block.tif 方案：
原图 → 裁出 block.tif → 在 block.tif 内滑窗读取 tile → 推理

logical block 方案：
原图 / working VRT → 基于 block_window 生成 tile 的全局 read_window → 直接读取 tile → 推理
```

对于模型而言，真正重要的是：

```text
最终送入模型的 tile 像素数组是否一致
```

而不是：

```text
这个 tile 是否先被保存成独立的 tile.tif 文件
```

## 7.1.2 【新增】tile 全局读取窗口的换算方式

假设：

```text
block_window = [bx, by, bw, bh]
tile_local_window = [tx, ty, tile_px, tile_px]
```

则 tile 在整张 working VRT 中的全局读取窗口为：

```text
global_tx = bx + tx
global_ty = by + ty
read_window = [global_tx, global_ty, tile_px, tile_px]
```

随后按如下方式读取模型输入 tile：

```python
with rasterio.open(working_dom_path) as src:
    tile = src.read(
        indexes=[1, 2, 3],
        window=Window(global_tx, global_ty, tile_px, tile_px),
        boundless=True,
        fill_value=0
    )
```

必要时再转为模型常用格式：

```python
tile = np.transpose(tile, (1, 2, 0))
```

## 7.1.3 【新增】logical tile 读取与物理 tile.tif 推理输入的等价性

在以下条件一致时：

```text
tile 对应空间范围一致；
数据源一致；
波段顺序一致；
归一化与预处理一致；
边缘 padding 规则一致；
```

则：

```text
从 working VRT 按 read_window 直接读取的 tile
```

与：

```text
先物理裁剪得到该 tile.tif 再读取
```

对于主模型输入来说可以做到相同或等价。

特别说明：

### 非边缘 tile

对于完全位于影像内部的普通 tile，如果没有发生额外重采样、压缩量化或拉伸变化，则两种方式通常可以做到一致。

### 边缘 tile

对于边缘 tile，若 logical 读取采用：

```text
boundless=True + fill_value=0
```

那么只有在物理 tile.tif 路线也采用相同 padding 规则时，两种输入才能视为一致。

因此，三级阶段建议统一采用：

```text
直接基于 read_window 从 working VRT 读取 tile；
边缘不足部分按统一 padding 规则补齐；
推理完成后仅将 valid_write_window 对应结果写回原图坐标。
```

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



## 10.3 【新增】按需物理落盘原则

在本方案中，block 与 tile 默认采用逻辑切分与按需读取，不建议在常规推理前批量物理导出全部 block/tile 文件。

只有以下场景建议按需落盘：

```text
调试复现；
失败 block / failed tile 追踪；
高风险样本导出；
记忆候选样本沉淀；
微调候选样本沉淀；
多机 / 分布式调度需要显式文件输入；
```

默认原则：

```text
能通过 working VRT + window read 完成的读取与推理，
就不额外批量导出物理切块文件。
```

# 十一、最终执行流程

```text
1. 读取原始 DOM
2. 生成 dom_input_contract
3. 生成 working DOM / working VRT 与 valid mask
4. 生成 logical block plan（仅生成 block_window / bounds / id，不预先批量导出 block.tif）
5. 遍历每个 block
6. 根据 block_window 从 working VRT 中完整读取该 block 对应范围影像
7. 计算 processing_block_profile，完整提取二级信息
8. 生成该 block 的默认策略
9. 在 block 内生成 tile plan
10. 遍历每个 tile
11. 生成 TileRunContext
12. 将 tile 的局部坐标换算为全局 read_window
13. 直接从 working VRT 中按 read_window 读取模型输入 tile
14. 若 tile 与 block 默认判断差异不大，则直接继承
15. 若差异明显，则轻量覆盖部分参数
16. 调用主模型推理
17. 裁掉 padding 输出
18. 只将 valid_write_window 对应结果写回原图坐标
19. 在 block 内融合
20. 拼回全图
21. 记录高风险 / 失败 / 候选样本
22. 仅在需要时按需导出 block 或 tile 文件
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



**【新增】执行层面的统一理解**

本方案最终应统一理解为：

```text
block 默认是 logical block；
二级信息完整提取，但默认通过 block_window 从 working VRT 按需读取；
tile 才是模型真实输入；
主模型推理前默认通过 read_window 从 working VRT 直接读取 tile；
只有在少数必要场景下，才按需导出真实 block/tile 文件。
```

这样可以同时满足：

```text
保留一级、二级、三级的完整信息结构；
保留 block 级完整 profile 提取；
保留 tile 级真实模型输入与推理逻辑；
避免大范围 DOM 下批量物理切块带来的额外 IO、落盘时间与文件管理负担。
```







