# output_layer 模块最终输出方案

# 一、最终只支持 3 种输入场景

本次 `output_layer` 不再泛泛支持所有图片，而是聚焦这三类：

```text
A. 无坐标 COCO 标注格式公开数据集
B. 带坐标、带人工标注的 DOM 影像
C. 带坐标、不带人工标注的 DOM 影像
```

对应关系如下：

| 输入类型         | 是否带坐标 | 是否有 GT | 主要目的                   |
| ---------------- | ---------: | --------: | -------------------------- |
| COCO 公共数据集  |         否 |        是 | 论文式模型精度评估         |
| DOM + 人工标注   |         是 |        是 | 遥感/GIS真实精度评估       |
| DOM + 无人工标注 |         是 |        否 | 真实应用成果提取与质量诊断 |

---

# 二、output_layer 最终职责边界

`output_layer` 最终只负责：

```
1. 生成最终树冠结果
2. 生成最终树点结果
3. 生成 semantic / instance mask
4. 生成必要可视化图
5. 生成 Markdown 评价或推理报告
```

不负责：

```
1. 自进化样本筛选
2. 微调池构建
3. 记忆沉淀
4. DEM / CHM 分析
5. 小班统计
6. 大量错误案例保存
7. 机器读取 summary_report.json
```

---

# 三、统一保留的最小默认输出

无论是哪种输入，最终都建议保留 4 类输出：

```text
output/
├── results/
├── masks/
├── visualization/
└── report/
```

其中：

| 目录             | 作用                                         |
| ---------------- | -------------------------------------------- |
| `results/`       | 树冠实例成果，包含 tree_crowns / tree_points |
| `masks/`         | semantic mask / instance mask                |
| `visualization/` | 人工检查图                                   |
| `report/`        | Markdown 格式结果分析报告                    |



# 四、场景 A：无坐标 COCO 标注格式公开数据集

## 1. 默认输出目录

```text
output_coco_dataset/
├── results/
│   └── coco_predictions.json
│
├── masks/
│   └── instance_masks/
│       ├── image_000001_instance_mask.png
│       ├── image_000002_instance_mask.png
│       └── ...
│
├── visualization/
│   ├── sample_overlays/
│   │   ├── sample_000001_overlay.png
│   │   ├── sample_000002_overlay.png
│   │   └── ...
│   └── selected_error_examples/
│       ├── false_positive_examples.png
│       ├── false_negative_examples.png
│       └── low_iou_examples.png
│
└── report/
    └── evaluation_report.md
```

---

## 2. 关于 error_cases 的最终处理
 最终应该改成 **选择性输出 selected_error_examples**。

建议规则：

```
默认只输出 Top-K 错误案例拼图；
不逐图保存所有错误图；
Top-K 默认 10 或 20；
可通过参数调整。
```

建议参数：

```
--save-error-examples
--max-error-examples 20
--error-example-types fp,fn,low_iou
```

默认可以只输出：

```
selected_error_examples/
├── false_positive_examples.png
├── false_negative_examples.png
└── low_iou_examples.png
```

这里每张 PNG 是一个拼图，不是一张图一个文件。



## 3. 各输出内容说明

### `results/coco_predictions.json`

这是 COCO 数据集最重要的标准预测结果，用于论文评估和模型对比。

应包含：

```text
image_id
category_id
bbox
score
segmentation
```

这里不需要再额外输出 `tree_crowns.gpkg`、`tree_points.gpkg`，因为无坐标 COCO 数据没有真实地理空间意义。

---

### `masks/instance_mask.png`

建议保留一张像素级实例 ID 图。

含义：

```text
背景 = 0
tree_1 = 1
tree_2 = 2
tree_3 = 3
...
```

如果是批量 COCO 测试集，可以按图片输出：

```text
masks/
├── image_000001_instance_mask.png
├── image_000002_instance_mask.png
└── ...
```

不建议默认输出每棵树一个 mask 文件，太碎、太乱。

---

### `visualization/`

COCO 数据集建议重点输出错误案例图，，而不是只输出普通叠加图。

应该包括：

| 文件                          | 作用                 |
| ----------------------------- | -------------------- |
| `sample_000001_overlay.png`   | 预测 mask 与原图叠加 |
| `false_positive_examples.png` | 误检案例             |
| `false_negative_examples.png` | 漏检案例             |
| `low_iou_examples.png`        | 边界差、IoU 低的案例 |

这对论文分析非常有用。

---

### `report/evaluation_report.md`

这是 COCO 场景最核心的人类可读报告。

报告中应该包含：

```text
1. 数据集基本信息
2. 模型与推理配置
3. COCO 指标结果表
4. 检测指标结果表
5. 掩码分割指标结果表
6. 错误类型统计
7. 典型失败案例
8. 总体结论
```

---

## 3. COCO 报告主表建议

Markdown 中建议放这张主表：

COCO 输入是公开数据集，报告主表应该评价 **整个测试集 / 验证集总体表现**，不是单张图。

### COCO 报告主表

| Dataset   | Images | GT Instances | Pred Instances | Precision | Recall | F1   | Bbox AP50 | Bbox AP75 | Bbox AP | Mask AP50 | Mask AP75 | Mask AP | mIoU | FP   | FN   |
| --------- | ------ | ------------ | -------------- | --------- | ------ | ---- | --------- | --------- | ------- | --------- | --------- | ------- | ---- | ---- | ---- |
| COCO_Test | 500    | 1200         | 1168           | 0.89      | 0.86   | 0.87 | 0.84      | 0.66      | 0.58    | 0.81      | 0.62      | 0.54    | 0.72 | 85   | 117  |

不建议在主报告中逐图列出所有图片指标。
 如果确实需要逐图指标，应作为可选导出：

```
--export-per-image-metrics
```

可选生成：

```
report/per_image_metrics.csv
```

但默认不生成。

# 五、场景 B：带坐标、带人工标注的 DOM 影像

这是最完整、最重要的真实遥感/GIS评估场景。

---

## 1. 默认输出目录

```text
output_dom_with_gt/
├── results/
│   ├── tree_crowns.shp
│   └── tree_points.shp
│
├── masks/
│   ├── semantic_mask.tif
│   └── instance_mask.tif
│
├── visualization/
│   ├── pred_overlay.png
│   ├── gt_pred_overlay.png
│   ├── instance_boundaries.png
│   └── evaluation_map.png
│
└── report/
    └── evaluation_report.md
```

---

## 2. `results/tree_crowns.shp`

每条记录代表一个预测树冠面。

建议字段：

| 字段名       | 中文名          |
| ------------ | --------------- |
| `tree_id`    | 树木编号        |
| `score`      | 置信度          |
| `area_m2`    | 树冠面积        |
| `perim_m`    | 树冠周长        |
| `eq_width_m` | 等效冠幅        |
| `center_x`   | 中心点 X        |
| `center_y`   | 中心点 Y        |
| `gt_id`      | 匹配 GT 编号    |
| `iou_gt`     | 与 GT 的 IoU    |
| `eval_type`  | 评价类型        |
| `src_tile`   | 来源 tile/block |

注意 Shapefile 字段名长度有限，所以字段名要短。
 例如不要用 `equivalent_crown_width_m`，建议用 `eq_width_m`。

`eval_type` 可以包括：

```
TP
FP
LOW_IOU
OVER_SEG
UNDER_SEG
```

是否把 `OVER_SEG / UNDER_SEG` 写入 shp，可以根据 evaluation_analysis 是否已经能稳定判断来决定。最低要求至少有：

```
TP / FP / LOW_IOU
```

------

## 3. `results/tree_points.shp`

每条记录代表一棵预测树冠中心点。

建议字段：

| 字段名       | 中文名          |
| ------------ | --------------- |
| `tree_id`    | 树木编号        |
| `score`      | 置信度          |
| `area_m2`    | 树冠面积        |
| `eq_width_m` | 等效冠幅        |
| `gt_id`      | 匹配 GT 编号    |
| `iou_gt`     | 与 GT 的 IoU    |
| `eval_type`  | 评价类型        |
| `src_tile`   | 来源 tile/block |

------

## 4. `masks/`

```
masks/
├── semantic_mask.tif
└── instance_mask.tif
```

| 文件                | 说明                              |
| ------------------- | --------------------------------- |
| `semantic_mask.tif` | 树冠/背景二值语义 mask，带 CRS    |
| `instance_mask.tif` | 每棵树唯一 ID 的实例 mask，带 CRS |

------

## 5. `visualization/`

DOM + GT 场景下，最终可视化应该更偏“结果检查”，而不是“训练样本挖掘”。

建议保留：

| 文件                      | 作用                              |
| ------------------------- | --------------------------------- |
| `pred_overlay.png`        | 预测树冠叠加原图                  |
| `gt_pred_overlay.png`     | GT 树冠与预测树冠叠加对比         |
| `instance_boundaries.png` | 预测实例边界图                    |
| `evaluation_map.png`      | TP / FP / FN / LOW_IOU 空间评价图 |

------

## 6. DOM + GT 的 evaluation_report.md

报告应该只关注 GT 评估和空间尺度树冠指标，不涉及 DEM、CHM、小班。

### 报告结构

```
# DOM Tree Crown Extraction Evaluation Report(中文版)

## 1. Input Overview
## 2. Model and Inference Setting
## 3. Overall Evaluation Results
## 4. Detection Accuracy
## 5. Instance Segmentation Accuracy
## 6. Crown Geometry Accuracy
## 7. Error Type Summary
## 8. Visualization Summary
## 9. Conclusion
```

### DOM + GT 报告主表

| Input   | Area(ha) | GT Count | Pred Count | Count Error(%) | Precision | Recall | F1   | Mask AP50 | Mask AP75 | Mask AP | mIoU | Mean Area Error(%) | Mean Crown Width Error(%) | FP   | FN   | Low IoU Count |
| ------- | -------- | -------- | ---------- | -------------- | --------- | ------ | ---- | --------- | --------- | ------- | ---- | ------------------ | ------------------------- | ---- | ---- | ------------- |
| DOM_001 | 1.00     | 812      | 790        | 2.7            | 0.89      | 0.86   | 0.87 | 0.82      | 0.61      | 0.54    | 0.71 | 4.8                | 3.2                       | 85   | 117  | 42            |



# 六、场景 C：带坐标、不带人工标注的 DOM 影像

这是最终真实应用报告。
 这里没有 GT，所以不能出现 AP、Precision、Recall、F1、IoU。

## 1. 最终默认输出目录

```
output_dom_without_gt/
├── results/
│   ├── tree_crowns.shp
│   └── tree_points.shp
│
├── masks/
│   ├── semantic_mask.tif
│   └── instance_mask.tif
│
├── visualization/
│   ├── pred_overlay.png
│   ├── instance_boundaries.png
│   ├── confidence_map.png
│   └── risk_map.png
│
└── report/
    └── inference_report.md
```

我这里也把无 GT DOM 的 `results/` 同步改成了 `.shp`，保持带坐标 DOM 输出格式一致。

------

## 2. `results/tree_crowns.shp`

建议字段：

| 字段名       | 中文名          |
| ------------ | --------------- |
| `tree_id`    | 树木编号        |
| `score`      | 置信度          |
| `area_m2`    | 树冠面积        |
| `perim_m`    | 树冠周长        |
| `eq_width_m` | 等效冠幅        |
| `center_x`   | 中心点 X        |
| `center_y`   | 中心点 Y        |
| `quality`    | 质量标记        |
| `risk_type`  | 风险类型        |
| `src_tile`   | 来源 tile/block |

`risk_type` 可包括：

```
normal
small_fragment
width_outlier
duplicate_overlap
edge_artifact
semantic_gap
fragmentation_risk
merge_blob_risk
```

------

## 3. `results/tree_points.shp`

建议字段：

| 字段名       | 中文名          |
| ------------ | --------------- |
| `tree_id`    | 树木编号        |
| `score`      | 置信度          |
| `area_m2`    | 树冠面积        |
| `eq_width_m` | 等效冠幅        |
| `quality`    | 质量标记        |
| `risk_type`  | 风险类型        |
| `src_tile`   | 来源 tile/block |

------

## 4. `masks/`

```
masks/
├── semantic_mask.tif
└── instance_mask.tif
```

仍然必须保留。
 因为无 GT 场景下，`semantic_mask` 和 `instance_mask` 的一致性本身就是质量诊断的重要依据。

------

## 5. `visualization/`

```
visualization/
├── pred_overlay.png
├── instance_boundaries.png
├── confidence_map.png
└── risk_map.png
```

| 文件                      | 作用             |
| ------------------------- | ---------------- |
| `pred_overlay.png`        | 预测树冠叠加原图 |
| `instance_boundaries.png` | 实例边界检查     |
| `confidence_map.png`      | 置信度空间分布   |
| `risk_map.png`            | 无 GT 风险区域图 |

------

## 6. DOM 无 GT 的 inference_report.md

### 报告结构

```
# DOM Tree Crown Extraction Inference Report

## 1. Input Overview
## 2. Model and Inference Setting
## 3. Prediction Result Summary
## 4. No-GT Quality Diagnosis
## 5. Visualization Summary
## 6. Conclusion
```

------

## 7. DOM 无 GT 报告主表

根据你给出的指标，结合本轮“去掉 CHM/DEM”要求，最终建议主表如下：

| pred_instance_count | pred_cover_ratio | mean_area_m2 | mean_equivalent_crown_width_m | small_fragment_ratio | width_outlier_ratio | duplicate_overlap_ratio | edge_artifact_score | semantic_instance_consistency | semantic_coverage_gap | fragmentation_score | merge_blob_score | online_risk_score | quality_score |
| ------------------- | ---------------- | ------------ | ----------------------------- | -------------------- | ------------------- | ----------------------- | ------------------- | ----------------------------- | --------------------- | ------------------- | ---------------- | ----------------- | ------------- |
| 790                 | 0.652            | 8.25         | 3.31                          | 0.041                | 0.028               | 0.017                   | 0.065               | 0.91                          | 0.083                 | 0.12                | 0.18             | 0.24              | 0.76          |



------

## 8. DOM 无 GT 指标最终清单

| 指标名                          | 中文名           | 是否保留 |
| ------------------------------- | ---------------- | -------- |
| `pred_instance_count`           | 预测树木数量     | 保留     |
| `pred_cover_ratio`              | 预测树冠覆盖率   | 保留     |
| `mean_area_m2`                  | 平均树冠面积     | 保留     |
| `mean_equivalent_crown_width_m` | 平均等效冠幅     | 保留     |
| `small_fragment_ratio`          | 小碎片比例       | 保留     |
| `width_outlier_ratio`           | 冠幅异常比例     | 保留     |
| `duplicate_overlap_ratio`       | 实例重叠比例     | 保留     |
| `edge_artifact_score`           | 边缘伪影风险     | 保留     |
| `semantic_instance_consistency` | 语义-实例一致性  | 保留     |
| `semantic_coverage_gap`         | 语义覆盖缺口     | 保留     |
| `fragmentation_score`           | 碎片化风险分数   | 保留     |
| `merge_blob_score`              | 合并斑块风险分数 | 保留     |
| `online_risk_score`             | 综合风险分数     | 保留     |
| `quality_score`                 | 综合质量分数     | 保留     |

# 七、最终 output_layer 输出内容总表

## 1. 三类输入的默认输出对比

| 输出内容                   | COCO 有 GT | DOM 有 GT | DOM 无 GT |
| -------------------------- | ---------- | --------- | --------- |
| `coco_predictions.json`    | 必须       | 不需要    | 不需要    |
| `tree_crowns.shp`          | 不需要     | 必须      | 必须      |
| `tree_points.shp`          | 不需要     | 必须      | 必须      |
| `semantic_mask.tif`        | 不需要     | 必须      | 必须      |
| `instance_mask.png`        | 必须       | 不需要    | 不需要    |
| `instance_mask.tif`        | 不需要     | 必须      | 必须      |
| `pred_overlay.png`         | 可选/抽样  | 必须      | 必须      |
| `gt_pred_overlay.png`      | 不需要     | 必须      | 不需要    |
| `instance_boundaries.png`  | 可选/抽样  | 必须      | 必须      |
| `evaluation_map.png`       | 不需要     | 必须      | 不需要    |
| `confidence_map.png`       | 不需要     | 可选      | 必须      |
| `risk_map.png`             | 不需要     | 不需要    | 必须      |
| `selected_error_examples/` | 可选 Top-K | 不需要    | 不需要    |
| `evaluation_report.md`     | 必须       | 必须      | 不需要    |
| `inference_report.md`      | 不需要     | 不需要    | 必须      |

# 八、最终推荐的 output_layer 目录结构

## A. COCO 有 GT

```
output_coco_dataset/
├── results/
│   └── coco_predictions.json
│
├── masks/
│   └── instance_masks/
│       ├── image_000001_instance_mask.png
│       ├── image_000002_instance_mask.png
│       └── ...
│
├── visualization/
│   ├── sample_overlays/
│   │   ├── sample_000001_overlay.png
│   │   ├── sample_000002_overlay.png
│   │   └── ...
│   └── selected_error_examples/
│       ├── false_positive_examples.png
│       ├── false_negative_examples.png
│       └── low_iou_examples.png
│
└── report/
    └── evaluation_report.md
```

其中：

```
sample_overlays/ 和 selected_error_examples/ 都应支持 Top-K 抽样输出。
```

------

## B. DOM 有 GT

```
output_dom_with_gt/
├── results/
│   ├── tree_crowns.shp
│   └── tree_points.shp
│
├── masks/
│   ├── semantic_mask.tif
│   └── instance_mask.tif
│
├── visualization/
│   ├── pred_overlay.png
│   ├── gt_pred_overlay.png
│   ├── instance_boundaries.png
│   └── evaluation_map.png
│
└── report/
    └── evaluation_report.md
```

------

## C. DOM 无 GT

```
output_dom_without_gt/
├── results/
│   ├── tree_crowns.shp
│   └── tree_points.shp
│
├── masks/
│   ├── semantic_mask.tif
│   └── instance_mask.tif
│
├── visualization/
│   ├── pred_overlay.png
│   ├── instance_boundaries.png
│   ├── confidence_map.png
│   └── risk_map.png
│
└── report/
    └── inference_report.md
```



# 九、最终可选导出参数

默认输出保持简洁。额外内容用参数控制：

```
--export-geojson
--export-gpkg
--export-csv
--export-xlsx
--export-coco
--save-error-examples
--max-error-examples 20
--save-sample-overlays
--max-sample-overlays 20
--export-per-image-metrics
--debug
```

其中：

| 参数                         | 用途                                   |
| ---------------------------- | -------------------------------------- |
| `--export-gpkg`              | 额外导出 GPKG                          |
| `--export-geojson`           | 额外导出 GeoJSON                       |
| `--export-csv`               | 额外导出属性表                         |
| `--export-coco`              | DOM 结果额外导出 COCO-style prediction |
| `--save-error-examples`      | COCO 场景保存 Top-K 错误案例           |
| `--export-per-image-metrics` | COCO 场景输出逐图指标                  |
| `--debug`                    | 输出调试级中间结果                     |