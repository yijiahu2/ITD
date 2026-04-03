# ITD_agent Data Processing Specification

`ITD_agent/data_processing/` 现在按“标准化输入 -> 先验提取 -> 请求驱动处理 -> 融合后处理”组织。

## Module Layout

- `contracts.py`
  数据处理模块统一输入输出契约
- `image_priors.py`
  高分辨率遥感影像先验提取与切块策略建议
- `dem_pipeline.py`
  DEM 对齐检查与 terrain quartet 摘要
- `survey_normalizer.py`
  样地调查表和行业矢量字段标准化摘要
- `knowledge_normalizer.py`
  领域知识统一分类
- `public_dataset_indexer.py`
  公开数据集索引
- `request_processor.py`
  数据处理请求生成与落盘
- `roi_extractor.py`
  ROI 差区域提取、几何裁剪、ROI 影像/矢量/DEM 裁剪执行
- `fusion_postprocess.py`
  多轮实例结果融合、去重、边界碎片抑制
- `processor.py`
  数据处理主汇总入口

## Output Layout

每次运行会在 `output_dir/data_processing/` 下生成：

- `input_profiles/`
- `raster_cache/`
- `terrain_cache/`
- `vector_cache/`
- `knowledge_cache/`
- `public_dataset_index/`
- `roi_cache/`
- `fusion_cache/`
- `requests/`
- `summaries/data_processing_summary.json`

若启用了临时运行目录，这些内容会先落在临时 `output_dir` 下，运行结束后仅将需要保留的摘要与请求同步到持久输出目录。

## High-Resolution Image Policy

影像处理策略采用两级判断：

1. 优先判定是否需要按小班/ROI 做几何裁剪
2. 裁剪后若仍然过大，则进入滑窗模式

ROI 差区域的裁剪执行也统一放在数据处理模块：

1. `评估分析` 输出问题区域
2. `规划调度` 生成 `roi_extraction` 计划
3. `data_processing/roi_extractor.py` 执行 ROI 影像、ROI 小班、ROI DEM 及地形产品裁剪
4. `分割模型` 仅消费这些 ROI 裁剪结果

配置示例：

```yaml
ITD_agent:
  data_processing:
    image_policy:
      max_direct_pixels: 30000000
      max_direct_area_ha: 25.0
```

## Current Standard Fields

调查表和行业矢量当前统一识别以下标准字段：

- `xiaoban_id`
- `tree_count`
- `crown_width`
- `closure`
- `density`
- `area_ha`

## Knowledge Normalization

领域知识统一规范到以下类型：

- `raster_prior`
- `tabular_prior`
- `rule_knowledge`
- `strategy_knowledge`
- `text_knowledge`

## Public Dataset Roles

公开数据集当前按微调用途索引为：

- `main_model`
- `child_model`

若未显式声明，默认同时对两类分割模型可见。
