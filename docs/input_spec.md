# ITD_agent Input Specification

`ITD_agent` 的输入层现在按 `Input Manifest + Validator + Preparer + Registry` 组织。

## Mainline Profile

项目只有一套统一智能体 pipeline，主线 A/B 由 `runtime.mainline_profile` 控制输入能力，不拆分流程。

```yaml
runtime:
  mainline_profile: A_DOM_ONLY
```

支持的标准 profile：

- `A_DOM_ONLY`
  - 在线观测输入：`DOM`
  - DEM / CHM / DSM：关闭
  - 公开数据集、自制 COCO 数据集、经验记忆、微调池：启用
  - 小班、调查表、领域知识：默认关闭，不作为可用决策证据
  - 用途：公平对比 DOM-only SOTA，验证智能体框架价值
- `B_DOM_DEM_CHM_KNOWLEDGE`
  - 在线观测输入：`DOM + DEM + CHM`
  - 公开数据集、自制 COCO 数据集、经验记忆、微调池：启用
  - 外部知识层：默认关闭，保留小班、调查表、领域知识接口
  - 用途：在 A 的同一套流程基础上提升分割精度，并输出树高、冠高、结构信息

如果旧配置没有显式声明 `mainline_profile`，输入层会按实际出现的 DEM/CHM 或外部知识自动推断为 B；公开数据集和 COCO 数据集不会触发 B profile。新实验建议显式声明，避免 benchmark 口径不清。

## Directory Convention

- `raw_inputs/`
  原始数据外部保存，不要求复制进项目。
- `output_dir/input_registry/`
  保存本次任务的 `input_manifest.json`、`input_validation_report.json`、`prepared_input_index.json`。
- `output_dir/prepared_inputs/`
  保存规范化后输入的目标落位和中间产物。

## Supported Modalities

### 1. High-Resolution Remote Sensing

```yaml
inputs:
  remote_sensing:
    images:
      - id: dom177_rgb
        path: /abs/path/dom177.tif
        sensor: aerial_rgb
        resolution_m: 0.1
        crs: EPSG:4547
        bands: [R, G, B]
        required: true
```

### 2. DEM

```yaml
inputs:
  terrain:
    dem:
      - id: shanxia_dem
        path: /abs/path/dem.tif
        resolution_m: 1.0
        crs: EPSG:4547
        vertical_unit: m
```

### 3. CHM

```yaml
inputs:
  canopy:
    chm:
      - id: dom177_chm
        path: /abs/path/dom177_chm.tif
        resolution_m: 0.1
        crs: EPSG:4547
        vertical_unit: m
```

### 4. DSM

```yaml
inputs:
  surface:
    dsm:
      - id: dom177_dsm
        path: /abs/path/dom177_dsm.tif
        resolution_m: 0.1
        crs: EPSG:4547
        vertical_unit: m
```

### 5. Survey Tables

```yaml
inputs:
  survey_data:
    tables:
      - id: plot_inventory_2024
        path: /abs/path/plots.xlsx
        sheet_name: Sheet1
        key_fields: [plot_id]
        field_mapping:
          tree_count: tree_num
          crown_width: crown_m
          closure: canopy
```

### 6. Industry Vectors

```yaml
inputs:
  industry_vectors:
    vectors:
      - id: xiaoban_inventory
        path: /abs/path/xiaoban.shp
        geometry_type: polygon
        crs: EPSG:4547
        key_fields: [XBH]
        field_mapping:
          xiaoban_id: XBH
          tree_count: LMSL
          crown_width: PJGF
          closure: YBD
          area_ha: MJ_hm2
```

### 7. Domain Knowledge

```yaml
inputs:
  domain_knowledge:
    items:
      - id: forestry_rules
        type: text
        path: /abs/path/knowledge.md
      - id: species_summary
        type: table
        path: /abs/path/species.xlsx
```

### 8. Public Datasets

```yaml
inputs:
  public_datasets:
    datasets:
      - id: coco_public_001
        format: coco
        image_root: /abs/path/images
        annotation_path: /abs/path/annotations.json
      - id: parquet_public_001
        format: parquet
        path: /abs/path/dataset.parquet
        schema_mapping:
          image_path: image
          mask_path: mask
```

## Validation Rules

- 路径存在性检查
- 常用格式后缀检查
- 表格和矢量字段存在性检查
- COCO 顶层结构检查
- CHM / DSM 栅格存在性和格式检查
- `required: true` 的输入缺失时标记为 `error`

## Recommended Core Inputs

当前推荐的核心在线输入按 profile 区分：

- `A_DOM_ONLY`: `DOM`
- `B_DOM_DEM_CHM_KNOWLEDGE`: `DOM + DEM + CHM`

`PublicDatasets` 与自制 COCO 数据集属于 A/B 共享训练、验证、推理数据能力，不属于 A/B 差异项。

`IndustryVectors / SurveyTables / DomainKnowledge` 属于可选外部知识层。A profile 默认关闭且不作为可用决策证据；B profile 保留接口但默认关闭，后续显式开启时可用于动态系统提示词、训练决策、路由决策、样本筛选和后处理约束；不作为默认模型 tensor 输入。

`IndustryVectors / SurveyTables` 当提供时仍沿用当前 `field_mapping` 形式接入，但不再作为默认主流程必需项。

## Output Profile

- A 输出标准树冠实例、树木点、分割评估指标、ROI/调度/记忆记录。
- B 输出 A 的全部字段，并在 CHM 可用时增加实例级高度结构属性：`tree_height_p95`、`tree_height_max`、`crown_height_mean`、`crown_height_std`、`height_gradient`、`structure_tag`。

## Runtime Output

每次运行会自动在 `output_dir/input_registry/` 下生成：

- `input_manifest.json`
- `input_validation_report.json`
- `prepared_input_index.json`
- `registry_index.json`

同时会在 `output_dir/final_outputs/` 下复制：

- `input_manifest.json`
- `input_validation_report.json`
- `prepared_input_index.json`
