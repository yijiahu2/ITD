# ITD_agent Input Specification

`ITD_agent` 的输入层现在按 `Input Manifest + Validator + Preparer + Registry` 组织。

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

### 3. Survey Tables

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

### 4. Industry Vectors

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

### 5. Domain Knowledge

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

### 6. Public Datasets

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
- `required: true` 的输入缺失时标记为 `error`

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
