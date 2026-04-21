# forest_agent_project Codemap

Last synced: 2026-04-13

This codemap reflects the current repository layout. It is the first document to update when code structure changes.

## Runtime Entry Points

- `python -m ITD_agent.orchestration.orchestrator --config <config>`
  Main ITD agent runtime. It loads and normalizes config, runs the single-scene orchestration flow, or dispatches grouped inference when enabled.
- `python -m ITD_agent.orchestration.grouped_inference --config <config>`
  Grouped inference flow for per-group local runtime configs and result merging.
- `python -m scripts.run_ITD_agent_experiment --config <config>`
  Thin script wrapper around the main runtime.
- `python -m scripts.run_grouped_experiment --config <config>`
  Thin script wrapper around grouped inference.

## Core Packages

- `input_layer/`
  Builds the `InputManifest`, validates input paths and schemas, prepares an input workspace, and writes `input_registry/` metadata through `registry.py`.
- `ITD_agent/config_adapter.py`
  Loads YAML configs and normalizes modern `inputs/runtime/outputs/ITD_agent` blocks into runtime keys consumed by orchestration.
- `ITD_agent/orchestration/`
  Owns runtime flow control, grouped inference, runtime path preparation, output retention, summary generation, and finalization.
- `ITD_agent/data_processing/`
  Owns input profiling, image/terrain/inventory/knowledge/public-dataset summaries, processing request records, ROI data preparation, and instance post-processing.
- `ITD_agent/evaluation_analysis/`
  Owns input, main-model, ROI, child-model, finetune-effect, reference-quality, and final assessment logic.
- `ITD_agent/llm_gateway/`
  Owns LLM gateway config, prompt builders, structured JSON calls, ROI decisions, planning advice, and run-retrospective input compaction.
- `ITD_agent/planning/`
  Contains legacy-style local refinement agents under `planning/agent/` and the scheduler under `planning/scheduler/`.
- `ITD_agent/planning/scheduler/`
  Converts templates, evaluation context, memory, finetune-pool context, parameter search, and expert taxonomy into structured runtime plans.
- `ITD_agent/segmentation/`
  Owns segmentation execution contracts, execution facade, model registry, registry adapters/runners, and finetune/training entry points.
- `ITD_agent/memory_store/`
  Stores compacted execution traces, success strategies, failure patterns, and run retrospectives.
- `ITD_agent/finetune_pool/`
  Stores failed ROI samples, replay samples, public dataset candidates, training trigger snapshots, clusters, and finetune dataset bundle export logic.
- `output_layer/`
  Publishes final tree-crown deliverables, tree points, visualizations, and final reports.

## Data Processing Subpackages

- `imagery/`: image profiles, quality estimates, texture metrics, and tile plans.
- `terrain/`: DEM alignment, terrain products, and terrain constraints.
- `inventory/`: survey table and industry vector normalization, crown metrics, and spatial context.
- `knowledge/`: domain knowledge normalization.
- `public_data/`: public dataset indexing and expert-family/domain tag inference.
- `roi/`: ROI extraction and ROI refinement input preparation.

## Config Surface

- `configs/examples/`: runnable example entry configs.
- `configs/templates/runtime/`: runtime templates for baseline and segmentation candidate/cascade/registry flows.
- `configs/templates/finetune/`: finetune templates for data-processing and segmentation-model training flows.
- `configs/templates/benchmark/`: benchmark and validation templates.
- `configs/expert_taxonomy/`: expert-family taxonomy consumed by `planning/scheduler/expert_taxonomy.py`.
- `configs/mmdet_custom/`: project-side MMDetection config fragments.
- `configs/generated/`: historical or manually retained config snapshots, not the scheduler runtime output target.

The current normalized config path accepts these top-level blocks:

- `runtime`: run name, environment, work directory, inventory field names.
- `inputs`: remote sensing, terrain, survey/inventory vectors, domain knowledge, public datasets.
- `ITD_agent`: planning, LLM gateway, data processing, and segmentation model blocks.
- `segmentation`: direct segmentation parameter overrides.
- `evaluation`: evaluation thresholds and final-report settings.
- `outputs`: persistent output root, cleanup policy, and temp runtime settings.

## Scripts

- `scripts/run_ITD_agent_experiment.py`: main runtime wrapper.
- `scripts/run_grouped_experiment.py`: grouped inference wrapper.
- `scripts/evaluate_reference_quality.py`: reference-quality metrics for tree crowns against inventory or benchmark data.
- `scripts/evaluate_roi_refinement_result.py`: ROI refinement result comparison helper.
- `scripts/list_segmentation_models.py`: dumps registered segmentation algorithms.
- `scripts/run_finetune_pipeline.py`: data-processing finetune pipeline.
- `scripts/run_public_finetune_pipeline.py`: public data-processing finetune wrapper.
- `scripts/run_public_segmentation_model_finetune_pipeline.py`: public segmentation-model finetune pipeline.
- `scripts/benchmark_coco_instance_dataset.py`: COCO instance benchmark utility.
- `scripts/prepare_isprs_itd_expert_splits.py`: expert split preparation for ISPRS ITD data.
- `scripts/generate_isprs_itd_expert_training_configs.py`: expert training config generation.
- `scripts/run_isprs_itd_expert_full_suite.sh`: shell suite runner for expert training configs.
- `scripts/compact_memory_store.py`: compacts memory store records and rebuilds indexes.
- `scripts/tile_dom_by_meters.py`: DOM tiling utility.
- `scripts/clean_historical_training_records.sh`: cleanup helper for historical training records.

## Tools

- `tools/runtime_cache_client.py`: starts and talks to the runtime cache worker.
- `tools/runtime_cache_worker.py`: worker process that executes cached semantic-prior and segmentation tasks.
- `tools/cached_stage_runners.py`: module-level cached runtime stage implementation.
- `tools/process_runner.py`: subprocess helper.
- `tools/stretch_tiles.py`: raster tile stretch utility.

## Output Surface

- Runtime outputs are rooted at `output_dir/`.
- Persistent outputs are rooted at `persistent_output_dir` when temp runtime is enabled.
- Final deliverables are published under `final_outputs/`.
- Minimal retention may remove stage directories while keeping compact summary and final deliverables.
