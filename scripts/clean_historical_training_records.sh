#!/usr/bin/env bash
set -euo pipefail

rm -rf /home/xth/forest_agent_project/outputs/train_runs
rm -rf /home/xth/forest_agent_project/outputs/verify
rm -rf /home/xth/forest_agent_project/outputs/verify_local
rm -rf /home/xth/forest_agent_project/outputs/expert_dataset_splits/isprs_itd_clean_v1
rm -rf /home/xth/forest_agent_project/outputs/expert_dataset_splits/isprs_itd_clean_v2
rm -rf /home/xth/forest_agent_project/outputs/expert_dataset_splits/isprs_itd_clean_v3

find /home/xth/forest_agent_project/outputs/JX_ShanXia -maxdepth 4 -type d -name finetune -prune -exec rm -rf {} +
find /home/xth/forest_agent_project/outputs/JX_ShanXia -maxdepth 5 -type f \( -name 'public_segmentation_model_finetune_summary.json' -o -name 'public_segmentation_finetune_pipeline_summary.json' -o -name 'train_summary.json' \) -delete || true

rm -rf /mnt/f/forest_agent_project/outputs/experts
rm -rf /mnt/f/forest_agent_project/outputs/experts_full_clean

echo "[OK] historical training/finetune artifacts cleared"
