#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=/home/xth/forest_agent_project
CONDA_SH=/home/xth/anaconda3/etc/profile.d/conda.sh
CONFIG_DIR=${1:-$REPO_ROOT/outputs/expert_training_suite/configs_meta}
LOG_DIR=$REPO_ROOT/outputs/expert_training_suite/logs
CHECKPOINT_DIR=/home/xth/mmdetection331/checkpoints

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

MASK2FORMER_CKPT=$CHECKPOINT_DIR/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth
MASK_SCORING_CKPT=$CHECKPOINT_DIR/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth

if [[ ! -f "$MASK2FORMER_CKPT" ]]; then
  wget -c -O "$MASK2FORMER_CKPT" "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
fi

if [[ ! -f "$MASK_SCORING_CKPT" ]]; then
  wget -c -O "$MASK_SCORING_CKPT" "https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"
fi

nvidia-smi -L

source "$CONDA_SH"
conda activate forest_agent
export PYTHONNOUSERSITE=1
export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}

python - <<'PY'
import torch
print("forest_agent torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY

run_one() {
  local config_path="$1"
  local stem
  stem=$(basename "$config_path" .yaml)
  local log_path="$LOG_DIR/${stem}.log"
  local output_dir
  output_dir=$(python - <<PY
from ITD_agent.segmentation.finetuning.io_utils import load_yaml
cfg = load_yaml("$config_path")
print(cfg["output_dir"])
PY
)
  local train_summary="$output_dir/segmentation_training/train_summary.json"
  local test_summary="$output_dir/segmentation_training/evaluation/test_summary.json"
  if [[ -f "$train_summary" && -f "$test_summary" ]]; then
    echo "[SKIP completed] $config_path"
    return 0
  fi
  echo "[RUN] $config_path"
  {
    echo "[RUN] $config_path"
    PYTHONUNBUFFERED=1 python -u -m scripts.run_public_segmentation_model_finetune_pipeline --config "$config_path"
  } 2>&1 | tee "$log_path"
}

failures=()
for config_path in \
  "$CONFIG_DIR/dense_adhesion_htc_20260405_clean_gpu_full.yaml" \
  "$CONFIG_DIR/shadow_topography_mask2former_20260405_clean_gpu_full.yaml" \
  "$CONFIG_DIR/large_crown_cascade_20260405_clean_gpu_full.yaml" \
  "$CONFIG_DIR/boundary_mask_scoring_20260405_clean_gpu_full.yaml" \
  "$CONFIG_DIR/generalist_maskdino_20260405_clean_gpu_full.yaml"
do
  if ! run_one "$config_path"; then
    echo "[FAIL] $config_path"
    failures+=("$config_path")
  fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "[FAIL] full expert suite finished with ${#failures[@]} failed runs"
  printf ' - %s\n' "${failures[@]}"
  exit 1
fi

echo "[OK] full expert suite finished"
