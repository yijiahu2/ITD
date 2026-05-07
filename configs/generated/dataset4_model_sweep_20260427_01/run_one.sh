#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY="$1"
CFG_DIR="/home/xth/forest_agent_project/configs/generated/dataset4_model_sweep_20260427_01"
OUT_ROOT="/home/xth/forest_agent_project/outputs/dataset4_model_sweep_20260427_01"
REPO="/home/xth/forest_agent_project"
CFG="$CFG_DIR/${MODEL_KEY}.yaml"
LOG_DIR="$OUT_ROOT/logs"
DONE_DIR="$OUT_ROOT/status"
mkdir -p "$LOG_DIR" "$DONE_DIR"
if [[ ! -f "$CFG" ]]; then
  echo "[ERROR] missing config: $CFG" >&2
  exit 2
fi
if [[ -f "$DONE_DIR/${MODEL_KEY}.done" ]]; then
  echo "[SKIP] $MODEL_KEY already done"
  exit 0
fi
cd "$REPO"
source /home/xth/anaconda3/etc/profile.d/conda.sh
conda activate forest_agent
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "[START] $MODEL_KEY $(date -Is)"
set +e
python -u -m scripts.run_public_segmentation_model_finetune_pipeline --config "$CFG" 2>&1 | tee "$LOG_DIR/${MODEL_KEY}.log"
RC=${PIPESTATUS[0]}
set -e
echo "$RC" > "$DONE_DIR/${MODEL_KEY}.exitcode"
if [[ "$RC" -eq 0 ]]; then
  date -Is > "$DONE_DIR/${MODEL_KEY}.done"
  echo "[DONE] $MODEL_KEY $(date -Is)"
else
  date -Is > "$DONE_DIR/${MODEL_KEY}.failed"
  echo "[FAILED] $MODEL_KEY rc=$RC $(date -Is)" >&2
fi
exit "$RC"
