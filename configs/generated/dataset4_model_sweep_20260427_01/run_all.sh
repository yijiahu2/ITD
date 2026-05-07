#!/usr/bin/env bash
set -u -o pipefail
CFG_DIR="/home/xth/forest_agent_project/configs/generated/dataset4_model_sweep_20260427_01"
OUT_ROOT="/home/xth/forest_agent_project/outputs/dataset4_model_sweep_20260427_01"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/status"
MODELS=(
  02_cascade_mask_rcnn
  03_mask_scoring_rcnn
  04_mask2former
  05_maskdino
)
OVERALL_RC=0
for MODEL in "${MODELS[@]}"; do
  echo "========== $MODEL =========="
  bash "$CFG_DIR/run_one.sh" "$MODEL"
  RC=$?
  if [[ "$RC" -ne 0 ]]; then
    OVERALL_RC=1
    echo "[WARN] $MODEL failed, continuing next model" >&2
  fi
done
exit "$OVERALL_RC"
