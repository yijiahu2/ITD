#!/usr/bin/env bash
set -euo pipefail

CFG_DIR="/home/xth/forest_agent_project/configs/generated/dataset4_model_sweep_fair_20260427_02"
OUT_ROOT="/home/xth/forest_agent_project/outputs/dataset4_model_sweep_fair_20260427_02"
STATUS_DIR="$OUT_ROOT/status"
LOG_DIR="$OUT_ROOT/logs"
WATCH_LOG="$LOG_DIR/watch_maskdino_then_rerun_mask2former.log"
MODEL_WAIT="05_maskdino"
MODEL_RERUN="04_mask2former"
REPO="/home/xth/forest_agent_project"

mkdir -p "$STATUS_DIR" "$LOG_DIR"

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCH_LOG"
}

maskdino_running() {
  ps -eo cmd | grep -F "run_one.sh $MODEL_WAIT" | grep -v grep >/dev/null 2>&1 \
    || ps -eo cmd | grep -F "maskdino_train_entry" | grep -F "$OUT_ROOT/$MODEL_WAIT/" | grep -v grep >/dev/null 2>&1
}

log "watcher started"

while true; do
  if [[ -f "$STATUS_DIR/${MODEL_WAIT}.done" ]]; then
    log "$MODEL_WAIT completed; starting $MODEL_RERUN"
    break
  fi
  if [[ -f "$STATUS_DIR/${MODEL_WAIT}.failed" ]]; then
    log "$MODEL_WAIT failed; still starting $MODEL_RERUN as requested"
    break
  fi
  sleep 120
done

log "launching $MODEL_RERUN via run_one.sh"
cd "$REPO"
bash "$CFG_DIR/run_one.sh" "$MODEL_RERUN" 2>&1 | tee -a "$WATCH_LOG"
RC=${PIPESTATUS[0]}
log "$MODEL_RERUN finished with rc=$RC"
exit "$RC"
