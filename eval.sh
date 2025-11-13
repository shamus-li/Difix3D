#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-../gs7/dataset}
RESULT_ROOT=${2:-results}
SCENES=(action-figure ball chicken dog espresso optics salt-pepper shelf)
LOG_DIR=slurm_logs
mkdir -p "$LOG_DIR"

PY_PATH="$PWD:$PWD/examples/gsplat:$PWD/../gsplat:$PWD/../gsplat/examples"

for idx in "${!SCENES[@]}"; do
  scene=${SCENES[$idx]}
  echo "[eval.sh] launching $scene (index $idx)"
  (
    EVAL_SCENE_INDEX=$idx \
    FORCE_ALIGN_REGEN=${FORCE_ALIGN_REGEN:-1} \
    USE_COVISIBLE_EXTERNAL=${USE_COVISIBLE_EXTERNAL:-1} \
    PYTHONPATH="$PY_PATH:${PYTHONPATH:-}" \
    bash examples/eval_only.slurm "$DATA_ROOT" "$RESULT_ROOT"
  ) >"$LOG_DIR/eval_${scene}.log" 2>&1 &
done

wait
echo "[eval.sh] all evaluations complete. Logs: $LOG_DIR/eval_<scene>.log"
