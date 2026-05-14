#!/usr/bin/env bash
# Run a small grid of trials (different LR seeds) and save logs.
# Usage: ./run_trials.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VENV="$ROOT/.venv312"

if [ ! -d "$VENV" ]; then
  echo "Virtualenv $VENV not found. Create it first with Python 3.12."
  exit 1
fi

source "$VENV/bin/activate"
cd "$ROOT"

TRAIN_SET=0
EPOCHS=100
LRS=(0.01 0.005 0.001)
SEEDS=(0 1 2)

mkdir -p logs
for lr in "${LRS[@]}"; do
  for s in "${SEEDS[@]}"; do
    export TRAIN_SET
    export EPOCHS
    export LR=$lr
    export SEED=$s
    logfile="logs/central_train_lr${lr}_seed${s}.log"
    echo "Starting lr=$lr seed=$s -> $logfile"
    python models/cifar.py > "$logfile" 2>&1 || echo "Run failed: $logfile"
  done
done

echo "Trials finished. Logs in logs/"