#!/usr/bin/env bash
# Run centralized CIFAR training using the Python 3.12 venv.
# Usage: ./run_centralized.sh [TRAIN_SET] [EPOCHS] [LR]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VENV="$ROOT/.venv312"
TRAIN_SET=${1:-0}
EPOCHS=${2:-100}
LR=${3:-0.01}

if [ ! -d "$VENV" ]; then
  echo "Virtualenv $VENV not found. Activate your env or create .venv312 with Python 3.12."
  exit 1
fi

source "$VENV/bin/activate"
cd "$ROOT"
export TRAIN_SET
export EPOCHS
export LR

echo "Running centralized CIFAR (TRAIN_SET=$TRAIN_SET, EPOCHS=$EPOCHS, LR=$LR)"
python models/cifar.py
