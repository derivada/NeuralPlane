#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/require_published.py
: "${DATA:?}" "${PY:?}" "${PRED_ROOT:?}" "${RESOLUTION:?}" "${DATASET:?}" "${SCENES:?}"
export CUDA_HOME="${CUDA_HOME:-$HOME/reconstr/.cuda/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export TCNN_CUDA_ARCHITECTURES=120
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
for scene in $SCENES; do
  "$PY" scripts/benchmark_scene.py --data "$DATA" --scene "$scene" --dataset "$DATASET" \
    --resolution "$RESOLUTION" --output "$PRED_ROOT/$scene"
done
