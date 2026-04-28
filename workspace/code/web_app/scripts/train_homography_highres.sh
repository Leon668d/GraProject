#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TASK="${TASK:-homography}"
export RESOLUTION_SUBSET="${RESOLUTION_SUBSET:-High-Resolution}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-0.0003}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
export INPUT_SIZE="${INPUT_SIZE:-512}"
export MAX_SHIFT="${MAX_SHIFT:-40}"
export MAX_CORNER_SHIFT="${MAX_CORNER_SHIFT:-5}"
export MAX_ROTATION_DEG="${MAX_ROTATION_DEG:-3}"
export MAX_SCALE_JITTER="${MAX_SCALE_JITTER:-0.03}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export CACHE_IMAGES="${CACHE_IMAGES:-1}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-models/multiscale_homography_stage3_full_best.pth}"

bash "$SCRIPT_DIR/train_remote.sh" "$@"
