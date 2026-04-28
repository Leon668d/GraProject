#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" && "$PYTHON_BIN" == "python3" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

DATASET_ROOT="${DATASET_ROOT:-}"
if [[ -z "$DATASET_ROOT" ]]; then
  echo "Set DATASET_ROOT to the dataset root on the Linux server." >&2
  exit 1
fi

TASK="${TASK:-homography}"
RESOLUTION_SUBSET="${RESOLUTION_SUBSET:-High-Resolution}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_RUN_NAME="${TASK}_${RESOLUTION_SUBSET// /_}_${RUN_STAMP}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/training_runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
INPUT_SIZE="${INPUT_SIZE:-512}"
MAX_SHIFT="${MAX_SHIFT:-40}"
MAX_CORNER_SHIFT="${MAX_CORNER_SHIFT:-5}"
MAX_ROTATION_DEG="${MAX_ROTATION_DEG:-3}"
MAX_SCALE_JITTER="${MAX_SCALE_JITTER:-0.03}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SEED="${SEED:-42}"
VAL_SEED_OFFSET="${VAL_SEED_OFFSET:-100000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
CACHE_IMAGES="${CACHE_IMAGES:-1}"
DISABLE_AMP="${DISABLE_AMP:-0}"
SKIP_PROMOTION="${SKIP_PROMOTION:-0}"
MAX_PAIRS="${MAX_PAIRS:-}"
FIXED_EVAL_SAMPLES="${FIXED_EVAL_SAMPLES:-}"
FIXED_EVAL_PYTHON="${FIXED_EVAL_PYTHON:-$PYTHON_BIN}"
FIXED_EVAL_BASELINE="${FIXED_EVAL_BASELINE:-}"
SAR_DIR="${SAR_DIR:-}"
OPTICAL_DIR="${OPTICAL_DIR:-}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

resolve_path() {
  local raw_path="$1"
  if [[ -z "$raw_path" ]]; then
    return 0
  fi
  if [[ "$raw_path" = /* ]]; then
    printf '%s\n' "$raw_path"
  else
    printf '%s\n' "$PROJECT_ROOT/$raw_path"
  fi
}

CMD=(
  "$PYTHON_BIN"
  "$PROJECT_ROOT/train_multiscale.py"
  "--dataset-root" "$DATASET_ROOT"
  "--resolution-subset" "$RESOLUTION_SUBSET"
  "--task" "$TASK"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
  "--grad-accum-steps" "$GRAD_ACCUM_STEPS"
  "--learning-rate" "$LEARNING_RATE"
  "--weight-decay" "$WEIGHT_DECAY"
  "--input-size" "$INPUT_SIZE"
  "--max-shift" "$MAX_SHIFT"
  "--max-corner-shift" "$MAX_CORNER_SHIFT"
  "--max-rotation-deg" "$MAX_ROTATION_DEG"
  "--max-scale-jitter" "$MAX_SCALE_JITTER"
  "--train-ratio" "$TRAIN_RATIO"
  "--seed" "$SEED"
  "--val-seed-offset" "$VAL_SEED_OFFSET"
  "--num-workers" "$NUM_WORKERS"
  "--prefetch-factor" "$PREFETCH_FACTOR"
  "--fixed-eval-python" "$FIXED_EVAL_PYTHON"
  "--output-dir" "$OUTPUT_DIR"
)

if [[ "$CACHE_IMAGES" == "1" ]]; then
  CMD+=("--cache-images")
else
  CMD+=("--no-cache-images")
fi

if [[ "$DISABLE_AMP" == "1" ]]; then
  CMD+=("--disable-amp")
fi

if [[ "$SKIP_PROMOTION" == "1" ]]; then
  CMD+=("--skip-promotion")
fi

if [[ -n "$INIT_CHECKPOINT" ]]; then
  CMD+=("--init-checkpoint" "$(resolve_path "$INIT_CHECKPOINT")")
fi

if [[ -n "$MAX_PAIRS" ]]; then
  CMD+=("--max-pairs" "$MAX_PAIRS")
fi

if [[ -n "$FIXED_EVAL_SAMPLES" ]]; then
  CMD+=("--fixed-eval-samples" "$FIXED_EVAL_SAMPLES")
fi

if [[ -n "$FIXED_EVAL_BASELINE" ]]; then
  CMD+=("--fixed-eval-baseline" "$(resolve_path "$FIXED_EVAL_BASELINE")")
fi

if [[ -n "$SAR_DIR" ]]; then
  CMD+=("--sar-dir" "$SAR_DIR")
fi

if [[ -n "$OPTICAL_DIR" ]]; then
  CMD+=("--optical-dir" "$OPTICAL_DIR")
fi

echo "Project root: $PROJECT_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "Log file:     $LOG_FILE"
echo "Python:       $PYTHON_BIN"
echo "Command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"
