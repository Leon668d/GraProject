#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TRAIN_REQUIREMENTS_FILE="${TRAIN_REQUIREMENTS_FILE:-$PROJECT_ROOT/requirements.train.txt}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-0}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if ! "$PYTHON_BIN" -c "import torch, torchvision" >/dev/null 2>&1; then
  if [[ -z "$TORCH_INDEX_URL" ]]; then
    cat >&2 <<'EOF'
torch or torchvision is missing in the active environment.
Use an image that already includes PyTorch, or rerun with TORCH_INDEX_URL set.

Example:
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/bootstrap_linux.sh
EOF
    exit 1
  fi

  "$PYTHON_BIN" -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX_URL"
fi

INDEX_ARGS=()
if [[ -n "$PIP_INDEX_URL" ]]; then
  INDEX_ARGS=(--index-url "$PIP_INDEX_URL")
fi

if [[ ! -f "$TRAIN_REQUIREMENTS_FILE" ]]; then
  echo "Training requirements file not found: $TRAIN_REQUIREMENTS_FILE" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install "${INDEX_ARGS[@]}" -r "$TRAIN_REQUIREMENTS_FILE"

"$PYTHON_BIN" - <<'PY'
import platform
import torch
import torchvision

print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
