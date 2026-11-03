#!/usr/bin/env bash
set -euo pipefail
GEMM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GEMM_ROOT
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda "$HOME"/.local/cuda-training-env/*/usr/local/cuda-*; do
    if [[ -x "$candidate/bin/nvcc" ]]; then export CUDA_HOME="$candidate"; fi
  done
fi
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA toolkit with SM120 support}"
export NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
export CUDA_ARCH="${CUDA_ARCH:-sm_120}"
export PATH="$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in "$HOME"/.venvs/trtllm-*/bin/python; do
    if [[ -x "$candidate" ]]; then PYTHON_BIN="$candidate"; break; fi
  done
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
mkdir -p "$GEMM_ROOT/build/tmp" "$GEMM_ROOT/build/runs"
export TMPDIR="$GEMM_ROOT/build/tmp"
export MPLCONFIGDIR="$GEMM_ROOT/build/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
unset NVCC_PREPEND_FLAGS
