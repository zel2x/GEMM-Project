#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec 9>"$GEMM_ROOT/build/gpu.lock"
flock -n 9 || { echo 'Another project GPU job is running.' >&2; exit 2; }
exec "$PYTHON_BIN" "$GEMM_ROOT/tools/project.py" test "$@"
