#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
exec "$PYTHON_BIN" "$GEMM_ROOT/tools/project.py" tables "$@"
