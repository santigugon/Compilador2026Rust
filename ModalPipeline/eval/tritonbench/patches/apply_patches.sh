#!/bin/bash
set -euo pipefail
REPO_DIR="${1:-/opt/TritonBench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}" \
  python -m pipeline.eval.tritonbench.patch_repo "${REPO_DIR}"
