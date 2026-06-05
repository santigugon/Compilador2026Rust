#!/usr/bin/env bash
# Use the project venv (avoids system python3 missing PyYAML, modal, etc.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Create the venv first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
exec "${ROOT}/.venv/bin/python" run_experiment.py "$@"
