#!/usr/bin/env bash
# Load ModalPipeline/.env and push HF_TOKEN to Modal secret hf-token.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Create .env from .env.example first:"
  echo "  cp .env.example .env"
  exit 1
fi

python3 scripts/sync_secrets_from_env.py "$@"
