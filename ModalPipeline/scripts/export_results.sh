#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="${1:-qwen3_coder_tb_v1}"
LOCAL_ROOT="${2:-./local_results/${EXPERIMENT_ID}}"
SEED="${3:-0}"
printf -v SEED_DIR "seed_%03d" "${SEED}"
LOCAL_DIR="${LOCAL_ROOT}/${SEED_DIR}"

mkdir -p "${LOCAL_DIR}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

modal volume get qwen3-coder-results "/${EXPERIMENT_ID}" "${TMP_DIR}"
cp -R "${TMP_DIR}/." "${LOCAL_DIR}/"
echo "Exported to ${LOCAL_DIR}"
