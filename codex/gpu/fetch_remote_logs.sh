#!/usr/bin/env bash
set -euo pipefail

GPU_USER="${GPU_USER:-root}"
GPU_HOST="${GPU_HOST:-175.155.64.225}"
GPU_PORT="${GPU_PORT:-19765}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/vllm}"
LOCAL_DIR="${LOCAL_DIR:-logs/remote_gpu}"

mkdir -p "${LOCAL_DIR}"
find "${LOCAL_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

echo "Mirroring logs from ${GPU_USER}@${GPU_HOST}:${REMOTE_REPO}/logs/"
echo "Local destination: ${LOCAL_DIR}"

ssh -p "${GPU_PORT}" "${GPU_USER}@${GPU_HOST}" \
  "test -d '${REMOTE_REPO}/logs' && cd '${REMOTE_REPO}/logs' && tar -cf - ." \
  | tar -xf - -C "${LOCAL_DIR}"

echo
echo "Mirrored files:"
find "${LOCAL_DIR}" -maxdepth 1 -type f -print | sort
