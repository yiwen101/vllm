#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || "${2:-}" != "--" ]]; then
  echo "Usage: bash codex/gpu/run_with_log.sh <log-name> -- <command> [args...]"
  echo
  echo "Example:"
  echo "  bash codex/gpu/run_with_log.sh hf_metadata -- .venv/bin/python codex/probes/probe_hf_metadata.py --model openvla/openvla-7b --trust-remote-code"
  exit 2
fi

LOG_NAME="$1"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
SAFE_NAME="$(echo "${LOG_NAME}" | tr -c 'A-Za-z0-9_.-' '_')"
LOG_FILE="logs/${SAFE_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Writing log to ${LOG_FILE}"
echo "Command: $*"
echo

set +e
{
  echo "== ${LOG_NAME} =="
  echo "repo: ${ROOT_DIR}"
  echo "date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "command: $*"
  echo
  "$@"
} 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

echo
echo "Exit status: ${STATUS}"
echo "Log file: ${LOG_FILE}"
exit "${STATUS}"
