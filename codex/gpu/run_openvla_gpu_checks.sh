#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

RUNNER="codex/gpu/run_with_log.sh"
PY=".venv/bin/python"
MODEL="openvla/openvla-7b"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}. Run: bash codex/gpu/setup_vllm_gpu.sh"
  exit 2
fi

bash "${RUNNER}" env_summary -- "${PY}" codex/probes/probe_env_summary.py

bash "${RUNNER}" openvla_hf_metadata -- \
  "${PY}" codex/probes/probe_hf_metadata.py \
  --model "${MODEL}" \
  --trust-remote-code \
  --json-out logs/openvla_hf_metadata.json

bash "${RUNNER}" openvla_repo_files -- \
  "${PY}" codex/probes/inspect_hf_repo_files.py \
  --model "${MODEL}" \
  --json-out logs/openvla_repo_files.json

bash "${RUNNER}" openvla_hf_predict_action -- \
  "${PY}" codex/probes/probe_openvla_hf_predict_action.py

bash "${RUNNER}" openvla_vllm_transformers -- \
  "${PY}" codex/probes/probe_vllm_generate.py

echo
echo "All checks completed. Newest logs:"
ls -lt logs | head -20
