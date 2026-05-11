#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="logs/setup_vllm_gpu_$(date +%Y%m%d_%H%M%S).log"

{
  echo "== setup_vllm_gpu =="
  echo "repo: ${ROOT_DIR}"
  echo "date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  echo "uv: $(uv --version)"
  echo

  echo "Creating Python 3.12 virtualenv"
  uv venv --python 3.12

  # shellcheck disable=SC1091
  source .venv/bin/activate

  echo "python: $(.venv/bin/python --version)"
  echo

  echo "Installing vLLM editably with precompiled native extensions"
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

  echo
  echo "Installing extra probing dependencies for OpenVLA remote code"
  uv pip install "timm>=0.9.10,<1.0.0" accelerate torchvision

  echo
  echo "Environment summary"
  .venv/bin/python - <<'PY'
import importlib.util
import sys

print("python:", sys.version)

for name in ["torch", "transformers", "huggingface_hub", "timm", "vllm"]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"{name}: not installed")
        continue
    module = __import__(name)
    print(f"{name}:", getattr(module, "__version__", "<no __version__>"))
PY

  echo
  echo "GPU summary"
  nvidia-smi || true

  echo
  echo "Setup complete"
  echo "Log: ${LOG_FILE}"
} 2>&1 | tee "${LOG_FILE}"
