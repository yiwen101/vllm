# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print a compact GPU/server environment summary."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    module = __import__(name)
    return str(getattr(module, "__version__", "<no __version__>"))


def run_optional(command: list[str]) -> None:
    if shutil.which(command[0]) is None:
        print(f"{command[0]}: not found")
        return
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    print("exit_status:", completed.returncode)


def main() -> None:
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    print()

    for name in [
        "torch",
        "torchvision",
        "transformers",
        "huggingface_hub",
        "timm",
        "vllm",
    ]:
        print(f"{name}: {module_version(name)}")

    print()
    try:
        import torch

        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.cuda.device_count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                print(f"cuda:{idx}:", torch.cuda.get_device_name(idx))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print("torch CUDA summary failed:", type(exc).__name__, exc)

    print()
    run_optional(["nvidia-smi"])


if __name__ == "__main__":
    main()
