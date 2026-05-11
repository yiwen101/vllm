# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a vLLM generate probe for a multimodal model."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "openvla/openvla-7b"
MODEL_IMPL = "transformers"
TRUST_REMOTE_CODE = True
DTYPE = "bfloat16"
PROMPT = "In: What action should the robot take to pick up the object?\nOut:"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
MAX_NEW_TOKENS = 8
JSON_OUT = Path("logs/openvla_vllm_transformers.json")


def make_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


def patch_transformers_remote_code_compat() -> list[str]:
    """Patch small Transformers API moves that old HF remote code imports."""
    patches: list[str] = []

    try:
        import transformers.tokenization_utils as tokenization_utils
        import transformers.tokenization_utils_base as tokenization_utils_base

        names = (
            "PaddingStrategy",
            "PreTokenizedInput",
            "TextInput",
            "TruncationStrategy",
        )
        for name in names:
            if hasattr(tokenization_utils, name):
                continue
            if not hasattr(tokenization_utils_base, name):
                patches.append(f"tokenization_utils_base.{name} missing")
                continue
            setattr(
                tokenization_utils,
                name,
                getattr(tokenization_utils_base, name),
            )
            patches.append(f"tokenization_utils.{name}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        patches.append(f"tokenization patch failed: {type(exc).__name__}: {exc}")

    return patches


def run_probe() -> dict[str, Any]:
    compat_patches = patch_transformers_remote_code_compat()
    from vllm import LLM, SamplingParams

    image = make_image()

    result: dict[str, Any] = {
        "model": MODEL_ID,
        "model_impl": MODEL_IMPL,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "prompt": PROMPT,
        "compat_patches": compat_patches,
        "status": "unknown",
    }

    llm = LLM(
        model=MODEL_ID,
        model_impl=MODEL_IMPL,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    outputs = llm.generate(
        {
            "prompt": PROMPT,
            "multi_modal_data": {"image": image},
        },
        SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        ),
        use_tqdm=False,
    )

    first = outputs[0].outputs[0]
    result["text"] = first.text
    result["token_ids"] = list(first.token_ids)
    result["status"] = "success"
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("vLLM generate probe")
    print("status:", result["status"])
    print("model:", result.get("model"))
    print("model_impl:", result.get("model_impl"))
    if "token_ids" in result:
        print("token_ids:", result["token_ids"])
    if "text" in result:
        print("text:", result["text"])


def main() -> None:
    try:
        result = run_probe()
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        result = {
            "status": "failed",
            "model_impl": MODEL_IMPL,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    print_summary(result)
    if result["status"] == "failed":
        print()
        print(result["traceback"])

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Wrote JSON:", JSON_OUT)

    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
