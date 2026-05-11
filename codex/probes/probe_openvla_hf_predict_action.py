# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run direct HuggingFace OpenVLA action prediction as a baseline."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "openvla/openvla-7b"
TRUST_REMOTE_CODE = True
REVISION = None
DTYPE = "bfloat16"
DEVICE = "cuda"
PROMPT = "In: What action should the robot take to pick up the object?\nOut:"
UNNORM_KEY = "bridge_orig"
MAX_NEW_TOKENS = 8
JSON_OUT = Path("logs/openvla_hf_predict_action.json")


def make_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


def dtype_from_name(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return mapping[name]


def tensor_to_jsonable(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass

    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): tensor_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(v) for v in value]
    return value


def patch_hf_model_instance_compat(model: Any) -> list[str]:
    """Patch instance state expected by newer Transformers generation code."""
    from transformers import GenerationConfig

    patches: list[str] = []

    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_model_config(model.config)
        patches.append("model.generation_config from model.config")

    return patches


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


def get_hf_model_class():
    import transformers
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    auto_map = getattr(config, "auto_map", None) or {}

    for auto_class_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        class_ref = auto_map.get(auto_class_name)
        auto_class = getattr(transformers, auto_class_name, None)
        if auto_class is not None and class_ref is not None:
            return auto_class_name, auto_class

    class_ref = auto_map.get("AutoModelForVision2Seq")
    if class_ref is None:
        raise ValueError(f"OpenVLA auto_map has no model class: {auto_map!r}")

    model_class = get_class_from_dynamic_module(
        class_ref,
        MODEL_ID,
        revision=REVISION,
    )
    return f"direct:{class_ref}", model_class


def patch_hf_model_class_compat(model_class: type) -> list[str]:
    """Patch small PreTrainedModel flags expected by newer Transformers."""
    from transformers.generation import GenerationMixin

    patches: list[str] = []

    for cls in model_class.mro():
        if cls.__module__.startswith("transformers_modules."):
            cls._supports_sdpa = False
            patches.append(f"{cls.__name__}._supports_sdpa=False")
            tie_weights = cls.__dict__.get("tie_weights")
            if tie_weights is not None:
                def make_tie_weights_compat(original):
                    def tie_weights_compat(self, *args, **kwargs):
                        try:
                            return original(self, *args, **kwargs)
                        except TypeError as exc:
                            if "unexpected keyword argument" not in str(exc):
                                raise
                            return original(self)

                    return tie_weights_compat

                cls.tie_weights = make_tie_weights_compat(tie_weights)
                patches.append(f"{cls.__name__}.tie_weights accepts new kwargs")
            if hasattr(cls, "prepare_inputs_for_generation") and not hasattr(
                cls, "generate"
            ):
                for name in dir(GenerationMixin):
                    if name.startswith("__") or hasattr(cls, name):
                        continue
                    value = getattr(GenerationMixin, name)
                    if callable(value):
                        setattr(cls, name, value)
                patches.append(f"{cls.__name__} gets GenerationMixin methods")

    return patches


def run_probe() -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    compat_patches = patch_transformers_remote_code_compat()
    model_loader_name, model_class = get_hf_model_class()
    model_class_patches = patch_hf_model_class_compat(model_class)
    dtype = dtype_from_name(DTYPE)
    image = make_image()

    result: dict[str, Any] = {
        "model": MODEL_ID,
        "revision": REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "dtype": str(dtype),
        "device": DEVICE,
        "prompt": PROMPT,
        "model_loader": model_loader_name,
        "compat_patches": compat_patches,
        "model_class_patches": model_class_patches,
        "status": "unknown",
    }

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    result["processor_class"] = (
        f"{type(processor).__module__}.{type(processor).__name__}"
    )

    model = model_class.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    result["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    result["model_instance_patches"] = patch_hf_model_instance_compat(model)
    model.to(DEVICE)
    model.eval()

    inputs = processor(PROMPT, image)
    if hasattr(inputs, "to"):
        inputs = inputs.to(DEVICE, dtype=dtype)
    result["input_keys"] = sorted(inputs.keys()) if hasattr(inputs, "keys") else None

    with torch.inference_mode():
        if hasattr(model, "predict_action"):
            action = model.predict_action(
                **inputs,
                do_sample=False,
                unnorm_key=UNNORM_KEY,
            )
            result["runner"] = "predict_action"
            result["action"] = tensor_to_jsonable(action)
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            result["runner"] = "generate"
            result["output_ids"] = tensor_to_jsonable(output_ids)
            if hasattr(processor, "decode"):
                result["decoded"] = processor.decode(output_ids[0])

    result["status"] = "success"
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("HF OpenVLA predict_action probe")
    print("status:", result["status"])
    print("model:", result.get("model"))
    print("processor_class:", result.get("processor_class"))
    print("model_class:", result.get("model_class"))
    print("runner:", result.get("runner"))
    print("input_keys:", result.get("input_keys"))
    if "action" in result:
        print("action:", result["action"])
    if "decoded" in result:
        print("decoded:", result["decoded"])


def main() -> None:
    try:
        result = run_probe()
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        result = {
            "status": "failed",
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
