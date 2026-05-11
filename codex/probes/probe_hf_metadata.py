# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect HuggingFace metadata without loading model weights."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def dotted_type(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(k): jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(v) for v in value]
        return repr(value)


def summarize_config(config: object) -> dict[str, Any]:
    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    interesting_keys = [
        "model_type",
        "architectures",
        "auto_map",
        "torch_dtype",
        "vocab_size",
        "hidden_size",
        "image_token_index",
        "vision_config",
        "text_config",
    ]

    return {
        "class": dotted_type(config),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "auto_map": getattr(config, "auto_map", None),
        "keys": sorted(config_dict.keys()),
        "selected": {
            key: jsonable(config_dict.get(key))
            for key in interesting_keys
            if key in config_dict
        },
    }


def summarize_processor(processor: object) -> dict[str, Any]:
    names = dir(processor)
    method_terms = [
        "token",
        "image",
        "modal",
        "process",
        "num",
        "size",
        "patch",
    ]
    interesting_names = [
        name
        for name in names
        if not name.startswith("__")
        and any(term in name.lower() for term in method_terms)
    ]

    required_probe_names = [
        "_get_num_multimodal_tokens",
        "get_num_multimodal_tokens",
        "_get_num_image_tokens",
        "get_num_image_tokens",
        "image_processor",
        "tokenizer",
    ]

    return {
        "class": dotted_type(processor),
        "module": type(processor).__module__,
        "selected_attrs": {
            name: hasattr(processor, name) for name in required_probe_names
        },
        "interesting_names": interesting_names[:120],
    }


def summarize_tokenizer(tokenizer: object) -> dict[str, Any]:
    return {
        "class": dotted_type(tokenizer),
        "module": type(tokenizer).__module__,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "additional_special_tokens": getattr(
            tokenizer, "additional_special_tokens", None
        ),
    }


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


def run_probe(
    model: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    result: dict[str, Any] = {
        "model": model,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "compat_patches": patch_transformers_remote_code_compat(),
        "checks": {},
    }

    checks: dict[str, Any] = result["checks"]

    try:
        config = AutoConfig.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        checks["config"] = {
            "status": "ok",
            "summary": summarize_config(config),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        checks["config"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        processor = AutoProcessor.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        checks["processor"] = {
            "status": "ok",
            "summary": summarize_processor(processor),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        checks["processor"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        checks["tokenizer"] = {
            "status": "ok",
            "summary": summarize_tokenizer(tokenizer),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        checks["tokenizer"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    return result


def print_summary(result: dict[str, Any]) -> None:
    print("HF metadata probe")
    print("model:", result["model"])
    print("revision:", result["revision"] or "<default>")
    print("trust_remote_code:", result["trust_remote_code"])
    print()

    for name, check in result["checks"].items():
        print(f"[{name}] {check['status']}")
        if check["status"] != "ok":
            print("  error_type:", check["error_type"])
            print("  error:", check["error"])
            continue

        summary = check["summary"]
        print("  class:", summary.get("class"))

        if name == "config":
            print("  model_type:", summary.get("model_type"))
            print("  architectures:", summary.get("architectures"))
            print("  auto_map:", summary.get("auto_map"))
            print("  key_count:", len(summary.get("keys", [])))

        if name == "processor":
            print("  selected_attrs:")
            for attr, exists in summary["selected_attrs"].items():
                print(f"    {attr}: {exists}")
            print("  interesting_names_sample:")
            for item in summary["interesting_names"][:30]:
                print("   ", item)

        if name == "tokenizer":
            print("  model_max_length:", summary.get("model_max_length"))
            print("  bos_token:", summary.get("bos_token"))
            print("  eos_token:", summary.get("eos_token"))
            print("  pad_token:", summary.get("pad_token"))

        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect HF config/processor/tokenizer metadata.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    print_summary(result)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("Wrote JSON:", out_path)


if __name__ == "__main__":
    main()
