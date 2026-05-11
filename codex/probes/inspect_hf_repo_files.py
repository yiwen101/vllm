# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect HuggingFace Hub repository files without downloading weights."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

SMALL_METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
}


def format_bytes(size: int | None) -> str:
    if size is None:
        return "<unknown>"

    value = float(size)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    raise AssertionError("unreachable")


def file_category(path: str) -> str:
    if path.endswith(".py"):
        return "remote_code"
    if path.endswith((".safetensors", ".bin", ".pt")):
        return "weights"
    if path in SMALL_METADATA_FILES or path.endswith(".json"):
        return "metadata"
    if "token" in path.lower() or path.endswith(".model"):
        return "tokenizer"
    return "other"


def load_small_json(
    model: str,
    path: str,
    *,
    revision: str | None,
    max_download_bytes: int,
) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=model,
        filename=path,
        revision=revision,
    )
    file_path = Path(local_path)
    size = file_path.stat().st_size
    if size > max_download_bytes:
        return {
            "skipped": True,
            "reason": f"file is {format_bytes(size)}, above download limit",
        }

    data = json.loads(file_path.read_text())
    summary: dict[str, Any] = {
        "skipped": False,
        "size": size,
        "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else None,
    }

    if path == "config.json" and isinstance(data, dict):
        selected = [
            "model_type",
            "architectures",
            "auto_map",
            "torch_dtype",
            "vision_backbone_id",
            "llm_backbone_id",
            "image_resize_strategy",
        ]
        summary["selected"] = {key: data.get(key) for key in selected if key in data}

    if path == "model.safetensors.index.json" and isinstance(data, dict):
        weight_map = data.get("weight_map", {})
        metadata = data.get("metadata", {})
        summary["metadata"] = metadata
        summary["weight_count"] = len(weight_map) if isinstance(weight_map, dict) else 0
        if isinstance(weight_map, dict):
            shards = sorted(set(weight_map.values()))
            summary["shards"] = shards

    return summary


def inspect_repo(
    model: str,
    *,
    revision: str | None,
    max_download_bytes: int,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    result: dict[str, Any] = {
        "model": model,
        "revision": revision,
        "max_download_bytes": max_download_bytes,
        "files": [],
        "categories": {},
        "small_json": {},
        "errors": {},
    }

    api = HfApi()
    info = api.model_info(
        repo_id=model,
        revision=revision,
        files_metadata=True,
    )

    siblings = sorted(info.siblings, key=lambda sibling: sibling.rfilename)

    categories: dict[str, list[str]] = {}
    for sibling in siblings:
        path = sibling.rfilename
        size = getattr(sibling, "size", None)
        category = file_category(path)
        categories.setdefault(category, []).append(path)
        result["files"].append(
            {
                "path": path,
                "category": category,
                "size": size,
                "size_human": format_bytes(size),
            }
        )

    result["categories"] = categories

    candidate_json_files = [
        file_info["path"]
        for file_info in result["files"]
        if file_info["path"] in SMALL_METADATA_FILES
    ]

    for path in candidate_json_files:
        try:
            result["small_json"][path] = load_small_json(
                model,
                path,
                revision=revision,
                max_download_bytes=max_download_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            result["errors"][path] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    return result


def print_summary(result: dict[str, Any]) -> None:
    print("HF repository file inspection")
    print("model:", result["model"])
    print("revision:", result["revision"] or "<default>")
    print()

    print("Categories:")
    for category, files in sorted(result["categories"].items()):
        print(f"  {category}: {len(files)}")
        for path in files[:20]:
            file_info = next(item for item in result["files"] if item["path"] == path)
            print(f"    {path} ({file_info['size_human']})")
        if len(files) > 20:
            print(f"    ... {len(files) - 20} more")
    print()

    print("Small JSON summaries:")
    for path, summary in result["small_json"].items():
        print(f"  {path}:")
        if summary is None:
            print("    <none>")
            continue
        if summary.get("skipped"):
            print("    skipped:", summary["reason"])
            continue
        print("    size:", format_bytes(summary.get("size")))
        print("    top_level_keys:", summary.get("top_level_keys"))
        if "selected" in summary:
            print("    selected:", summary["selected"])
        if "shards" in summary:
            print("    shard_count:", len(summary["shards"]))
            for shard in summary["shards"]:
                print("     ", shard)
            print("    weight_count:", summary["weight_count"])

    if result["errors"]:
        print()
        print("Errors:")
        for path, error in result["errors"].items():
            print(f"  {path}: {error['error_type']}: {error['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect HF repo files and small metadata files.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=1_000_000,
        help="Maximum size for metadata files downloaded for JSON inspection.",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_repo(
        args.model,
        revision=args.revision,
        max_download_bytes=args.max_download_bytes,
    )
    print_summary(result)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("Wrote JSON:", out_path)


if __name__ == "__main__":
    main()
