#!/usr/bin/env python3
"""Split the unified model.safetensors into per-component files.

This dramatically reduces memory usage on 32GB machines by allowing each
component (transformer, vae_decoder, etc.) to be loaded independently
without pulling the entire 15GB+ file into memory.

Usage:
    python scripts/split_model.py [--model-dir PATH]

Default model dir: ~/.cache/huggingface/hub/ltx2-mlx-av-int4/
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx


# Component prefixes and their output filenames
COMPONENT_MAP = {
    "transformer": "transformer.safetensors",
    "connector": "connector.safetensors",
    "text_embedding_projection": "connector.safetensors",  # merge with connector
    "vae_decoder": "vae_decoder.safetensors",
    "vae_encoder": "vae_encoder.safetensors",
    "vocoder": "vocoder.safetensors",
    "audio_vae": "audio_vae.safetensors",
}


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def split_model(model_dir: Path) -> None:
    unified_path = model_dir / "model.safetensors"
    if not unified_path.exists():
        print(f"ERROR: {unified_path} not found")
        sys.exit(1)

    print(f"Loading: {unified_path}")
    all_weights = mx.load(str(unified_path))
    print(f"Loaded {len(all_weights)} tensors")

    # Group weights by output file
    file_weights: dict[str, dict[str, mx.array]] = defaultdict(dict)
    unmatched: dict[str, mx.array] = {}

    for key, value in all_weights.items():
        prefix = key.split(".")[0]
        if prefix in COMPONENT_MAP:
            output_file = COMPONENT_MAP[prefix]
            file_weights[output_file][key] = value
        else:
            unmatched[key] = value

    if unmatched:
        print(f"WARNING: {len(unmatched)} unmatched keys, adding to transformer.safetensors:")
        for k in sorted(unmatched)[:5]:
            print(f"  {k}")
        file_weights["transformer.safetensors"].update(unmatched)

    # Save each component
    for filename, weights in sorted(file_weights.items()):
        output_path = model_dir / filename
        total_bytes = sum(v.nbytes for v in weights.values())
        print(f"Saving: {filename} ({len(weights)} tensors, {format_bytes(total_bytes)})")
        mx.save_safetensors(str(output_path), weights)

    # Write a marker file so our patch knows this is a split model
    marker = model_dir / "split_model.json"
    import json
    with open(marker, "w") as f:
        json.dump({
            "split": True,
            "files": {name: len(weights) for name, weights in file_weights.items()},
        }, f, indent=2)

    print()
    print("Split complete. Original model.safetensors can be removed to save disk space.")
    print(f"To remove: rm '{unified_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split unified model.safetensors into components")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/ltx2-mlx-av-int4",
        help="Model directory containing model.safetensors",
    )
    args = parser.parse_args()
    split_model(args.model_dir)


if __name__ == "__main__":
    main()
