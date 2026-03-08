#!/usr/bin/env python3
"""Quantize the LTX-2 MLX unified model from bf16 to int4 or int8.

Only quantizes transformer weights (Linear layers with ndim >= 2 and enough
elements). VAE, vocoder, connector, upsampler, and audio weights are kept in
their original precision.

Usage:
    python scripts/quantize_model.py [--bits 4|8] [--output-dir PATH] [--group-size 64]

The script auto-detects the model in the HuggingFace cache at:
    ~/.cache/huggingface/hub/models--notapalindrome--ltx2-mlx-av/snapshots/<hash>/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

# Weight prefixes that should NOT be quantized (keep original precision).
KEEP_PREFIXES = (
    "vae_decoder.",
    "vae_encoder.",
    "audio_vae.",
    "vocoder.",
    "connector.",
    "upsampler.",
)

# The prefix for weights that SHOULD be quantized.
QUANTIZE_PREFIX = "transformer."

# Minimum number of elements for a weight to be worth quantizing.
MIN_ELEMENTS = 256


def find_model_dir() -> Path:
    """Locate the notapalindrome/ltx2-mlx-av snapshot in the HF cache."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / "models--notapalindrome--ltx2-mlx-av"

    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}")
        print("Download it first with:")
        print('  python -c "from huggingface_hub import snapshot_download; '
              "snapshot_download('notapalindrome/ltx2-mlx-av')\"")
        sys.exit(1)

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print(f"ERROR: No snapshots directory found at {snapshots_dir}")
        sys.exit(1)

    # Pick the most recently modified snapshot hash directory.
    snapshot_dirs = sorted(
        snapshots_dir.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshot_dirs:
        print(f"ERROR: No snapshot hashes found in {snapshots_dir}")
        sys.exit(1)

    snapshot = snapshot_dirs[0]
    safetensors = snapshot / "model.safetensors"
    if not safetensors.exists():
        # Check for sharded safetensors.
        shards = list(snapshot.glob("model-*.safetensors")) + list(
            snapshot.glob("*.safetensors")
        )
        if not shards:
            print(f"ERROR: No safetensors files found in {snapshot}")
            sys.exit(1)

    print(f"Found model at: {snapshot}")
    return snapshot


def get_safetensors_files(model_dir: Path) -> list[Path]:
    """Return the list of safetensors files to process."""
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    # Sharded format.
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        return shards
    # Fallback: any safetensors.
    return sorted(model_dir.glob("*.safetensors"))


def should_quantize_weight(key: str, weight: mx.array) -> bool:
    """Decide whether a weight tensor should be quantized.

    Only quantizes nn.Linear ``.weight`` tensors in the transformer.
    Other 2D parameters (e.g. ``scale_shift_table``, embedding tables)
    are kept in their original precision.
    """
    # Only quantize transformer weights.
    if not key.startswith(QUANTIZE_PREFIX):
        return False

    # Only quantize actual Linear layer weights (keys ending in ".weight").
    # This excludes scale_shift_table, embedding tables, and other 2D params.
    if not key.endswith(".weight"):
        return False

    # Skip biases, norms, and other 1D tensors.
    if weight.ndim < 2:
        return False

    # Skip very small weights.
    if weight.size < MIN_ELEMENTS:
        return False

    # Skip 2D tensors where one dimension is 1 (e.g. norm weights reshaped).
    if weight.ndim == 2 and min(weight.shape) == 1:
        return False

    return True


def format_bytes(n: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def materialize_tensor(tensor: mx.array) -> None:
    """Force MLX to materialize a tensor (triggers actual GPU computation)."""
    # mx.eval is MLX's tensor materialization function — NOT Python's eval().
    # It forces lazy computation to complete and allocate the result in memory.
    mx.eval(tensor)  # noqa: S307 — this is mlx.core.eval, not builtins.eval


def quantize_model(
    bits: int = 4,
    group_size: int = 64,
    output_dir: Path | None = None,
) -> None:
    """Quantize the LTX-2 MLX model."""
    model_dir = find_model_dir()
    safetensors_files = get_safetensors_files(model_dir)

    if not safetensors_files:
        print("ERROR: No safetensors files found.")
        sys.exit(1)

    if output_dir is None:
        output_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"ltx2-mlx-av-int{bits}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Quantization: int{bits}, group_size={group_size}")
    print()

    # Copy metadata files (config.json, tokenizer files, etc.).
    metadata_extensions = {".json", ".txt", ".model", ".yaml", ".yml"}
    copied_metadata: list[str] = []
    for f in model_dir.iterdir():
        if f.suffix in metadata_extensions and f.is_file():
            shutil.copy2(f, output_dir / f.name)
            copied_metadata.append(f.name)
    if copied_metadata:
        print(f"Copied metadata: {', '.join(copied_metadata)}")
        print()

    # Process each safetensors file.
    total_original_bytes = 0
    total_quantized_bytes = 0
    total_keys = 0
    quantized_count = 0
    kept_count = 0
    skipped_not_divisible = 0

    t_start = time.time()

    for shard_path in safetensors_files:
        print(f"Loading: {shard_path.name}")
        # mx.load memory-maps the file — safe for large models.
        weights: dict[str, mx.array] = mx.load(str(shard_path))

        output_weights: dict[str, mx.array] = {}
        keys = sorted(weights.keys())
        total_keys += len(keys)

        # Try to use tqdm for a progress bar, fall back to plain printing.
        use_tqdm = False
        try:
            from tqdm import tqdm

            iterator = tqdm(keys, desc="Processing weights", unit="param")
            use_tqdm = True
        except ImportError:
            iterator = keys

        for key in iterator:
            w = weights[key]
            orig_bytes = w.nbytes
            total_original_bytes += orig_bytes

            if should_quantize_weight(key, w):
                # Check divisibility by group_size on the last dimension.
                if w.shape[-1] % group_size != 0:
                    skipped_not_divisible += 1
                    output_weights[key] = w
                    total_quantized_bytes += w.nbytes
                    kept_count += 1
                    continue

                # Quantize: returns (quantized_w, scales, biases) for affine mode.
                w_q, scales, biases = mx.quantize(
                    w, group_size=group_size, bits=bits
                )
                materialize_tensor(w_q)
                materialize_tensor(scales)
                materialize_tensor(biases)

                # Store with standard MLX quantized naming convention.
                output_weights[key] = w_q

                # Derive the base name for scales/biases keys.
                if key.endswith(".weight"):
                    base = key.removesuffix(".weight")
                else:
                    base = key
                output_weights[f"{base}.scales"] = scales
                output_weights[f"{base}.biases"] = biases

                q_bytes = w_q.nbytes + scales.nbytes + biases.nbytes
                total_quantized_bytes += q_bytes
                quantized_count += 1

                if not use_tqdm:
                    ratio = orig_bytes / q_bytes if q_bytes > 0 else 0
                    print(
                        f"  Quantized: {key} {list(w.shape)} "
                        f"-> {ratio:.1f}x reduction"
                    )
            else:
                output_weights[key] = w
                total_quantized_bytes += w.nbytes
                kept_count += 1

        # Save the quantized shard.
        output_filename = shard_path.name
        output_path = output_dir / output_filename

        print(f"Saving: {output_path.name} ({len(output_weights)} tensors)...")
        mx.save_safetensors(str(output_path), output_weights)
        print(f"Saved: {output_path}")
        print()

        # Free memory.
        del weights, output_weights

    elapsed = time.time() - t_start

    # Write a quantization config so loaders know the format.
    quant_config = {
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
            "quantized_prefixes": [QUANTIZE_PREFIX],
            "kept_prefixes": list(KEEP_PREFIXES),
        },
        "source_model": "notapalindrome/ltx2-mlx-av",
    }
    quant_config_path = output_dir / "quantize_config.json"
    with open(quant_config_path, "w") as f:
        json.dump(quant_config, f, indent=2)

    # Summary.
    reduction_pct = (
        (1 - total_quantized_bytes / total_original_bytes) * 100
        if total_original_bytes > 0
        else 0
    )

    print("=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"  Bits:              int{bits}")
    print(f"  Group size:        {group_size}")
    print(f"  Total parameters:  {total_keys}")
    print(f"  Quantized:         {quantized_count} (transformer Linear weights)")
    print(f"  Kept original:     {kept_count}")
    if skipped_not_divisible > 0:
        print(
            f"  Skipped (not divisible by group_size): {skipped_not_divisible}"
        )
    print(f"  Original size:     {format_bytes(total_original_bytes)}")
    print(f"  Quantized size:    {format_bytes(total_quantized_bytes)}")
    print(f"  Reduction:         {reduction_pct:.1f}%")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"  Output:            {output_dir}")
    print()

    if bits == 4:
        print(
            "  Estimated RAM usage: ~10-12 GB "
            "(down from ~38 GB for transformer weights)"
        )
    elif bits == 8:
        print(
            "  Estimated RAM usage: ~19-21 GB "
            "(down from ~38 GB for transformer weights)"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize the LTX-2 MLX model (transformer weights only) "
            "to int4 or int8."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quantize to int4 (default, maximum compression)
  python scripts/quantize_model.py

  # Quantize to int8 (better quality, less compression)
  python scripts/quantize_model.py --bits 8

  # Custom output directory
  python scripts/quantize_model.py --bits 4 --output-dir ~/models/ltx2-int4
        """,
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bit width (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory "
            "(default: ~/.cache/huggingface/hub/ltx2-mlx-av-int{bits}/)"
        ),
    )

    args = parser.parse_args()
    quantize_model(
        bits=args.bits, group_size=args.group_size, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
