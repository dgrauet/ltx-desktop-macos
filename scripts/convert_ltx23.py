#!/usr/bin/env python3
"""Convert Lightricks/LTX-2.3 PyTorch checkpoint to MLX split format.

Downloads the official LTX-2.3 distilled checkpoint from HuggingFace,
extracts components (transformer, VAE, vocoder, connector), applies
key sanitization and conv transpositions, and saves as split MLX files.

Processes tensors one component at a time via safe_open to stay within
32GB RAM. Optionally quantizes the transformer to int8 or int4.

Usage:
    cd backend
    uv run python ../scripts/convert_ltx23.py --output ~/.cache/huggingface/hub/ltx23-mlx
    uv run python ../scripts/convert_ltx23.py --output ~/.cache/huggingface/hub/ltx23-mlx --quantize --bits 8
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Key classification — which component does a weight key belong to?
# ---------------------------------------------------------------------------

def classify_key(key: str) -> str | None:
    """Classify a PyTorch weight key into a component name.

    Returns one of: transformer, connector, vae_decoder, vae_encoder,
    audio_vae, vocoder, or None (skip).
    """
    if key.startswith("model.diffusion_model."):
        suffix = key[len("model.diffusion_model."):]
        if suffix.startswith("video_embeddings_connector.") or suffix.startswith("audio_embeddings_connector."):
            return "connector"
        return "transformer"
    if key.startswith("vae.per_channel_statistics."):
        return "vae_shared_stats"  # Duplicated to both decoder and encoder
    if key.startswith("vae.encoder."):
        return "vae_encoder"
    if key.startswith("vae.decoder."):
        return "vae_decoder"
    if key.startswith("audio_vae."):
        return "audio_vae"
    if key.startswith("vocoder."):
        return "vocoder"
    if key.startswith("text_embedding_projection."):
        return "connector"
    return None  # Skip unknown keys


# ---------------------------------------------------------------------------
# Key sanitization — PyTorch names → MLX names
# ---------------------------------------------------------------------------

def sanitize_transformer_key(key: str) -> str:
    """Convert a PyTorch transformer key to MLX format."""
    k = key.replace("model.diffusion_model.", "")
    # Sequential wrapper removal
    k = k.replace(".to_out.0.", ".to_out.")
    # FeedForward renaming
    k = k.replace(".ff.net.0.proj.", ".ff.proj_in.")
    k = k.replace(".ff.net.2.", ".ff.proj_out.")
    k = k.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
    k = k.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
    # AdaLN linear naming
    k = k.replace(".linear_1.", ".linear1.")
    k = k.replace(".linear_2.", ".linear2.")
    return k


def sanitize_connector_key(key: str) -> str:
    """Convert a PyTorch connector/text_embedding key to MLX format."""
    if key.startswith("model.diffusion_model."):
        # Keep as connector.{video,audio}_embeddings_connector.xxx
        k = key.replace("model.diffusion_model.", "")
        # Don't sanitize connector FF keys here — they'll be sanitized
        # when the connector is loaded by the text encoder patch
        return k
    if key.startswith("text_embedding_projection."):
        # Keep as-is (text_embedding_projection.aggregate_embed.weight etc.)
        return key
    return key


def sanitize_vae_decoder_key(key: str) -> str:
    """Convert a PyTorch VAE decoder key to MLX format."""
    if key.startswith("vae.per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics.mean"
        if "std-of-means" in key:
            return "per_channel_statistics.std"
        return None  # Skip other stats
    if key.startswith("vae.decoder."):
        return key.replace("vae.decoder.", "")
    return None


def sanitize_vae_encoder_key(key: str) -> str:
    """Convert a PyTorch VAE encoder key to MLX format."""
    if key.startswith("vae.per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics._mean_of_means"
        if "std-of-means" in key:
            return "per_channel_statistics._std_of_means"
        return None
    if key.startswith("vae.encoder."):
        return key.replace("vae.encoder.", "")
    return None


def sanitize_audio_vae_key(key: str) -> str:
    """Convert a PyTorch audio VAE key to MLX format."""
    if key.startswith("audio_vae.decoder."):
        return key.replace("audio_vae.decoder.", "")
    if key.startswith("audio_vae.per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics._mean_of_means"
        if "std-of-means" in key:
            return "per_channel_statistics._std_of_means"
        return None
    return None


def sanitize_vocoder_key(key: str) -> str:
    """Convert a PyTorch vocoder key to MLX format."""
    if key.startswith("vocoder."):
        return key.replace("vocoder.", "")
    return None


SANITIZERS = {
    "transformer": sanitize_transformer_key,
    "connector": sanitize_connector_key,
    "vae_decoder": sanitize_vae_decoder_key,
    "vae_encoder": sanitize_vae_encoder_key,
    "audio_vae": sanitize_audio_vae_key,
    "vocoder": sanitize_vocoder_key,
}


# ---------------------------------------------------------------------------
# Conv transposition — PyTorch conv layout → MLX layout
# ---------------------------------------------------------------------------

def maybe_transpose_conv(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose conv weights from PyTorch to MLX layout if needed.

    Transformer Linear weights do NOT need transposition (both use [out, in]).
    Conv3d: PyTorch (O, I, D, H, W) → MLX (O, D, H, W, I)
    Conv2d: PyTorch (O, I, H, W) → MLX (O, H, W, I)
    Conv1d: PyTorch (O, I, K) → MLX (O, K, I)
    ConvTranspose1d: PyTorch (I, O, K) → MLX (O, K, I)
    """
    if component == "transformer":
        # Transformer has no conv layers — all Linear, no transpose needed
        return value

    is_conv = "conv" in key.lower() and "weight" in key

    if not is_conv:
        return value

    if value.ndim == 5:
        # Conv3d: (O, I, D, H, W) → (O, D, H, W, I)
        return mx.transpose(value, (0, 2, 3, 4, 1))
    if value.ndim == 4:
        # Conv2d: (O, I, H, W) → (O, H, W, I)
        return mx.transpose(value, (0, 2, 3, 1))
    if value.ndim == 3:
        if component == "vocoder" and "ups" in key:
            # ConvTranspose1d: (I, O, K) → (O, K, I)
            return mx.transpose(value, (1, 2, 0))
        # Conv1d: (O, I, K) → (O, K, I)
        return mx.transpose(value, (0, 2, 1))

    return value


# ---------------------------------------------------------------------------
# Config extraction from safetensors metadata
# ---------------------------------------------------------------------------

def extract_config(checkpoint_path: str) -> dict:
    """Read model config from safetensors file metadata."""
    _, metadata = mx.load(checkpoint_path, return_metadata=True)

    model_version = metadata.get("model_version", "unknown")
    is_v2 = model_version.startswith("2.3")

    config = {
        "model_version": model_version,
        "is_v2": is_v2,
        "model_type": "AudioVideo",
        # Transformer config
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        # V2 changes
        "caption_channels": None if is_v2 else 3840,
        "apply_gated_attention": is_v2,
        "cross_attention_adaln": is_v2,
        # Audio config
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        # Positional embedding config
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20],
        # Timestep config
        "timestep_scale_multiplier": 1000,
        "av_ca_timestep_scale_multiplier": 1000,
        "norm_eps": 1e-6,
    }

    # Try to parse embedded config JSON for VAE/vocoder details
    if "config" in metadata:
        try:
            embedded = json.loads(metadata["config"])
            config["embedded_config"] = embedded
        except json.JSONDecodeError:
            pass

    return config


# ---------------------------------------------------------------------------
# Component processing — load, sanitize, transpose, save
# ---------------------------------------------------------------------------

def process_component(
    all_weights: dict,
    component: str,
    keys: list[str],
    output_dir: Path,
    prefix: str,
) -> int:
    """Process one component: extract keys, sanitize, transpose, save.

    Uses the lazily-loaded all_weights dict from mx.load().
    Only materializes tensors for this component.

    Returns number of weights saved.
    """
    sanitizer = SANITIZERS[component]
    comp_weights = {}

    print(f"  Processing {len(keys)} tensors...")
    for key in keys:
        new_key = sanitizer(key)
        if new_key is None:
            continue

        tensor = all_weights[key]

        # Apply conv transposition
        tensor = maybe_transpose_conv(new_key, tensor, component)

        # Force-eval to materialize the tensor — mx.save_safetensors may not
        # correctly handle lazy tensors (especially transposed views of
        # memory-mapped data from another file).
        mx.eval(tensor)

        # Add prefix for split file format
        comp_weights[f"{prefix}.{new_key}"] = tensor

    if not comp_weights:
        print(f"  No weights for {component}, skipping")
        return 0

    count = len(comp_weights)

    # Save
    output_file = output_dir / f"{component}.safetensors"
    print(f"  Saving {count} weights to {output_file.name}...")
    mx.save_safetensors(str(output_file), comp_weights)

    # Free memory — delete references so MLX can release the evaluated tensors
    del comp_weights
    gc.collect()
    mx.clear_cache()

    return count


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_transformer(
    output_dir: Path,
    bits: int = 8,
    group_size: int = 64,
) -> None:
    """Quantize transformer weights in-place.

    Only quantizes transformer_blocks Linear layers.
    Keeps adaln_single, proj_out, patchify_proj, caption_projection in bf16.
    """
    transformer_file = output_dir / "transformer.safetensors"
    if not transformer_file.exists():
        print("ERROR: transformer.safetensors not found")
        return

    print(f"\nQuantizing transformer to int{bits} (group_size={group_size})...")
    weights = mx.load(str(transformer_file))

    # Separate: transformer_blocks weights get quantized, rest stays bf16
    quantized = {}
    kept_bf16 = {}

    for key, value in weights.items():
        # Only quantize weight matrices in transformer_blocks
        bare_key = key.replace("transformer.", "", 1)
        if (
            "transformer_blocks" in bare_key
            and bare_key.endswith(".weight")
            and value.ndim == 2
            and not bare_key.endswith(".scales")
            and not bare_key.endswith(".biases")
        ):
            quantized[key] = value
        else:
            kept_bf16[key] = value

    # Force-evaluate kept_bf16 tensors BEFORE quantizing.
    # mx.quantize() triggers GPU work that can evict the memory-mapped buffers
    # backing lazy tensors, zeroing them out.
    print(f"  Materializing {len(kept_bf16)} non-quantizable weights...")
    mx.eval(*kept_bf16.values())

    del weights  # Safe now — kept_bf16 values are materialized
    gc.collect()

    print(f"  Quantizing {len(quantized)} weight matrices...")
    result = {}
    result.update(kept_bf16)
    del kept_bf16

    for key, weight in quantized.items():
        # Force-eval lazy weight before quantizing — prevents OOM from
        # accumulated lazy graph and ensures mx.quantize gets real data.
        mx.eval(weight)
        q_weight, scales, biases = mx.quantize(weight, bits=bits, group_size=group_size)
        mx.eval(q_weight, scales, biases)
        result[key] = q_weight
        result[key.replace(".weight", ".scales")] = scales
        result[key.replace(".weight", ".biases")] = biases
        del weight, q_weight, scales, biases

    del quantized

    print(f"  Saving quantized transformer ({len(result)} keys)...")
    mx.save_safetensors(str(transformer_file), result)

    # Save quantize config
    qconfig = {
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "only_transformer_blocks": True,
        }
    }
    with open(output_dir / "quantize_config.json", "w") as f:
        json.dump(qconfig, f, indent=2)

    del result
    gc.collect()
    mx.clear_cache()
    print("  Quantization complete")


# ---------------------------------------------------------------------------
# Shared per_channel_statistics — duplicated to decoder and encoder
# ---------------------------------------------------------------------------

def process_shared_stats(
    all_weights: dict,
    keys: list[str],
    output_dir: Path,
) -> None:
    """Load shared VAE per_channel_statistics and append to decoder/encoder files."""
    for key in keys:
        tensor = all_weights[key]
        # Force-eval to materialize lazy tensor from the original file —
        # mx.save_safetensors may not properly handle cross-file lazy refs.
        mx.eval(tensor)

        # Append to decoder
        dec_file = output_dir / "vae_decoder.safetensors"
        if dec_file.exists():
            dec_weights = mx.load(str(dec_file))
            # Force-eval existing weights too
            for k in dec_weights:
                mx.eval(dec_weights[k])
        else:
            dec_weights = {}

        if "mean-of-means" in key:
            dec_weights["vae_decoder.per_channel_statistics.mean"] = tensor
        elif "std-of-means" in key:
            dec_weights["vae_decoder.per_channel_statistics.std"] = tensor
        mx.save_safetensors(str(dec_file), dec_weights)
        del dec_weights

        # Append to encoder
        enc_file = output_dir / "vae_encoder.safetensors"
        if enc_file.exists():
            enc_weights = mx.load(str(enc_file))
            for k in enc_weights:
                mx.eval(enc_weights[k])
        else:
            enc_weights = {}

        if "mean-of-means" in key:
            enc_weights["vae_encoder.per_channel_statistics._mean_of_means"] = tensor
        elif "std-of-means" in key:
            enc_weights["vae_encoder.per_channel_statistics._std_of_means"] = tensor
        mx.save_safetensors(str(enc_file), enc_weights)
        del enc_weights

        del tensor
        gc.collect()
        mx.clear_cache()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

# Component → save prefix (matches quantized_patch.py expectations)
COMPONENT_PREFIX = {
    "transformer": "transformer",
    "connector": "connector",
    "vae_decoder": "vae_decoder",
    "vae_encoder": "vae_encoder",
    "audio_vae": "audio_vae",
    "vocoder": "vocoder",
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lightricks/LTX-2.3 to MLX split format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.home() / ".cache/huggingface/hub/ltx23-mlx"),
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint file (skips download)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="distilled",
        choices=["distilled", "dev"],
        help="Model variant (default: distilled)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize transformer after conversion",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using local checkpoint: {checkpoint_path}")
    else:
        filename = f"ltx-2.3-22b-{args.variant}.safetensors"
        print(f"Downloading {filename} from Lightricks/LTX-2.3...")
        print("(This is ~46 GB, may take a while)")
        checkpoint_path = hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename=filename,
        )
        print(f"Downloaded to: {checkpoint_path}")

    # Step 2: Extract config from metadata
    print("\nExtracting config from safetensors metadata...")
    config = extract_config(checkpoint_path)
    print(f"  Model version: {config['model_version']}")
    print(f"  Is V2 (2.3): {config['is_v2']}")
    print(f"  Gated attention: {config['apply_gated_attention']}")
    print(f"  Cross-attention AdaLN: {config['cross_attention_adaln']}")

    # Save config
    config_out = {k: v for k, v in config.items() if k != "embedded_config"}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    # Save full embedded config separately if present
    if "embedded_config" in config:
        with open(output_dir / "embedded_config.json", "w") as f:
            json.dump(config["embedded_config"], f, indent=2)
    print(f"  Config saved to {output_dir / 'config.json'}")

    # Step 3: Load all weights lazily via mx.load
    # mx.load memory-maps the safetensors file — tensors are not materialized
    # until accessed, so this uses ~0 GB RAM initially.
    print("\nLoading weights lazily via mx.load...")
    t0 = time.monotonic()
    all_weights = mx.load(checkpoint_path)
    print(f"  {len(all_weights)} keys loaded (lazy, {mx.get_active_memory() / (1024**3):.2f} GB active)")

    # Classify keys by component
    print("\nClassifying weight keys...")
    component_keys: dict[str, list[str]] = {}
    for key in all_weights:
        comp = classify_key(key)
        if comp:
            component_keys.setdefault(comp, []).append(key)

    for comp, keys in sorted(component_keys.items()):
        print(f"  {comp}: {len(keys)} keys")
    print(f"  Loaded + classified in {time.monotonic() - t0:.1f}s")

    # Step 4: Process each component
    total_weights = 0
    process_order = ["transformer", "connector", "vae_decoder", "vae_encoder", "audio_vae", "vocoder"]

    for component in process_order:
        keys = component_keys.get(component, [])
        if not keys:
            print(f"\n[{component}] No keys found, skipping")
            continue

        prefix = COMPONENT_PREFIX[component]
        print(f"\n[{component}] Processing {len(keys)} keys...")
        t0 = time.monotonic()
        count = process_component(all_weights, component, keys, output_dir, prefix)
        elapsed = time.monotonic() - t0
        total_weights += count
        print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    # Handle shared per_channel_statistics
    shared_keys = component_keys.get("vae_shared_stats", [])
    if shared_keys:
        print(f"\n[shared stats] Processing {len(shared_keys)} per_channel_statistics keys...")
        process_shared_stats(all_weights, shared_keys, output_dir)

    # Free the lazy-loaded weights
    del all_weights
    gc.collect()
    mx.clear_cache()

    # Step 5: Create split_model.json marker
    split_info = {
        "format": "split",
        "model_version": config["model_version"],
        "components": list(COMPONENT_PREFIX.keys()),
        "source": "Lightricks/LTX-2.3",
        "variant": args.variant,
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")

    # List output files with sizes
    for p in sorted(output_dir.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}: {size_mb:.1f} MB")

    # Step 6: Optional quantization
    if args.quantize:
        quantize_transformer(output_dir, bits=args.bits, group_size=args.group_size)

        # Update split_model.json
        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
        with open(output_dir / "split_model.json", "w") as f:
            json.dump(split_info, f, indent=2)

        # Show final sizes
        print(f"\nFinal files after quantization:")
        for p in sorted(output_dir.iterdir()):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {p.name}: {size_mb:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
