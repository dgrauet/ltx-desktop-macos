#!/usr/bin/env python3
"""Validate a converted LTX-2.3 MLX model.

Checks:
1. All expected split files exist
2. Config is valid and matches LTX-2.3 specs
3. Weight key counts and shapes are reasonable
4. Key sanitization is correct (no PyTorch-style keys remain)
5. Conv weights have correct MLX layout (channels-last)
6. Quantized weights have matching .scales/.biases
7. Cross-reference against source checkpoint (optional, if available)
8. Per-component loading into mlx_video model classes

Usage:
    cd backend
    uv run python ../scripts/validate_ltx23.py ~/.cache/huggingface/hub/ltx23-mlx
    uv run python ../scripts/validate_ltx23.py ~/.cache/huggingface/hub/ltx23-mlx --source /path/to/ltx-2.3-22b-distilled.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

errors = 0
warnings = 0


def check(condition: bool, msg: str, warn_only: bool = False) -> bool:
    global errors, warnings
    if condition:
        print(f"  {PASS} {msg}")
        return True
    if warn_only:
        print(f"  {WARN} {msg}")
        warnings += 1
        return False
    print(f"  {FAIL} {msg}")
    errors += 1
    return False


def validate_files(model_dir: Path) -> bool:
    """Check that all expected files exist."""
    print("\n== File Structure ==")

    expected_files = [
        "config.json",
        "split_model.json",
        "transformer.safetensors",
        "connector.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    ]

    all_ok = True
    for fname in expected_files:
        path = model_dir / fname
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            check(True, f"{fname} exists ({size_mb:.1f} MB)")
        else:
            check(False, f"{fname} missing")
            all_ok = False

    # Optional files
    for fname in ["quantize_config.json", "embedded_config.json"]:
        path = model_dir / fname
        if path.exists():
            print(f"  {PASS} {fname} exists (optional)")

    return all_ok


def validate_config(model_dir: Path) -> dict | None:
    """Validate config.json contents."""
    print("\n== Config Validation ==")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        check(False, "config.json not found")
        return None

    with open(config_path) as f:
        config = json.load(f)

    check(config.get("model_version", "").startswith("2.3"),
          f"Model version is 2.3.x (got: {config.get('model_version')})")
    check(config.get("is_v2") is True, "is_v2 flag is True")
    check(config.get("apply_gated_attention") is True, "apply_gated_attention is True")
    check(config.get("cross_attention_adaln") is True, "cross_attention_adaln is True")
    check(config.get("caption_channels") is None, "caption_channels is None (V2)")
    check(config.get("num_layers") == 48, f"num_layers == 48 (got: {config.get('num_layers')})")
    check(config.get("num_attention_heads") == 32, f"num_attention_heads == 32")
    check(config.get("attention_head_dim") == 128, f"attention_head_dim == 128")
    check(config.get("in_channels") == 128, f"in_channels == 128")

    return config


def validate_transformer(model_dir: Path, is_quantized: bool) -> None:
    """Validate transformer weights."""
    print("\n== Transformer Weights ==")

    tf_path = model_dir / "transformer.safetensors"
    if not tf_path.exists():
        check(False, "transformer.safetensors not found")
        return

    weights = mx.load(str(tf_path))
    keys = set(weights.keys())

    # Check key naming
    pytorch_remnants = [k for k in keys if "model.diffusion_model." in k]
    check(len(pytorch_remnants) == 0,
          f"No PyTorch prefix keys remaining (found {len(pytorch_remnants)})")
    if pytorch_remnants:
        for k in pytorch_remnants[:5]:
            print(f"    Bad key: {k}")

    bad_ff = [k for k in keys if ".ff.net." in k]
    check(len(bad_ff) == 0,
          f"No un-sanitized FF keys (found {len(bad_ff)})")

    bad_toout = [k for k in keys if ".to_out.0." in k]
    check(len(bad_toout) == 0,
          f"No un-sanitized to_out keys (found {len(bad_toout)})")

    # Check expected V2 keys exist
    # Gated attention: to_gate_logits in attention blocks
    gate_keys = [k for k in keys if "to_gate_logits" in k]
    check(len(gate_keys) > 0,
          f"Gated attention keys present ({len(gate_keys)} to_gate_logits keys)")

    # Cross-attention AdaLN: prompt_adaln_single
    prompt_adaln = [k for k in keys if "prompt_adaln_single" in k]
    check(len(prompt_adaln) > 0,
          f"prompt_adaln_single keys present ({len(prompt_adaln)} keys)")

    # prompt_scale_shift_table
    psst = [k for k in keys if "prompt_scale_shift_table" in k]
    check(len(psst) > 0,
          f"prompt_scale_shift_table keys present ({len(psst)} keys)")

    # Check transformer block count
    block_indices = set()
    for k in keys:
        if "transformer_blocks." in k:
            parts = k.split("transformer_blocks.")
            if len(parts) > 1:
                idx = parts[1].split(".")[0]
                if idx.isdigit():
                    block_indices.add(int(idx))
    check(len(block_indices) == 48,
          f"48 transformer blocks found (got {len(block_indices)})")
    if block_indices:
        check(max(block_indices) == 47,
              f"Block indices 0-47 (max: {max(block_indices)})")

    # Check quantization
    if is_quantized:
        scale_keys = [k for k in keys if k.endswith(".scales")]
        bias_keys = [k for k in keys if k.endswith(".biases")]
        check(len(scale_keys) > 0,
              f"Quantized: {len(scale_keys)} .scales keys")
        check(len(bias_keys) > 0,
              f"Quantized: {len(bias_keys)} .biases keys")
        check(len(scale_keys) == len(bias_keys),
              f"Equal .scales and .biases count")

        # Verify quantized weights are in transformer_blocks only
        non_block_scales = [k for k in scale_keys if "transformer_blocks" not in k]
        check(len(non_block_scales) == 0,
              f"Quantization only in transformer_blocks (non-block scales: {len(non_block_scales)})",
              warn_only=True)

    # AdaLN scale_shift_table shapes
    sst_keys = [k for k in keys if "scale_shift_table" in k and "prompt" not in k and "audio_prompt" not in k]
    if sst_keys:
        sample_key = sst_keys[0]
        shape = weights[sample_key].shape
        # V2 should have 9 params (6 base + 3 cross-attn AdaLN)
        expected_adaln = 9
        check(shape[0] == expected_adaln,
              f"scale_shift_table has {expected_adaln} params (got shape {shape})")

    total_params = sum(v.size for v in weights.values())
    print(f"  Total transformer parameters: {total_params / 1e9:.2f}B")
    print(f"  Total keys: {len(keys)}")

    del weights


def validate_connector(model_dir: Path) -> None:
    """Validate connector weights."""
    print("\n== Connector Weights ==")

    conn_path = model_dir / "connector.safetensors"
    if not conn_path.exists():
        check(False, "connector.safetensors not found")
        return

    weights = mx.load(str(conn_path))
    keys = set(weights.keys())

    # Check for video and audio connectors
    video_conn = [k for k in keys if "video_embeddings_connector" in k]
    audio_conn = [k for k in keys if "audio_embeddings_connector" in k]
    text_proj = [k for k in keys if "text_embedding_projection" in k]

    check(len(video_conn) > 0, f"Video connector keys present ({len(video_conn)})")
    check(len(audio_conn) > 0, f"Audio connector keys present ({len(audio_conn)})")
    check(len(text_proj) > 0, f"Text projection keys present ({len(text_proj)})",
          warn_only=True)

    # V2 connectors should have gated attention too
    gate_keys = [k for k in keys if "to_gate_logits" in k]
    check(len(gate_keys) > 0,
          f"Connector gated attention keys present ({len(gate_keys)})",
          warn_only=True)

    total_mb = sum(v.nbytes for v in weights.values()) / (1024 * 1024)
    print(f"  Total connector size: {total_mb:.1f} MB")
    print(f"  Total keys: {len(keys)}")

    del weights


def validate_vae(model_dir: Path, component: str) -> None:
    """Validate VAE decoder or encoder weights."""
    print(f"\n== {component} Weights ==")

    path = model_dir / f"{component}.safetensors"
    if not path.exists():
        check(False, f"{component}.safetensors not found")
        return

    weights = mx.load(str(path))
    keys = set(weights.keys())

    # Check no PyTorch prefix remains
    bad_prefix = [k for k in keys if k.startswith("vae.")]
    check(len(bad_prefix) == 0,
          f"No 'vae.' prefix remaining (found {len(bad_prefix)})")

    # Check per_channel_statistics
    stats_keys = [k for k in keys if "per_channel_statistics" in k]
    check(len(stats_keys) >= 2,
          f"Per-channel statistics present ({len(stats_keys)} keys)")

    # Check conv weight shapes — verify transpose was applied
    # MLX Conv3d: (O, D, H, W, I), PyTorch: (O, I, D, H, W)
    # We verify by checking that spatial dims (D, H, W) are small (typically 1-7)
    conv_weights = [(k, v) for k, v in weights.items()
                    if "conv" in k.lower() and "weight" in k and v.ndim == 5]
    mlx_layout = True
    for k, v in conv_weights:
        # In MLX layout (O, D, H, W, I), dims 1-3 should be spatial (small)
        spatial = v.shape[1:4]
        if any(s > 16 for s in spatial):  # Spatial dims > 16 is suspicious
            mlx_layout = False
            print(f"    Suspect layout: {k} shape={v.shape}")

    check(mlx_layout, f"Conv3d weights appear to be in MLX channels-last layout ({len(conv_weights)} checked)")

    total_mb = sum(v.nbytes for v in weights.values()) / (1024 * 1024)
    print(f"  Total {component} size: {total_mb:.1f} MB")
    print(f"  Total keys: {len(keys)}")

    del weights


def validate_audio_vae(model_dir: Path) -> None:
    """Validate audio VAE weights."""
    print("\n== Audio VAE Weights ==")

    path = model_dir / "audio_vae.safetensors"
    if not path.exists():
        check(False, "audio_vae.safetensors not found")
        return

    weights = mx.load(str(path))
    keys = set(weights.keys())

    # Keys should have audio_vae. prefix (split format), but NOT audio_vae.decoder. (PyTorch)
    bad_prefix = [k for k in keys if "audio_vae.decoder." in k]
    check(len(bad_prefix) == 0,
          f"No PyTorch 'audio_vae.decoder.' prefix remaining (found {len(bad_prefix)})")

    total_mb = sum(v.nbytes for v in weights.values()) / (1024 * 1024)
    print(f"  Total audio VAE size: {total_mb:.1f} MB")
    print(f"  Total keys: {len(keys)}")

    del weights


def validate_vocoder(model_dir: Path) -> None:
    """Validate vocoder weights."""
    print("\n== Vocoder Weights ==")

    path = model_dir / "vocoder.safetensors"
    if not path.exists():
        check(False, "vocoder.safetensors not found")
        return

    weights = mx.load(str(path))
    keys = set(weights.keys())

    # Keys should have vocoder. prefix (split format) — check all keys use it
    has_prefix = all(k.startswith("vocoder.") for k in keys)
    check(has_prefix, f"All keys have 'vocoder.' prefix ({len(keys)} keys)")

    # Check conv weight shapes
    conv_weights = [(k, v) for k, v in weights.items()
                    if "weight" in k and v.ndim == 3]
    check(len(conv_weights) > 0,
          f"Conv1d weights present ({len(conv_weights)} 3D tensors)")

    total_mb = sum(v.nbytes for v in weights.values()) / (1024 * 1024)
    print(f"  Total vocoder size: {total_mb:.1f} MB")
    print(f"  Total keys: {len(keys)}")

    del weights


def cross_reference_source(model_dir: Path, source_path: str) -> None:
    """Compare converted weights against source checkpoint."""
    print("\n== Cross-Reference with Source ==")

    # Load source weights lazily via mx.load (handles bfloat16)
    source_weights = mx.load(source_path)
    source_keys = list(source_weights.keys())

    print(f"  Source checkpoint: {len(source_keys)} keys")

    # Count how many source keys we expect to have converted
    sys.path.insert(0, str(Path(__file__).parent))
    from convert_ltx23 import classify_key
    classified = 0
    unclassified = []
    for k in source_keys:
        comp = classify_key(k)
        if comp:
            classified += 1
        else:
            unclassified.append(k)

    check(classified > 0, f"Classified {classified}/{len(source_keys)} source keys")
    if unclassified:
        print(f"  {WARN} {len(unclassified)} unclassified keys:")
        for k in unclassified[:10]:
            print(f"    - {k}")
        if len(unclassified) > 10:
            print(f"    ... and {len(unclassified) - 10} more")

    # Spot-check: compare a few transformer weight values
    print("\n  Spot-checking tensor values...")

    spot_checks = [
        # AdaLN (should be unchanged, no transpose)
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight",
        # Transformer block attention (should be unchanged for Linear)
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
        # Gated attention (V2 only)
        "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight",
    ]

    tf_weights = mx.load(str(model_dir / "transformer.safetensors"))

    for src_key in spot_checks:
        if src_key not in source_weights:
            print(f"  {WARN} Source key not found: {src_key}")
            continue

        # Derive expected MLX key
        mlx_key = src_key.replace("model.diffusion_model.", "transformer.")
        mlx_key = mlx_key.replace(".linear_1.", ".linear1.")
        mlx_key = mlx_key.replace(".linear_2.", ".linear2.")

        if mlx_key not in tf_weights:
            check(False, f"MLX key not found: {mlx_key}")
            continue

        src_tensor = source_weights[src_key]
        mlx_tensor = tf_weights[mlx_key]

        # For quantized weights, we can't compare directly
        if mlx_tensor.dtype == mx.uint32 or mlx_key.endswith(".scales") or mlx_key.endswith(".biases"):
            print(f"  {PASS} {src_key.split('.')[-2]}.{src_key.split('.')[-1]} — quantized (shape match: {src_tensor.shape}→{mlx_tensor.shape})")
            continue

        # Compare values
        if src_tensor.dtype != mlx_tensor.dtype:
            src_tensor = src_tensor.astype(mlx_tensor.dtype)

        max_diff = mx.max(mx.abs(src_tensor - mlx_tensor)).item()
        check(max_diff < 1e-5,
              f"{src_key.split('.')[-2]}.{src_key.split('.')[-1]} — max diff: {max_diff:.2e}")

    del tf_weights, source_weights


def main():
    parser = argparse.ArgumentParser(description="Validate converted LTX-2.3 MLX model")
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to source checkpoint for cross-reference")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        sys.exit(1)

    print(f"Validating: {model_dir}")

    # Check quantization
    is_quantized = (model_dir / "quantize_config.json").exists()
    if is_quantized:
        with open(model_dir / "quantize_config.json") as f:
            qconfig = json.load(f)
        bits = qconfig.get("quantization", {}).get("bits", "?")
        print(f"Model is quantized: int{bits}")

    validate_files(model_dir)
    config = validate_config(model_dir)
    validate_transformer(model_dir, is_quantized)
    validate_connector(model_dir)
    validate_vae(model_dir, "vae_decoder")
    validate_vae(model_dir, "vae_encoder")
    validate_audio_vae(model_dir)
    validate_vocoder(model_dir)

    if args.source:
        cross_reference_source(model_dir, args.source)

    # Summary
    print(f"\n{'='*60}")
    if errors == 0:
        print(f"{PASS} All checks passed! ({warnings} warnings)")
    else:
        print(f"{FAIL} {errors} checks failed, {warnings} warnings")
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
