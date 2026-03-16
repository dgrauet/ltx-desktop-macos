"""Entry point for LTX-2.3 quantized model inference.

Loads the vendored LTX-2.3 model architecture and runs the distilled
generation pipeline. Supports precomputed embeddings via env var.

Two-stage pipeline (when --upscale is provided):
  Stage 1: Generate at half resolution (8 steps)
  Upscale: Neural latent upsampler 2x
  Stage 2: Refine at target resolution (3 steps)

Single-stage pipeline (default):
  Generate at target resolution (8 steps)

Usage::

    python -m engine.generate_v23 --prompt "..." --model-dir /path/to/model ...

Progress is reported on stderr as::

    STAGE:1:STEP:1:8
    STAGE:1:STEP:2:8
    ...
    STAGE:2:STEP:1:3
    ...
    STATUS:Decoding video
    STATUS:Saving
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _progress(msg: str) -> None:
    """Write progress line to stderr for subprocess parsing."""
    print(msg, file=sys.stderr, flush=True)


def _report_memory(label: str) -> None:
    """Report current Metal memory stats via stderr for parent process parsing.

    Format: MEMORY:<label>:active=<GB>:cache=<GB>:peak=<GB>
    """
    active = mx.get_active_memory() / (1024**3)
    cache = mx.get_cache_memory() / (1024**3)
    peak = mx.get_peak_memory() / (1024**3)
    _progress(f"MEMORY:{label}:active={active:.3f}:cache={cache:.3f}:peak={peak:.3f}")


def _decode_preview_frame(
    latent: mx.array,
    decoder,
    preview_dir: str,
    step: int,
) -> str | None:
    """Decode a single frame from intermediate latent to JPEG for preview.

    Extracts the middle temporal frame from the latent, decodes via VAE,
    and saves as a JPEG file. Returns the file path, or None on failure.

    Args:
        latent: (B, 128, F', H', W') intermediate latent tensor.
        decoder: Loaded VideoDecoder instance.
        preview_dir: Directory to write preview JPEG files.
        step: Current diffusion step (used for filename).

    Returns:
        Path to the JPEG file, or None if decoding failed.
    """
    try:
        from PIL import Image

        # Extract middle temporal frame: (B, 128, 1, H', W')
        frame_idx = latent.shape[2] // 2
        frame_latent = latent[:, :, frame_idx : frame_idx + 1, :, :]

        # Decode via VAE — single latent frame produces ~8 pixel frames
        pixels = decoder(frame_latent)  # (B, 3, F_out, H, W)
        mx.eval(pixels)

        # Take middle output frame
        mid = pixels.shape[2] // 2
        frame_np = np.array(pixels[0, :, mid, :, :])  # (3, H, W)
        frame_np = frame_np.transpose(1, 2, 0)  # (H, W, 3)
        frame_np = np.clip((frame_np + 1) * 127.5, 0, 255).astype(np.uint8)

        img = Image.fromarray(frame_np)

        # Resize to max 512px on long side for small JPEG
        max_dim = max(img.width, img.height)
        if max_dim > 512:
            scale = 512 / max_dim
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )

        preview_path = os.path.join(preview_dir, f"preview_step{step}.jpg")
        img.save(preview_path, format="JPEG", quality=60)

        # Free intermediate tensors
        del pixels, frame_np
        gc.collect()

        return preview_path
    except Exception as e:
        log.warning("Preview frame decode failed at step %d: %s", step, e)
        return None


def _apply_loras(model, lora_specs: list[str]) -> None:
    """Parse --lora arguments and apply LoRA weight deltas to the model.

    Args:
        model: The loaded LTX model (X0Model or LTXModel).
        lora_specs: List of strings in format "/path/to/file.safetensors:strength".
    """
    from engine.lora_manager import load_lora_weights, apply_lora_to_model

    for spec in lora_specs:
        # Parse "path:strength" format
        if ":" in spec:
            # Find the last colon (strength separator) — path may contain colons on macOS
            last_colon = spec.rfind(":")
            lora_path = spec[:last_colon]
            try:
                strength = float(spec[last_colon + 1:])
            except ValueError:
                lora_path = spec
                strength = 0.7
        else:
            lora_path = spec
            strength = 0.7

        log.info("Loading LoRA: %s (strength=%.2f)", lora_path, strength)
        _progress(f"STATUS:Applying LoRA ({Path(lora_path).stem})")

        try:
            weight_deltas, metadata = load_lora_weights(lora_path, strength)
            applied = apply_lora_to_model(model, weight_deltas)
            log.info(
                "LoRA applied: %s — %d/%d layers, rank=%d",
                Path(lora_path).stem,
                applied,
                metadata["num_adapted_layers"],
                metadata["rank"],
            )
            # Free the deltas after application
            del weight_deltas
            gc.collect()
        except Exception as e:
            log.error("Failed to apply LoRA %s: %s", lora_path, e)
            _progress(f"STATUS:LoRA failed — {e}")


def _load_precomputed_embeddings(path: str) -> dict:
    """Load precomputed text embeddings from npz file."""
    data = np.load(path)
    result = {}
    for key in data.files:
        result[key] = mx.array(data[key])
    log.info("Loaded precomputed embeddings from %s: keys=%s", path, list(result.keys()))
    return result


def _load_upsampler(weights_path: str):
    """Load LatentUpsampler with manual weight assignment.

    Uses manual assignment instead of mlx tree_unflatten which breaks
    with numeric dict keys in the upsampler's res_blocks.
    """
    from mlx_video.models.ltx.upsampler import LatentUpsampler

    raw = mx.load(weights_path)

    # Detect mid_channels from weight shape
    sample = raw.get("res_blocks.0.conv1.weight")
    mid_channels = sample.shape[0] if sample is not None else 1024

    upsampler = LatentUpsampler(
        in_channels=128, mid_channels=mid_channels, num_blocks_per_stage=4
    )

    def assign(obj, key_prefix, raw_weights):
        """Assign weight and bias from raw weights to module, transposing convolutions."""
        for suffix in ("weight", "bias"):
            full_key = f"{key_prefix}.{suffix}"
            if full_key not in raw_weights:
                continue
            val = raw_weights[full_key]
            if suffix == "weight":
                if val.ndim == 5:
                    # Conv3d: PyTorch (O,I,D,H,W) -> MLX (O,D,H,W,I)
                    val = val.transpose(0, 2, 3, 4, 1)
                elif val.ndim == 4:
                    # Conv2d: PyTorch (O,I,H,W) -> MLX (O,H,W,I)
                    val = val.transpose(0, 2, 3, 1)
            setattr(obj, suffix, val)

    assign(upsampler.initial_conv, "initial_conv", raw)
    assign(upsampler.initial_norm, "initial_norm", raw)
    assign(upsampler.final_conv, "final_conv", raw)

    for i in range(4):
        assign(upsampler.res_blocks[i].conv1, f"res_blocks.{i}.conv1", raw)
        assign(upsampler.res_blocks[i].norm1, f"res_blocks.{i}.norm1", raw)
        assign(upsampler.res_blocks[i].conv2, f"res_blocks.{i}.conv2", raw)
        assign(upsampler.res_blocks[i].norm2, f"res_blocks.{i}.norm2", raw)
        assign(
            upsampler.post_upsample_res_blocks[i].conv1,
            f"post_upsample_res_blocks.{i}.conv1",
            raw,
        )
        assign(
            upsampler.post_upsample_res_blocks[i].norm1,
            f"post_upsample_res_blocks.{i}.norm1",
            raw,
        )
        assign(
            upsampler.post_upsample_res_blocks[i].conv2,
            f"post_upsample_res_blocks.{i}.conv2",
            raw,
        )
        assign(
            upsampler.post_upsample_res_blocks[i].norm2,
            f"post_upsample_res_blocks.{i}.norm2",
            raw,
        )

    # upsampler.0 in weights maps to upsampler.conv (SpatialRationalResampler)
    assign(upsampler.upsampler.conv, "upsampler.0", raw)

    log.info(
        "Loaded upsampler: mid_channels=%d, %d weights", mid_channels, len(raw)
    )
    return upsampler


def _load_latent_stats(model_dir: Path) -> tuple[mx.array, mx.array]:
    """Load per-channel latent mean/std from VAE decoder weights.

    Returns:
        (latent_mean, latent_std) each of shape (128,).
    """
    weights_path = model_dir / "vae_decoder.safetensors"
    raw = mx.load(str(weights_path))

    prefix = "vae_decoder."
    mean_key = f"{prefix}per_channel_statistics.mean"
    std_key = f"{prefix}per_channel_statistics.std"

    latent_mean = raw[mean_key]
    latent_std = raw[std_key]
    log.info("Loaded latent stats: mean=%s, std=%s", latent_mean.shape, latent_std.shape)
    return latent_mean, latent_std


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.3 video generation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument(
        "--upscale", type=str, default=None,
        help="Path to upsampler weights for two-stage pipeline (neural latent upscale)",
    )
    parser.add_argument(
        "--ffmpeg-upscale", action="store_true",
        help="Apply 2x lanczos upscale via ffmpeg as post-processing after video is saved",
    )
    parser.add_argument(
        "--preview-interval", type=int, default=0,
        help="Emit a preview frame every N diffusion steps (0 = disabled)",
    )
    parser.add_argument(
        "--no-bwe", action="store_true",
        help="Skip bandwidth extension — output audio at 16kHz instead of 48kHz. "
             "Reduces metallic artifacts from synthesized harmonics.",
    )
    parser.add_argument(
        "--lora", action="append", default=[],
        help="LoRA to apply. Format: /path/to/file.safetensors:strength "
             "(can be specified multiple times for stacking LoRAs)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load precomputed embeddings (from text encoding subprocess)
    embeddings_path = os.environ.get("LTX_PRECOMPUTED_EMBEDDINGS")
    if embeddings_path and Path(embeddings_path).exists():
        embeddings = _load_precomputed_embeddings(embeddings_path)
        prompt_embeds = embeddings.get("prompt_embeds")
        prompt_mask = embeddings.get("prompt_attention_mask")
        audio_embeds = embeddings.get("audio_prompt_embeds")
        audio_mask = embeddings.get("audio_prompt_attention_mask")
    else:
        log.error("No precomputed embeddings found. Set LTX_PRECOMPUTED_EMBEDDINGS env var.")
        sys.exit(1)

    # Load model
    _progress("STATUS:Loading model")
    from engine.ltx23_model.loader import load_ltx23_transformer

    model = load_ltx23_transformer(model_dir, low_memory=True, as_x0=True)
    log.info("Model loaded")
    _report_memory("after_model_load")

    # Apply LoRAs if specified
    if args.lora:
        _apply_loras(model, args.lora)
        _report_memory("after_lora_apply")

    # NOTE: TeaCache disabled — 0% cache hit rate with 8-step distilled model
    # (large sigma jumps between steps -> features change too much for caching)
    # NOTE: mx.compile disabled — subprocess-per-generation architecture means
    # tracing overhead is paid every time, compiled kernels lost on exit

    from engine.ltx23_model.pipeline import (
        GenerationConfig,
        STAGE_2_SIGMAS,
        generate,
    )

    if args.upscale:
        # Two-stage pipeline: half res -> upscale -> refine at target res
        _run_two_stage(args, model, model_dir, prompt_embeds, prompt_mask,
                       audio_embeds, audio_mask)
    else:
        # Single-stage pipeline: generate at target resolution
        _run_single_stage(args, model, model_dir, prompt_embeds, prompt_mask,
                          audio_embeds, audio_mask)


def _run_single_stage(
    args,
    model,
    model_dir: Path,
    prompt_embeds: mx.array,
    prompt_mask: mx.array | None,
    audio_embeds: mx.array | None,
    audio_mask: mx.array | None,
) -> None:
    """Single-stage generation at target resolution."""
    from engine.ltx23_model.pipeline import GenerationConfig, generate

    config = GenerationConfig(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=float(args.fps),
        seed=args.seed,
        num_steps=args.num_steps,
        audio_prompt_embeds=audio_embeds,
        audio_prompt_attention_mask=audio_mask,
        generate_audio=args.generate_audio and audio_embeds is not None,
        low_memory=True,
    )

    # Image conditioning
    if args.image:
        config.image_latent = _encode_image(args.image, model_dir, args.height, args.width)
        config.image_strength = args.image_strength

    # Load VAE decoder for intermediate preview frames (if enabled)
    preview_decoder = None
    preview_dir = None
    if args.preview_interval > 0:
        from engine.ltx23_model.vae_decoder import load_vae_decoder

        preview_decoder = load_vae_decoder(model_dir)
        preview_dir = tempfile.mkdtemp(prefix="ltx_preview_")
        log.info("Preview decoder loaded, interval=%d", args.preview_interval)

    def on_progress(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:1:STEP:{step}:{total}")
        if preview_decoder and preview_dir and step % args.preview_interval == 0:
            path = _decode_preview_frame(latent, preview_decoder, preview_dir, step)
            if path:
                _progress(f"PREVIEW:{path}")

    _progress("STATUS:Generating video")
    output = generate(model, config, progress_callback=on_progress)
    _report_memory("after_diffusion")

    # Free model and preview decoder before final VAE decode
    del model
    if preview_decoder is not None:
        del preview_decoder
    gc.collect()
    mx.clear_cache()

    # VAE decode
    _progress("STATUS:Decoding video")
    _decode_and_save(output, args, model_dir)

    # Optional ffmpeg 2x lanczos upscale as post-processing
    if args.ffmpeg_upscale:
        _progress("STATUS:Upscaling 2x (ffmpeg)")
        _ffmpeg_upscale_2x(args.output_path)

    # Clean up preview temp dir
    if preview_dir:
        import shutil
        shutil.rmtree(preview_dir, ignore_errors=True)

    _report_memory("final")
    _progress("STATUS:Complete")
    log.info("Generation complete: %s", args.output_path)


def _run_two_stage(
    args,
    model,
    model_dir: Path,
    prompt_embeds: mx.array,
    prompt_mask: mx.array | None,
    audio_embeds: mx.array | None,
    audio_mask: mx.array | None,
) -> None:
    """Two-stage pipeline: half res -> neural upscale -> refine at target res."""
    from engine.ltx23_model.pipeline import GenerationConfig, STAGE_2_SIGMAS, generate
    from mlx_video.models.ltx.upsampler import upsample_latents

    half_height = args.height // 2
    half_width = args.width // 2
    log.info(
        "Two-stage pipeline: Stage 1 at %dx%d, Stage 2 at %dx%d",
        half_width, half_height, args.width, args.height,
    )

    # --- Image conditioning for both stages ---
    stage1_image_latent = None
    stage2_image_latent = None
    if args.image:
        # Stage 1: encode at half resolution
        stage1_image_latent = _encode_image(args.image, model_dir, half_height, half_width)
        # Stage 2: encode at target resolution
        stage2_image_latent = _encode_image(args.image, model_dir, args.height, args.width)

    # --- Stage 1: Generate at half resolution ---
    _progress("STATUS:Stage 1 - generating at half resolution")
    log.info("Stage 1: %dx%d, %d steps", half_width, half_height, args.num_steps)

    config_s1 = GenerationConfig(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        height=half_height,
        width=half_width,
        num_frames=args.num_frames,
        fps=float(args.fps),
        seed=args.seed,
        num_steps=args.num_steps,
        audio_prompt_embeds=audio_embeds,
        audio_prompt_attention_mask=audio_mask,
        generate_audio=args.generate_audio and audio_embeds is not None,
        low_memory=True,
    )

    if stage1_image_latent is not None:
        config_s1.image_latent = stage1_image_latent
        config_s1.image_strength = args.image_strength

    # Load VAE decoder for intermediate preview frames (if enabled)
    preview_decoder = None
    preview_dir = None
    if args.preview_interval > 0:
        from engine.ltx23_model.vae_decoder import load_vae_decoder

        preview_decoder = load_vae_decoder(model_dir)
        preview_dir = tempfile.mkdtemp(prefix="ltx_preview_")
        log.info("Preview decoder loaded for two-stage, interval=%d", args.preview_interval)

    def on_progress_s1(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:1:STEP:{step}:{total}")
        if preview_decoder and preview_dir and step % args.preview_interval == 0:
            path = _decode_preview_frame(latent, preview_decoder, preview_dir, step)
            if path:
                _progress(f"PREVIEW:{path}")

    output_s1 = generate(model, config_s1, progress_callback=on_progress_s1)
    _report_memory("after_diffusion")

    video_latent = output_s1.video_latent
    audio_latent = output_s1.audio_latent
    mx.eval(video_latent)
    if audio_latent is not None:
        mx.eval(audio_latent)

    log.info("Stage 1 complete. Video latent: %s", video_latent.shape)

    # Unload transformer and preview decoder before upscale — frees ~21GB
    del model
    del output_s1
    if preview_decoder is not None:
        del preview_decoder
        preview_decoder = None
    gc.collect()
    mx.clear_cache()
    log.info("Transformer unloaded for upscale phase")

    # --- Upscale latent 2x with neural upsampler ---
    _progress("STATUS:Upscaling latent 2x")
    log.info("Loading upsampler from %s", args.upscale)

    upsampler = _load_upsampler(args.upscale)
    mx.eval(upsampler.parameters())

    # Load latent statistics for un-normalization / re-normalization
    latent_mean, latent_std = _load_latent_stats(model_dir)

    video_latent = upsample_latents(video_latent, upsampler, latent_mean, latent_std)
    mx.eval(video_latent)
    log.info("Upscaled video latent: %s", video_latent.shape)

    # Free upsampler immediately
    del upsampler, latent_mean, latent_std
    gc.collect()
    mx.clear_cache()

    # --- Stage 2: Reload transformer and refine at target resolution ---
    _progress("STATUS:Reloading model for Stage 2")
    from engine.ltx23_model.loader import load_ltx23_transformer

    model = load_ltx23_transformer(model_dir, low_memory=True, as_x0=True)
    log.info("Transformer reloaded for Stage 2")

    # Re-apply LoRAs after model reload
    if args.lora:
        _apply_loras(model, args.lora)

    num_stage2_steps = len(STAGE_2_SIGMAS) - 1
    _progress("STATUS:Stage 2 - refining at target resolution")
    log.info("Stage 2: %dx%d, %d steps, sigmas=%s",
             args.width, args.height, num_stage2_steps, STAGE_2_SIGMAS)

    config_s2 = GenerationConfig(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=float(args.fps),
        seed=args.seed,
        num_steps=num_stage2_steps,
        sigmas=STAGE_2_SIGMAS,
        initial_latent=video_latent,
        initial_audio_latent=audio_latent,
        audio_prompt_embeds=audio_embeds,
        audio_prompt_attention_mask=audio_mask,
        generate_audio=args.generate_audio and audio_embeds is not None,
        low_memory=True,
    )

    if stage2_image_latent is not None:
        config_s2.image_latent = stage2_image_latent
        config_s2.image_strength = args.image_strength

    # Reload preview decoder for Stage 2 previews (if enabled)
    if args.preview_interval > 0 and preview_dir:
        from engine.ltx23_model.vae_decoder import load_vae_decoder

        preview_decoder = load_vae_decoder(model_dir)
        log.info("Preview decoder reloaded for Stage 2")

    def on_progress_s2(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:2:STEP:{step}:{total}")
        if preview_decoder and preview_dir and step % args.preview_interval == 0:
            path = _decode_preview_frame(latent, preview_decoder, preview_dir, step)
            if path:
                _progress(f"PREVIEW:{path}")

    output_s2 = generate(model, config_s2, progress_callback=on_progress_s2)

    # Free model and preview decoder before VAE decode
    del model, video_latent, audio_latent
    if preview_decoder is not None:
        del preview_decoder
    gc.collect()
    mx.clear_cache()

    # VAE decode at target resolution
    _progress("STATUS:Decoding video")
    _decode_and_save(output_s2, args, model_dir)

    # Optional ffmpeg 2x lanczos upscale as post-processing
    if args.ffmpeg_upscale:
        _progress("STATUS:Upscaling 2x (ffmpeg)")
        _ffmpeg_upscale_2x(args.output_path)

    # Clean up preview temp dir
    if preview_dir:
        import shutil
        shutil.rmtree(preview_dir, ignore_errors=True)

    _report_memory("final")
    _progress("STATUS:Complete")
    log.info("Two-stage generation complete: %s", args.output_path)


def _encode_image(image_path: str, model_dir: Path, height: int, width: int) -> mx.array:
    """Encode a reference image using the mlx_video VAE encoder.

    Uses the reference mlx_video VideoEncoder implementation directly
    for correct Conv3d operations (native mx.conv3d vs our custom slice-based).

    Args:
        image_path: Path to the source image file.
        model_dir: Path to the model directory.
        height: Target video height (must be divisible by 32).
        width: Target video width (must be divisible by 32).

    Returns:
        (1, 128, 1, H/32, W/32) normalized latent tensor.
    """
    import json
    from PIL import Image
    from mlx_video.models.ltx.video_vae.encoder import VideoEncoder
    from mlx_video.models.ltx.video_vae.video_vae import NormLayerType, LogVarianceType
    from mlx_video.utils import prepare_image_for_encoding

    log.info("Encoding reference image: %s at %dx%d", image_path, width, height)

    # Load encoder config
    config_path = model_dir / "embedded_config.json"
    encoder_blocks = []
    patch_size = 4
    if config_path.exists():
        with open(config_path) as f:
            vae_cfg = json.load(f).get("vae", {})
        encoder_blocks = [(b[0], b[1]) for b in vae_cfg.get("encoder_blocks", [])]
        patch_size = vae_cfg.get("patch_size", 4)
        log.info("Using encoder config from embedded_config.json")

    # Create reference encoder
    encoder = VideoEncoder(
        encoder_blocks=encoder_blocks,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.UNIFORM,
        patch_size=patch_size,
    )

    # Load weights (manual prefix stripping to avoid tree_unflatten numeric key bug)
    weights_path = model_dir / "vae_encoder.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"vae_encoder.safetensors not found in {model_dir}")

    raw = mx.load(str(weights_path))
    prefix = "vae_encoder."
    encoder.load_weights(
        [(k[len(prefix):], v) for k, v in raw.items() if k.startswith(prefix)],
        strict=False,
    )
    # Load per-channel normalization statistics
    mean_key = f"{prefix}per_channel_statistics._mean_of_means"
    std_key = f"{prefix}per_channel_statistics._std_of_means"
    if mean_key in raw:
        encoder.latent_mean = raw[mean_key]
        encoder.latent_std = raw[std_key]

    mx.eval(encoder.parameters())  # noqa: S307
    log.info("VAE encoder loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    img_arr = mx.array(np.array(img).astype(np.float32) / 255.0)
    img_tensor = prepare_image_for_encoding(img_arr, height, width)

    # Encode
    latent = encoder(img_tensor)
    mx.eval(latent)  # noqa: S307
    log.info("Image encoded to latent: %s", latent.shape)

    # Free encoder
    del encoder, raw
    gc.collect()
    mx.clear_cache()

    return latent


def _ffmpeg_upscale_2x(video_path: str) -> None:
    """Upscale video 2x using ffmpeg lanczos filter (in-place)."""
    import subprocess

    ffmpeg_bin = _find_ffmpeg()
    tmp_path = video_path + ".tmp.mp4"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vf", "scale=iw*2:ih*2:flags=lanczos",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        tmp_path,
    ]
    log.info("Upscaling 2x with ffmpeg lanczos: %s", video_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg upscale failed: %s", result.stderr[-500:])
        return

    # Replace original with upscaled version
    os.replace(tmp_path, video_path)
    log.info("Upscaled video saved to %s", video_path)


def _decode_and_save(output, args, model_dir: Path) -> None:
    """Decode latents with VAE and stream to MP4 via ffmpeg."""
    video_latent = output.video_latent
    log.info("Video latent shape: %s", video_latent.shape)

    # Decode audio if present
    audio_path = None
    audio_latent = getattr(output, "audio_latent", None)
    if audio_latent is not None and args.generate_audio:
        _progress("STATUS:Decoding audio")
        audio_path = _decode_audio(audio_latent, model_dir, args.output_path)
        gc.collect()
        mx.clear_cache()

    # Load VAE decoder
    from engine.ltx23_model.vae_decoder import load_vae_decoder, streaming_decode_to_ffmpeg

    decoder = load_vae_decoder(model_dir)

    def on_frame(frame_idx: int, total: int) -> None:
        if frame_idx % 10 == 0 or frame_idx == total:
            log.info("Decoded frame %d/%d", frame_idx, total)

    _progress("STATUS:Saving")

    # If we have audio, stream video without audio first, then mux
    video_only_path = args.output_path
    if audio_path:
        video_only_path = args.output_path.replace(".mp4", "_video_only.mp4")

    streaming_decode_to_ffmpeg(
        latent=video_latent,
        decoder=decoder,
        output_path=video_only_path,
        fps=args.fps,
        progress_fn=on_frame,
    )

    # Free decoder
    del decoder
    gc.collect()
    mx.clear_cache()

    # Mux video + audio if both present
    if audio_path and os.path.exists(audio_path):
        _mux_video_audio(video_only_path, audio_path, args.output_path)
        # Clean up intermediate files
        try:
            os.remove(video_only_path)
            os.remove(audio_path)
        except OSError:
            pass


def _decode_audio(
    audio_latent: mx.array, model_dir: Path, output_path: str,
) -> str | None:
    """Decode audio latent to WAV via audio VAE decoder + BigVGAN v2 vocoder.

    Args:
        audio_latent: Audio latent tensor (B, 8, T, 16) from the diffusion pipeline.
        model_dir: Path to model weights directory.
        output_path: Base path for the output (used to derive WAV filename).

    Returns:
        Path to WAV file, or None on failure.
    """
    import wave as wave_mod
    from mlx_video.generate_av import AUDIO_LATENT_CHANNELS
    from mlx_video.models.ltx.audio_vae import AudioDecoder, CausalityAxis, NormType

    log.info("Audio latent shape: %s", audio_latent.shape)

    # Reshape from pipeline format (B, 128, T) -> audio VAE format (B, 8, T, 16)
    if audio_latent.ndim == 3:
        B, C, T = audio_latent.shape
        audio_latent = audio_latent.reshape(B, 8, 16, T).transpose(0, 1, 3, 2)
        log.info("Reshaped audio latent to: %s", audio_latent.shape)

    # --- Audio VAE decoder (mlx_video reference) ---
    audio_decoder = AudioDecoder(
        ch=128, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=2,
        attn_resolutions={8, 16, 32}, resolution=256,
        z_channels=AUDIO_LATENT_CHANNELS, norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT, mel_bins=64,
    )
    raw = mx.load(str(model_dir / "audio_vae.safetensors"))
    # Our split audio_vae.safetensors has keys like "audio_vae.conv_in.conv.weight"
    # already in MLX format (out, H, W, in). sanitize_audio_vae_weights expects
    # "audio_vae.decoder.*" prefix (original PyTorch format) and returns empty dict
    # for our split format. Load directly with prefix stripping instead.
    weights = []
    for k, v in raw.items():
        if k.startswith("audio_vae."):
            clean = k[len("audio_vae."):]
            weights.append((clean, v))
    audio_decoder.load_weights(weights, strict=False)
    log.info("Audio VAE: loaded %d weights", len(weights))
    mx.eval(audio_decoder.parameters())  # noqa: S307
    log.info("Audio VAE decoder loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    mel = audio_decoder(audio_latent)
    mx.eval(mel)  # noqa: S307
    log.info(
        "Mel spectrogram shape: %s, range: [%.3f, %.3f]",
        mel.shape, float(mel.min()), float(mel.max()),
    )

    del audio_decoder, raw
    gc.collect()
    mx.clear_cache()

    # --- Vocoder (BigVGAN v2 — ltx-core reference port) ---
    from engine.ltx23_model.vocoder import load_vocoder
    vocoder = load_vocoder(model_dir)
    waveform = vocoder(mel)
    mx.eval(waveform)  # noqa: S307
    sample_rate = 16000
    log.info("Waveform shape: %s, sample_rate: %d", waveform.shape, sample_rate)

    del vocoder, mel
    gc.collect()
    mx.clear_cache()

    # Convert to numpy: (B, 2, T) → (T, 2)
    wav_np = np.array(waveform[0])
    if wav_np.shape[0] == 2:
        wav_np = wav_np.T
    del waveform

    # Normalize to audible level
    peak = np.max(np.abs(wav_np))
    if peak > 1e-8:
        wav_np = wav_np * (0.95 / peak)
    wav_np = np.clip(wav_np, -1.0, 1.0)

    # Save WAV
    wav_path = output_path.replace(".mp4", "_audio.wav")
    audio_int16 = (wav_np * 32767).astype(np.int16)
    nchannels = 2 if wav_np.ndim == 2 and wav_np.shape[1] == 2 else 1
    with wave_mod.open(wav_path, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    log.info("Audio saved to %s", wav_path)
    return wav_path


def _find_ffmpeg() -> str:
    """Find ffmpeg binary, checking common Homebrew paths."""
    import shutil

    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("ffmpeg not found. Install with: brew install ffmpeg")


def _mux_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux video and audio streams into final MP4 via ffmpeg."""
    import subprocess

    ffmpeg_bin = _find_ffmpeg()
    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    log.info("Muxing: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg mux failed: %s", result.stderr)
    else:
        log.info("Final output: %s", output_path)


if __name__ == "__main__":
    main()
