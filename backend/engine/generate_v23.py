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

    python -m engine.generate_v23 --prompt "..." --model-dir /path/to/ltx23-mlx ...

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
import gc
import logging
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _progress(msg: str) -> None:
    """Write progress line to stderr for subprocess parsing."""
    print(msg, file=sys.stderr, flush=True)


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
    parser.add_argument("--image-strength", type=float, default=0.85)
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument(
        "--upscale", type=str, default=None,
        help="Path to upsampler weights for two-stage pipeline (neural latent upscale)",
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

    def on_progress(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:1:STEP:{step}:{total}")

    _progress("STATUS:Generating video")
    output = generate(model, config, progress_callback=on_progress)

    # Free model before VAE decode
    del model
    gc.collect()
    mx.clear_cache()

    # VAE decode
    _progress("STATUS:Decoding video")
    _decode_and_save(output, args, model_dir)

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

    def on_progress_s1(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:1:STEP:{step}:{total}")

    output_s1 = generate(model, config_s1, progress_callback=on_progress_s1)

    video_latent = output_s1.video_latent
    audio_latent = output_s1.audio_latent
    mx.eval(video_latent)
    if audio_latent is not None:
        mx.eval(audio_latent)

    log.info("Stage 1 complete. Video latent: %s", video_latent.shape)

    # Unload transformer before upscale — frees ~21GB for upsampler activations
    del model
    del output_s1
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

    def on_progress_s2(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:2:STEP:{step}:{total}")

    output_s2 = generate(model, config_s2, progress_callback=on_progress_s2)

    # Free model before VAE decode
    del model, video_latent, audio_latent
    gc.collect()
    mx.clear_cache()

    # VAE decode at target resolution
    _progress("STATUS:Decoding video")
    _decode_and_save(output_s2, args, model_dir)

    _progress("STATUS:Complete")
    log.info("Two-stage generation complete: %s", args.output_path)


def _encode_image(image_path: str, model_dir: Path, height: int, width: int) -> mx.array:
    """Encode a reference image using VAE encoder.

    Resizes the image to match the target generation resolution, then
    encodes it to the latent space via the VAE encoder.

    Args:
        image_path: Path to the source image file.
        model_dir: Path to the model directory.
        height: Target video height (must be divisible by 32).
        width: Target video width (must be divisible by 32).

    Returns:
        (1, 128, 1, H/32, W/32) normalized latent tensor.
    """
    from PIL import Image

    log.info("Encoding reference image: %s at %dx%d", image_path, width, height)

    # Load and resize to target resolution
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)

    # Normalize to [-1, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array * 2.0 - 1.0
    # [H, W, 3] -> [1, 3, 1, H, W]
    img_tensor = mx.array(img_array).transpose(2, 0, 1)[None, :, None, :, :]

    # Load VAE encoder
    from engine.ltx23_model.vae_encoder import load_vae_encoder, encode_image

    encoder = load_vae_encoder(model_dir)
    latent = encode_image(img_tensor, encoder)
    log.info("Image encoded to latent: %s", latent.shape)

    # Free encoder
    del encoder
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


def _decode_audio(audio_latent: mx.array, model_dir: Path, output_path: str) -> str | None:
    """Decode audio latent to WAV via audio VAE + vocoder.

    Returns path to WAV file, or None on failure.
    """
    import soundfile as sf

    log.info("Audio latent shape: %s", audio_latent.shape)

    # Reshape from pipeline format (B, 128, T) -> audio VAE format (B, 8, T, 16)
    # 128 channels = 8 groups x 16 features. Must split channels first, then transpose.
    if audio_latent.ndim == 3:
        B, C, T = audio_latent.shape
        audio_latent = audio_latent.reshape(B, 8, 16, T).transpose(0, 1, 3, 2)
        log.info("Reshaped audio latent to: %s", audio_latent.shape)

    # Audio VAE: latent (B, 8, T, 16) -> mel (B, 2, T', 64)
    from engine.ltx23_model.audio_decoder import load_audio_decoder

    audio_decoder = load_audio_decoder(model_dir)
    mel_spec = audio_decoder(audio_latent)
    log.info("Mel spectrogram shape: %s", mel_spec.shape)

    del audio_decoder
    gc.collect()
    mx.clear_cache()

    # Vocoder: mel -> waveform at 48 kHz
    from engine.ltx23_model.vocoder import load_vocoder

    vocoder = load_vocoder(model_dir)
    waveform = vocoder(mel_spec)
    log.info("Waveform shape: %s, sample_rate: %d", waveform.shape, vocoder.output_sample_rate)

    sample_rate = vocoder.output_sample_rate

    del vocoder, mel_spec
    gc.collect()
    mx.clear_cache()

    # Save as WAV: (B, 2, T) -> (T, 2)
    audio_np = np.array(waveform[0].transpose(1, 0))
    wav_path = output_path.replace(".mp4", "_audio.wav")
    sf.write(wav_path, audio_np, sample_rate)
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
