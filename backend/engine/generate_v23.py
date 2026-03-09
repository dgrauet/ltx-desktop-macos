"""Entry point for LTX-2.3 quantized model inference.

Loads the vendored LTX-2.3 model architecture and runs the distilled
generation pipeline. Supports precomputed embeddings via env var.

Usage::

    python -m engine.generate_v23 --prompt "..." --model-dir /path/to/ltx23-mlx ...

Progress is reported on stderr as::

    STAGE:1:STEP:1:8
    STAGE:1:STEP:2:8
    ...
    STATUS:Decoding video
    STATUS:Saving
"""

from __future__ import annotations

import argparse
import gc
import json
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

    # Configure generation
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

    # Progress callback
    def on_progress(step: int, total: int, latent: mx.array) -> None:
        _progress(f"STAGE:1:STEP:{step}:{total}")

    # Run generation
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

    log.info("Encoding reference image: %s", image_path)

    # Load and resize to target resolution
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)

    # Normalize to [-1, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array * 2.0 - 1.0
    # [H, W, 3] → [1, 3, 1, H, W]
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

    # Audio VAE: latent (B, 8, T, 16) → mel (B, 2, T', 64)
    from engine.ltx23_model.audio_decoder import load_audio_decoder

    audio_decoder = load_audio_decoder(model_dir)
    mel_spec = audio_decoder(audio_latent)
    log.info("Mel spectrogram shape: %s", mel_spec.shape)

    del audio_decoder
    gc.collect()
    mx.clear_cache()

    # Vocoder: mel → waveform at 48 kHz
    from engine.ltx23_model.vocoder import load_vocoder

    vocoder = load_vocoder(model_dir)
    waveform = vocoder(mel_spec)
    log.info("Waveform shape: %s, sample_rate: %d", waveform.shape, vocoder.output_sample_rate)

    sample_rate = vocoder.output_sample_rate

    del vocoder, mel_spec
    gc.collect()
    mx.clear_cache()

    # Save as WAV: (B, 2, T) → (T, 2)
    audio_np = np.array(waveform[0].transpose(1, 0))
    wav_path = output_path.replace(".mp4", "_audio.wav")
    sf.write(wav_path, audio_np, sample_rate)
    log.info("Audio saved to %s", wav_path)

    return wav_path


def _mux_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux video and audio streams into final MP4 via ffmpeg."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
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
