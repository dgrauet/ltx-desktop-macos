"""Distilled generation pipeline for LTX-2.3 on MLX.

Implements the single-stage distilled denoising loop with Euler stepping.
Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .model import LTXModel, Modality, X0Model
from .patchifier import (
    SpatioTemporalScaleFactors,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)

log = logging.getLogger(__name__)

# Predefined sigmas for distilled pipeline (8 steps)
DISTILLED_SIGMAS = [1.0, 0.94, 0.82, 0.64, 0.44, 0.27, 0.13, 0.03]


def aggressive_cleanup() -> None:
    """Force-free Metal memory."""
    gc.collect()
    mx.clear_cache()
    # Barrier: wait for all pending GPU ops
    force_materialize = mx.eval
    force_materialize(mx.zeros(1))


@dataclass
class GenerationConfig:
    """Configuration for video generation."""
    prompt_embeds: mx.array  # Shape: (1, seq_len, dim)
    prompt_attention_mask: Optional[mx.array] = None  # Shape: (1, seq_len)
    height: int = 512
    width: int = 768
    num_frames: int = 97
    fps: float = 24.0
    seed: int = 42
    num_steps: int = 8
    sigmas: Optional[List[float]] = None
    guidance_scale: float = 1.0  # Not used for distilled (no CFG)

    # Audio (optional)
    audio_prompt_embeds: Optional[mx.array] = None
    audio_prompt_attention_mask: Optional[mx.array] = None
    generate_audio: bool = False

    # Image conditioning (optional)
    image_latent: Optional[mx.array] = None  # Encoded reference image
    image_frame_idx: int = 0
    image_strength: float = 0.85

    # Memory
    low_memory: bool = True


@dataclass
class GenerationOutput:
    """Output from the generation pipeline."""
    video_latent: mx.array  # Shape: [B, C, F, H, W] in latent space
    audio_latent: Optional[mx.array] = None  # Shape: [B, C, T] in latent space


def _create_denoise_mask(
    shape: VideoLatentShape,
    image_latent: Optional[mx.array] = None,
    image_frame_idx: int = 0,
    image_strength: float = 0.85,
) -> mx.array:
    """Create per-token denoising mask.

    1.0 = full denoise (generate from noise)
    (1.0 - strength) = partial denoise (conditioned frame)
    """
    mask = mx.ones(shape.mask_shape().to_tuple())

    if image_latent is not None:
        # Set conditioned frame to reduced denoising
        mask_value = 1.0 - image_strength
        # mask shape: (B, 1, F, H, W) — set frame at image_frame_idx
        mask = np.array(mask)
        mask[:, :, image_frame_idx, :, :] = mask_value
        mask = mx.array(mask)

    return mask


def _euler_step(
    latent: mx.array,
    x0: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """Euler diffusion step: compute next latent from denoised prediction.

    velocity = (latent - x0) / sigma
    latent_next = latent + (sigma_next - sigma) * velocity
    """
    velocity = (latent - x0) / sigma
    return latent + (sigma_next - sigma) * velocity


def generate(
    model: X0Model | LTXModel,
    config: GenerationConfig,
    progress_callback: Optional[callable] = None,
) -> GenerationOutput:
    """Run distilled generation pipeline.

    Args:
        model: LTX-2.3 model (X0Model wraps velocity model for denoised output).
        config: Generation configuration.
        progress_callback: Optional callback(step, total_steps, latent) for progress.

    Returns:
        GenerationOutput with video (and optionally audio) latents.
    """
    is_x0 = isinstance(model, X0Model)
    velocity_model = model.velocity_model if is_x0 else model

    scale_factors = SpatioTemporalScaleFactors()
    patchifier = VideoLatentPatchifier(patch_size=1)

    # Compute latent shape from pixel dimensions
    latent_shape = VideoLatentShape.from_pixel_shape(
        batch=1,
        channels=velocity_model.inner_dim // 1,  # 128 for LTX-2.3
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        scale_factors=scale_factors,
    )
    # Actually channels is in_channels (128), not inner_dim
    latent_shape = VideoLatentShape(
        batch=1,
        channels=128,  # LTX-2.3 VAE latent channels
        frames=latent_shape.frames,
        height=latent_shape.height,
        width=latent_shape.width,
    )

    log.info(
        f"Latent shape: {latent_shape.to_tuple()} "
        f"(from {config.height}×{config.width}, {config.num_frames} frames)"
    )

    # Initialize noise
    mx.random.seed(config.seed)
    latent = mx.random.normal(latent_shape.to_tuple())

    # Create denoise mask
    denoise_mask = _create_denoise_mask(
        latent_shape,
        image_latent=config.image_latent,
        image_frame_idx=config.image_frame_idx,
        image_strength=config.image_strength,
    )

    # Apply image conditioning if provided
    clean_latent = mx.zeros_like(latent)
    if config.image_latent is not None:
        # Insert encoded image at the specified frame
        clean_np = np.array(clean_latent)
        img_np = np.array(config.image_latent)
        clean_np[:, :, config.image_frame_idx, :, :] = img_np[:, :, 0, :, :]
        clean_latent = mx.array(clean_np)
        # Mix: latent = clean * (1 - mask) + noise * mask (at sigma=1)
        latent = clean_latent * (1.0 - denoise_mask) + latent * denoise_mask

    # Compute positions
    grid_bounds = patchifier.get_patch_grid_bounds(latent_shape)
    pixel_coords = get_pixel_coords(grid_bounds, scale_factors, causal_fix=True)
    # Convert temporal axis to seconds
    pixel_coords_np = np.array(pixel_coords)
    pixel_coords_np[:, 0, :, :] = pixel_coords_np[:, 0, :, :] / config.fps
    positions = mx.array(pixel_coords_np)

    # Sigmas
    sigmas = config.sigmas or DISTILLED_SIGMAS
    if len(sigmas) > config.num_steps:
        sigmas = sigmas[: config.num_steps]

    # Audio setup
    audio_latent = None
    audio_positions = None
    if config.generate_audio and config.audio_prompt_embeds is not None:
        # Compute audio latent shape (1D temporal)
        audio_duration_s = (config.num_frames - 1) / config.fps
        audio_sr = 44100  # LTX-2.3 audio sample rate
        audio_latent_len = int(audio_duration_s * audio_sr / 2048)  # Audio VAE compression
        audio_latent = mx.random.normal((1, 128, audio_latent_len))

        # Audio positions: temporal only
        audio_bounds = np.zeros((1, 1, audio_latent_len, 2), dtype=np.float32)
        for i in range(audio_latent_len):
            audio_bounds[0, 0, i, 0] = i * 2048 / audio_sr
            audio_bounds[0, 0, i, 1] = (i + 1) * 2048 / audio_sr
        audio_positions = mx.array(audio_bounds)

    log.info(f"Starting denoising: {len(sigmas)} steps, sigmas={sigmas}")

    # Denoising loop
    force_materialize = mx.eval
    for step_idx in range(len(sigmas)):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else 0.0

        # Compute per-token timesteps: sigma * denoise_mask
        timesteps_5d = denoise_mask * sigma
        # Patchify everything
        latent_patches = patchifier.patchify(latent)
        timestep_patches = patchifier.patchify(timesteps_5d).squeeze(-1)  # [B, T]

        # Create video modality
        video_mod = Modality(
            latent=latent_patches,
            context=config.prompt_embeds,
            context_mask=config.prompt_attention_mask,
            timesteps=timestep_patches,
            positions=positions,
            enabled=True,
            sigma=mx.array([sigma]),
        )

        # Create audio modality (if generating audio)
        audio_mod = None
        if audio_latent is not None and config.audio_prompt_embeds is not None:
            audio_patches = audio_latent.transpose(0, 2, 1)  # [B, T, C]
            audio_timesteps = mx.full((1, audio_patches.shape[1]), sigma)
            audio_mod = Modality(
                latent=audio_patches,
                context=config.audio_prompt_embeds,
                context_mask=config.audio_prompt_attention_mask,
                timesteps=audio_timesteps,
                positions=audio_positions,
                enabled=True,
                sigma=mx.array([sigma]),
            )

        # Model forward pass
        if is_x0:
            output = model(video=video_mod, audio=audio_mod)
        else:
            output = model(video=video_mod, audio=audio_mod)

        # Extract denoised prediction
        if isinstance(output, tuple):
            x0_patches, audio_x0_patches = output
        else:
            x0_patches = output
            audio_x0_patches = None

        # Unpatchify back to 5D
        x0 = patchifier.unpatchify(x0_patches, latent_shape)

        # Euler step
        if sigma_next > 0:
            latent = _euler_step(latent, x0, sigma, sigma_next)
        else:
            latent = x0

        # Re-apply conditioning (keep conditioned frame clean)
        if config.image_latent is not None:
            latent = clean_latent * (1.0 - denoise_mask) + latent * denoise_mask

        # Audio euler step
        if audio_x0_patches is not None:
            audio_x0 = audio_x0_patches.transpose(0, 2, 1)  # [B, C, T]
            if sigma_next > 0:
                audio_latent = _euler_step(audio_latent, audio_x0, sigma, sigma_next)
            else:
                audio_latent = audio_x0

        # Force materialization for memory management
        force_materialize(latent)
        if audio_latent is not None:
            force_materialize(audio_latent)

        # Progress callback
        if progress_callback is not None:
            progress_callback(step_idx + 1, len(sigmas), latent)

        log.info(f"Step {step_idx + 1}/{len(sigmas)}: sigma={sigma:.3f} → {sigma_next:.3f}")

    aggressive_cleanup()

    return GenerationOutput(
        video_latent=latent,
        audio_latent=audio_latent,
    )
