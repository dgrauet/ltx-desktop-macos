"""Distilled generation pipeline for LTX-2.3 on MLX.

Implements the single-stage distilled denoising loop with Euler stepping.
Uses mlx_video.conditioning.latent for I2V conditioning (reference implementation).
Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_video.conditioning.latent import (
    VideoConditionByLatentIndex,
    LatentState,
    create_initial_state,
    apply_conditioning,
    apply_denoise_mask,
    add_noise_with_state,
)

from .model import LTXModel, Modality, X0Model
from .patchifier import (
    SpatioTemporalScaleFactors,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)

log = logging.getLogger(__name__)

# Predefined sigmas for distilled pipeline (8 steps + final 0.0)
# From reference mlx_video implementation: fine steps early (detail preservation),
# aggressive denoising in mid-range.
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# Stage 2 sigmas for two-stage upscale refinement (3 steps).
# Matches reference mlx_video implementation.
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


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
    image_strength: float = 1.0  # 1.0 = keep conditioned frame clean (no denoising)

    # Two-stage refinement: upscaled latent from Stage 1 to refine in Stage 2
    initial_latent: Optional[mx.array] = None
    # Audio latent from Stage 1 (partially noised for Stage 2)
    initial_audio_latent: Optional[mx.array] = None

    # Retake: regenerate a temporal segment while preserving the rest
    retake_clean_latent: Optional[mx.array] = None  # Full video encoded to latent
    retake_denoise_mask: Optional[mx.array] = None  # (1, 1, F', 1, 1): 1.0=regenerate, 0.0=preserve

    # Memory
    low_memory: bool = True


@dataclass
class GenerationOutput:
    """Output from the generation pipeline."""
    video_latent: mx.array  # Shape: [B, C, F, H, W] in latent space
    audio_latent: Optional[mx.array] = None  # Shape: [B, 8, T, 16] in latent space


def generate(
    model: X0Model | LTXModel,
    config: GenerationConfig,
    progress_callback: Optional[callable] = None,
) -> GenerationOutput:
    """Run distilled generation pipeline.

    Uses mlx_video.conditioning.latent for I2V conditioning, matching the
    reference mlx_video implementation exactly.

    Args:
        model: LTX-2.3 model (X0Model wraps velocity model for denoised output).
        config: Generation configuration.
        progress_callback: Optional callback(step, total_steps, latent) for progress.

    Returns:
        GenerationOutput with video (and optionally audio) latents.
    """
    is_x0 = isinstance(model, X0Model)
    velocity_model = model.velocity_model if is_x0 else model
    dtype = mx.bfloat16

    scale_factors = SpatioTemporalScaleFactors()
    patchifier = VideoLatentPatchifier(patch_size=1)

    # Compute latent shape from pixel dimensions
    latent_shape = VideoLatentShape.from_pixel_shape(
        batch=1,
        channels=velocity_model.inner_dim // 1,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        scale_factors=scale_factors,
    )
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

    # Sigmas
    sigmas = config.sigmas or DISTILLED_SIGMAS

    # --- Initialize video latent state using mlx_video functions ---
    if config.retake_clean_latent is not None and config.retake_denoise_mask is not None:
        # Retake: start from clean latent with noise added only in the retake region
        mx.random.seed(config.seed)
        clean = config.retake_clean_latent
        mask = config.retake_denoise_mask  # (1, 1, F', 1, 1): 1.0=regen, 0.0=preserve

        video_state = LatentState(
            latent=clean,  # Will be noised below
            clean_latent=clean,
            denoise_mask=mask,
        )
        # Add noise only to the masked region via add_noise_with_state
        video_state = add_noise_with_state(video_state, noise_scale=sigmas[0])
        log.info("Retake: initialized with mask, noise_scale=%.4f", sigmas[0])

    elif config.initial_latent is not None:
        # Stage 2: start from upscaled Stage 1 latent
        video_state = LatentState(
            latent=config.initial_latent,
            clean_latent=mx.zeros_like(config.initial_latent),
            denoise_mask=mx.ones((1, 1, latent_shape.frames, 1, 1)),
        )
        # Apply I2V conditioning if present
        if config.image_latent is not None:
            video_state = apply_conditioning(video_state, [
                VideoConditionByLatentIndex(
                    latent=config.image_latent,
                    frame_idx=config.image_frame_idx,
                    strength=config.image_strength,
                ),
            ])
        # Add noise for Stage 2 refinement
        video_state = add_noise_with_state(video_state, noise_scale=sigmas[0])
    else:
        # Stage 1: start from pure noise
        mx.random.seed(config.seed)
        video_state = create_initial_state(
            shape=latent_shape.to_tuple(),
            noise_scale=1.0,
        )
        # Apply I2V conditioning if present
        if config.image_latent is not None:
            video_state = apply_conditioning(video_state, [
                VideoConditionByLatentIndex(
                    latent=config.image_latent,
                    frame_idx=config.image_frame_idx,
                    strength=config.image_strength,
                ),
            ])

    latent = video_state.latent

    # Compute positions
    grid_bounds = patchifier.get_patch_grid_bounds(latent_shape)
    pixel_coords = get_pixel_coords(grid_bounds, scale_factors, causal_fix=True)
    pixel_coords_np = np.array(pixel_coords)
    pixel_coords_np[:, 0, :, :] = pixel_coords_np[:, 0, :, :] / config.fps
    positions = mx.array(pixel_coords_np)

    # --- Audio setup ---
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_HOP_LENGTH = 160
    AUDIO_DOWNSAMPLE_FACTOR = 4
    AUDIO_LATENT_CHANNELS = 8
    AUDIO_MEL_BINS = 16
    AUDIO_LATENTS_PER_SECOND = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_DOWNSAMPLE_FACTOR

    audio_latent = None
    audio_positions = None
    if config.generate_audio and config.audio_prompt_embeds is not None:
        audio_duration_s = config.num_frames / config.fps
        audio_latent_len = round(audio_duration_s * AUDIO_LATENTS_PER_SECOND)

        if config.initial_audio_latent is not None:
            sigma_start = sigmas[0]
            audio_noise = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_latent_len, AUDIO_MEL_BINS))
            audio_latent = config.initial_audio_latent * (1.0 - sigma_start) + audio_noise * sigma_start
            log.info(f"Stage 2 audio: blending with noise at sigma={sigma_start:.4f}")
        else:
            audio_latent = mx.random.normal((1, AUDIO_LATENT_CHANNELS, audio_latent_len, AUDIO_MEL_BINS))

        def _audio_latent_time_in_sec(start_idx: int, end_idx: int) -> np.ndarray:
            latent_frame = np.arange(start_idx, end_idx, dtype=np.float32)
            mel_frame = latent_frame * AUDIO_DOWNSAMPLE_FACTOR
            mel_frame = np.clip(mel_frame + 1 - AUDIO_DOWNSAMPLE_FACTOR, 0, None)
            return mel_frame * AUDIO_HOP_LENGTH / AUDIO_SAMPLE_RATE

        start_times = _audio_latent_time_in_sec(0, audio_latent_len)
        end_times = _audio_latent_time_in_sec(1, audio_latent_len + 1)
        audio_bounds = np.stack([start_times, end_times], axis=-1)
        audio_bounds = audio_bounds[np.newaxis, np.newaxis, :, :]
        audio_positions = mx.array(audio_bounds, dtype=mx.float32)

    # --- Denoising loop (matches mlx_video.generate_av.denoise_av) ---
    num_steps = len(sigmas) - 1
    log.info(f"Starting denoising: {num_steps} steps, sigmas={sigmas}")

    force_materialize = mx.eval
    b, c, f, h, w = latent_shape.to_tuple()
    num_video_tokens = f * h * w

    for step_idx in range(num_steps):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # Flatten video latent: (B, C, F, H, W) → (B, F*H*W, C)
        video_flat = mx.transpose(mx.reshape(latent, (b, c, -1)), (0, 2, 1))

        # Per-token timesteps (reference: mlx_video.generate_av lines 246-256)
        # Use masked timesteps when we have a denoise mask (I2V or retake)
        _has_denoise_mask = (
            config.image_latent is not None or config.retake_clean_latent is not None
        )
        if _has_denoise_mask:
            # mask (B,1,F,1,1) → broadcast → flatten to (B, num_tokens)
            denoise_mask_flat = mx.broadcast_to(
                video_state.denoise_mask, (b, 1, f, h, w)
            )
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_video_tokens))
            video_timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            video_timesteps = mx.full((b, num_video_tokens), sigma, dtype=dtype)

        # Video modality
        video_mod = Modality(
            latent=video_flat,
            context=config.prompt_embeds,
            context_mask=config.prompt_attention_mask,
            timesteps=video_timesteps,
            positions=positions,
            enabled=True,
            sigma=mx.array([sigma]),
        )

        # Audio modality
        audio_mod = None
        if audio_latent is not None and config.audio_prompt_embeds is not None:
            ab, ac, at, af = audio_latent.shape
            audio_patches = audio_latent.transpose(0, 2, 1, 3).reshape(ab, at, ac * af)
            audio_mod = Modality(
                latent=audio_patches,
                context=config.audio_prompt_embeds,
                context_mask=config.audio_prompt_attention_mask,
                timesteps=mx.full((1, at), sigma, dtype=dtype),
                positions=audio_positions,
                enabled=True,
                sigma=mx.array([sigma]),
            )

        # TeaCache step info
        if hasattr(velocity_model, "set_step_info"):
            velocity_model.set_step_info(step_idx, len(sigmas))

        # Model forward pass → velocity output (use velocity_model directly, not X0Model)
        output = velocity_model(video=video_mod, audio=audio_mod)

        if isinstance(output, tuple):
            video_velocity, audio_velocity = output
        else:
            video_velocity = output
            audio_velocity = None

        # Reshape velocity back: (B, F*H*W, C) → (B, C, F, H, W)
        video_velocity = mx.reshape(
            mx.transpose(video_velocity, (0, 2, 1)), (b, c, f, h, w)
        )

        # Compute denoised x0 (reference: to_denoised)
        x0 = latent - sigma * video_velocity

        # Apply conditioning mask on x0 BEFORE Euler step
        # (reference: mlx_video.conditioning.latent.apply_denoise_mask)
        # Used for I2V (preserve conditioned frame) and retake (preserve unmasked region)
        if _has_denoise_mask:
            x0 = apply_denoise_mask(x0, video_state.clean_latent, video_state.denoise_mask)

        force_materialize(x0)

        # Euler step (reference: mlx_video.generate_av lines 112-125)
        if sigma_next > 0:
            sigma_next_arr = mx.array(sigma_next, dtype=dtype)
            sigma_arr = mx.array(sigma, dtype=dtype)
            latent = x0 + sigma_next_arr * (latent - x0) / sigma_arr
        else:
            latent = x0

        # Audio euler step
        if audio_velocity is not None:
            audio_velocity = audio_velocity.reshape(ab, at, AUDIO_LATENT_CHANNELS, AUDIO_MEL_BINS)
            audio_velocity = audio_velocity.transpose(0, 2, 1, 3)
            audio_x0 = audio_latent - sigma * audio_velocity
            if sigma_next > 0:
                sigma_next_arr = mx.array(sigma_next, dtype=dtype)
                sigma_arr = mx.array(sigma, dtype=dtype)
                audio_latent = audio_x0 + sigma_next_arr * (audio_latent - audio_x0) / sigma_arr
            else:
                audio_latent = audio_x0

        force_materialize(latent)
        if audio_latent is not None:
            force_materialize(audio_latent)

        if progress_callback is not None:
            progress_callback(step_idx + 1, num_steps, latent)

        log.info(f"Step {step_idx + 1}/{num_steps}: sigma={sigma:.3f} → {sigma_next:.3f}")

    aggressive_cleanup()

    return GenerationOutput(
        video_latent=latent,
        audio_latent=audio_latent,
    )
