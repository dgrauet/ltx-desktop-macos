"""TeaCache — block-level output caching for diffusion transformers.

Caches the cumulative residual of all transformer blocks between
consecutive denoising steps. When timestep-to-timestep change is below
threshold, reuses cached residual instead of running 48 expensive blocks.

Reference: https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video
"""

import logging
from typing import Optional

import mlx.core as mx
import numpy as np

log = logging.getLogger(__name__)

# Rescale polynomial coefficients fitted to LTX-Video block dynamics
_RESCALE_COEFFS = [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03]


class TeaCacheMLX:
    """Block-level residual caching for MLX diffusion transformers.

    Caches the cumulative residual (output - input) of ALL transformer blocks
    between consecutive denoising steps. When the modulated input hasn't changed
    much (measured by relative L1 distance, rescaled through a polynomial fitted
    to LTX-Video dynamics), the cached residual is reused instead of running
    all 48 blocks.

    Args:
        rel_l1_thresh: Accumulated rescaled distance threshold for cache
            invalidation.
            - 0.0: disabled (always recompute)
            - 0.03: near-lossless (recommended for production)
            - 0.05: minimal degradation (good for rapid preview)
            - 0.08+: visible quality loss
    """

    def __init__(self, rel_l1_thresh: float = 0.03) -> None:
        self.rel_l1_thresh = rel_l1_thresh
        self._rescale_func = np.poly1d(_RESCALE_COEFFS)
        self._accumulated_distance: float = 0.0
        self._prev_modulated_input: Optional[mx.array] = None
        self._cached_video_residual: Optional[mx.array] = None
        self._cached_audio_residual: Optional[mx.array] = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def should_skip_blocks(
        self, modulated_input: mx.array, step_idx: int, total_steps: int
    ) -> bool:
        """Decide whether to skip transformer blocks using cached residual.

        Always computes on first and last step. For middle steps, compares
        current modulated input to previous via relative L1 distance,
        rescales through polynomial, and accumulates distance.

        Args:
            modulated_input: The video hidden state after patchify+AdaLN
                (i.e. video_args.x after preprocessing), used as the
                change-detection signal.
            step_idx: Current denoising step index (0-based).
            total_steps: Total number of denoising steps.

        Returns:
            True if cached residual should be used (skip blocks),
            False if blocks must be recomputed.
        """
        # Always compute first and last step
        if step_idx == 0 or step_idx == total_steps - 1:
            self._prev_modulated_input = modulated_input
            self._accumulated_distance = 0.0
            return False

        # No previous input to compare against
        if self._prev_modulated_input is None:
            self._prev_modulated_input = modulated_input
            return False

        # Compute relative L1 distance
        diff = mx.abs(modulated_input - self._prev_modulated_input)
        mean_diff = float(mx.mean(diff).item())
        mean_prev = float(mx.mean(mx.abs(self._prev_modulated_input)).item())
        rel_l1 = mean_diff / max(mean_prev, 1e-8)

        # Rescale through polynomial fitted to LTX-Video dynamics
        rescaled = float(self._rescale_func(rel_l1))

        # Accumulate distance
        self._accumulated_distance += rescaled

        # Update previous input
        self._prev_modulated_input = modulated_input

        # If accumulated distance exceeds threshold, must recompute
        if self._accumulated_distance >= self.rel_l1_thresh or self._cached_video_residual is None:
            self._accumulated_distance = 0.0
            self._cache_misses += 1
            return False

        self._cache_hits += 1
        return True

    def store_residuals(
        self,
        video_input: mx.array,
        video_output: mx.array,
        audio_input: Optional[mx.array] = None,
        audio_output: Optional[mx.array] = None,
    ) -> None:
        """Store residuals (output - input) for future reuse.

        Args:
            video_input: Video hidden state before transformer blocks.
            video_output: Video hidden state after transformer blocks.
            audio_input: Audio hidden state before transformer blocks (if audio enabled).
            audio_output: Audio hidden state after transformer blocks (if audio enabled).
        """
        self._cached_video_residual = video_output - video_input
        if audio_input is not None and audio_output is not None:
            self._cached_audio_residual = audio_output - audio_input

    def apply_cached_residual(
        self, video_input: mx.array, audio_input: Optional[mx.array] = None
    ) -> tuple[mx.array, Optional[mx.array]]:
        """Apply cached residual to current input.

        Args:
            video_input: Current video hidden state.
            audio_input: Current audio hidden state (if audio enabled).

        Returns:
            Tuple of (video_output, audio_output). audio_output is None
            if no audio residual is cached.
        """
        video_out = video_input + self._cached_video_residual
        audio_out = None
        if audio_input is not None and self._cached_audio_residual is not None:
            audio_out = audio_input + self._cached_audio_residual
        return video_out, audio_out

    def reset(self) -> None:
        """Reset state between generations."""
        self._accumulated_distance = 0.0
        self._prev_modulated_input = None
        self._cached_video_residual = None
        self._cached_audio_residual = None
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def stats(self) -> str:
        """Human-readable cache statistics."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "TeaCache: no steps processed"
        hit_pct = self._cache_hits / total * 100
        return f"TeaCache: {self._cache_hits}/{total} steps cached ({hit_pct:.0f}%)"
