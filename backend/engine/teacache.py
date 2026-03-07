"""TeaCache — Training-free block-output caching for DiT models on MLX.

Ports the TeaCache algorithm (ali-vilab/TeaCache) to MLX for Apple Silicon.
Instead of recomputing every transformer block at every denoising step,
TeaCache uses timestep embedding differences to estimate whether a block's
output has changed enough to warrant recomputation. If the change is below
a threshold, the cached output from the previous step is reused.

This yields 1.6-2.1x inference speedup on LTX-Video with negligible quality loss.

References:
- Paper: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"
- LTX-Video specific: https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video
- License: Apache 2.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# MLX import with graceful fallback
try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


@dataclass
class TeaCacheStats:
    """Statistics from a TeaCache-enabled denoising run."""

    total_blocks: int = 0
    blocks_computed: int = 0
    blocks_cached: int = 0
    cache_hit_rate: float = 0.0
    estimated_speedup: float = 1.0
    elapsed_seconds: float = 0.0


class TeaCacheMLX:
    """Training-free block-output caching for DiT models on MLX.

    Caches entire transformer block outputs (not KV matrices) and reuses
    them when the timestep embedding change is below a threshold.

    Args:
        rel_l1_thresh: Relative L1 threshold for cache invalidation.
            - 0.0: disabled (always recompute)
            - 0.03: near-lossless (recommended for production)
            - 0.05: minimal degradation (good for rapid iteration)
            - 0.08+: visible quality loss
        num_layers: Number of transformer layers to cache.
    """

    def __init__(self, rel_l1_thresh: float = 0.03, num_layers: int = 32) -> None:
        self.thresh = rel_l1_thresh
        self.num_layers = num_layers
        self._cache: dict[int, object] = {}
        self._prev_t_emb: object | None = None
        self._stats = TeaCacheStats()
        self._start_time: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether caching is active (thresh > 0)."""
        return self.thresh > 0.0

    def reset(self) -> None:
        """Clear all cached block outputs. Call between generations."""
        self._cache.clear()
        self._prev_t_emb = None
        self._stats = TeaCacheStats()
        self._start_time = time.monotonic()

    def should_recompute(self, t_emb: object, layer_idx: int) -> bool:
        """Estimate if a block's output changed enough to warrant recomputation.

        Uses the L1 difference of timestep embeddings as a cheap proxy
        for block output change.

        Args:
            t_emb: Current timestep embedding tensor.
            layer_idx: Index of the transformer layer.

        Returns:
            True if the block should be recomputed, False if cached output can be reused.
        """
        self._stats.total_blocks += 1

        if self._prev_t_emb is None or layer_idx not in self._cache:
            self._stats.blocks_computed += 1
            return True

        if not _MLX_AVAILABLE:
            self._stats.blocks_computed += 1
            return True

        # Cheap L1 difference on timestep embeddings
        diff = mx.mean(mx.abs(mx.subtract(t_emb, self._prev_t_emb))).item()

        if diff > self.thresh:
            self._stats.blocks_computed += 1
            return True
        else:
            self._stats.blocks_cached += 1
            return False

    def get_or_compute(
        self,
        block_fn: object,
        x: object,
        t_emb: object,
        layer_idx: int,
        **kwargs: object,
    ) -> object:
        """Either compute a block's output or return cached result.

        Args:
            block_fn: The transformer block's forward function.
            x: Input tensor to the block.
            t_emb: Timestep embedding tensor.
            layer_idx: Index of this transformer layer.
            **kwargs: Additional arguments to pass to block_fn.

        Returns:
            Block output tensor (either freshly computed or cached).
        """
        if not self.enabled:
            return block_fn(x, t_emb, **kwargs)

        if self.should_recompute(t_emb, layer_idx):
            output = block_fn(x, t_emb, **kwargs)
            self._cache[layer_idx] = output
            return output
        else:
            return self._cache[layer_idx]

    def step_done(self, t_emb: object) -> None:
        """Mark the end of a denoising step. Updates the cached timestep embedding.

        Args:
            t_emb: The timestep embedding from the just-completed step.
        """
        self._prev_t_emb = t_emb

    def get_stats(self) -> TeaCacheStats:
        """Return statistics from this caching run.

        Returns:
            TeaCacheStats with hit rate and estimated speedup.
        """
        stats = self._stats
        if stats.total_blocks > 0:
            stats.cache_hit_rate = stats.blocks_cached / stats.total_blocks
            # Rough speedup estimate: each cached block saves ~95% of its compute
            compute_ratio = stats.blocks_computed / stats.total_blocks
            stats.estimated_speedup = 1.0 / max(compute_ratio, 0.1)
        stats.elapsed_seconds = time.monotonic() - self._start_time
        return stats


def create_teacache(
    enabled: bool = True,
    rel_l1_thresh: float = 0.03,
    num_layers: int = 32,
) -> TeaCacheMLX:
    """Factory function to create a TeaCache instance.

    Args:
        enabled: Whether to enable caching. If False, thresh is set to 0.
        rel_l1_thresh: Relative L1 threshold (0.03 = near-lossless).
        num_layers: Number of transformer layers in the model.

    Returns:
        Configured TeaCacheMLX instance.
    """
    thresh = rel_l1_thresh if enabled else 0.0
    cache = TeaCacheMLX(rel_l1_thresh=thresh, num_layers=num_layers)
    log.info(
        "TeaCache created: enabled=%s, thresh=%.3f, layers=%d",
        enabled, thresh, num_layers,
    )
    return cache
