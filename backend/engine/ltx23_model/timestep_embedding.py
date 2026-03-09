"""Timestep embeddings for LTX-2.3 Transformer.

Adapted from Acelogic/LTX-2-MLX. Key naming: linear1/linear2 instead of
linear_1/linear_2 to match our converted weight format.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings."""
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding generator."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """MLP to project timestep embeddings to model dimension.

    Weight keys: linear1.{weight,bias}, linear2.{weight,bias}
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=bias)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear1(sample)
        sample = nn.silu(sample)
        sample = self.linear2(sample)
        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Combined timestep and size embeddings."""

    def __init__(self, embedding_dim: int, size_emb_dim: int = 0):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Norm with scale and shift from timestep embedding.

    For V2 (LTX-2.3): num_embeddings=9 (6 base + 3 cross-attention adaln).
    For V1 (LTX-2.0): num_embeddings=6.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int = 6):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, num_embeddings * embedding_dim, bias=True)

    def __call__(self, timestep: mx.array) -> tuple:
        embedded_timestep = self.emb(timestep)
        emb = self.silu(embedded_timestep)
        emb = self.linear(emb)
        return emb, embedded_timestep
