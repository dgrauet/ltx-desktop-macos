"""3D Rotary Position Embeddings (RoPE) for LTX-2.3 Transformer.

Adapted from Acelogic/LTX-2-MLX. Removed custom Metal kernel dependency.
"""

import math
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: mx.array,
    freqs_cis: Tuple[mx.array, mx.array],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> mx.array:
    """Apply rotary position embeddings to input tensor."""
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply interleaved rotary embeddings."""
    shape = input_tensor.shape
    t_dup = input_tensor.reshape(*shape[:-1], shape[-1] // 2, 2)
    t1 = t_dup[..., 0]
    t2 = t_dup[..., 1]
    t_rot = mx.stack([-t2, t1], axis=-1)
    input_tensor_rot = t_rot.reshape(shape)
    return input_tensor * cos_freqs + input_tensor_rot * sin_freqs


def apply_split_rotary_emb(
    input_tensor: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply split rotary embeddings."""
    needs_reshape = False

    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1)
        input_tensor = input_tensor.transpose(0, 2, 1, 3)
        needs_reshape = True

    dim = input_tensor.shape[-1]
    split_input = input_tensor.reshape(*input_tensor.shape[:-1], 2, dim // 2)
    first_half = split_input[..., 0, :]
    second_half = split_input[..., 1, :]

    first_half_out = first_half * cos_freqs - second_half * sin_freqs
    second_half_out = second_half * cos_freqs + first_half * sin_freqs

    output = mx.stack([first_half_out, second_half_out], axis=-2)
    output = output.reshape(*output.shape[:-2], dim)

    if needs_reshape:
        b, h, t, d = output.shape
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(b, t, h * d)

    return output


def generate_freq_grid(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
) -> mx.array:
    """Generate frequency grid using MLX."""
    theta = positional_embedding_theta
    start = 1.0
    end = float(theta)
    n_elem = 2 * positional_embedding_max_pos_count
    log_start = math.log(start) / math.log(theta)
    log_end = math.log(end) / math.log(theta)
    num_indices = inner_dim // n_elem
    linspace = mx.linspace(log_start, log_end, num_indices)
    indices = theta ** linspace
    indices = indices * (math.pi / 2)
    return indices.astype(mx.float32)


def get_fractional_positions(
    indices_grid: mx.array,
    max_pos: List[int],
) -> mx.array:
    """Convert position indices to fractional positions in [0, 1]."""
    n_pos_dims = indices_grid.shape[1]
    fractional = []
    for i in range(n_pos_dims):
        fractional.append(indices_grid[:, i, :] / max_pos[i])
    return mx.stack(fractional, axis=-1)


def generate_freqs(
    indices: mx.array,
    indices_grid: mx.array,
    max_pos: List[int],
    use_middle_indices_grid: bool,
) -> mx.array:
    """Generate frequencies from position indices."""
    if use_middle_indices_grid:
        assert indices_grid.ndim == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start = indices_grid[..., 0]
        indices_grid_end = indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif indices_grid.ndim == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    scaled_positions = fractional_positions * 2 - 1
    scaled_positions = scaled_positions[..., None]
    indices = indices[None, None, None, :]
    freqs = indices * scaled_positions
    freqs = freqs.transpose(0, 1, 3, 2)
    freqs = freqs.reshape(freqs.shape[0], freqs.shape[1], -1)
    return freqs


def split_freqs_cis(
    freqs: mx.array,
    pad_size: int,
    num_attention_heads: int,
) -> Tuple[mx.array, mx.array]:
    """Compute cos/sin frequencies for split RoPE format."""
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)

    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    b, t, _ = cos_freq.shape
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)
    cos_freq = cos_freq.transpose(0, 2, 1, 3)
    sin_freq = sin_freq.transpose(0, 2, 1, 3)
    return cos_freq, sin_freq


def interleaved_freqs_cis(
    freqs: mx.array,
    pad_size: int,
) -> Tuple[mx.array, mx.array]:
    """Compute cos/sin frequencies for interleaved RoPE format."""
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)
    cos_freq = mx.repeat(cos_freq, 2, axis=-1)
    sin_freq = mx.repeat(sin_freq, 2, axis=-1)

    if pad_size != 0:
        cos_padding = mx.ones((cos_freq.shape[0], cos_freq.shape[1], pad_size))
        sin_padding = mx.zeros((sin_freq.shape[0], sin_freq.shape[1], pad_size))
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)

    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: mx.array,
    dim: int,
    out_dtype: mx.Dtype = mx.float32,
    theta: float = 10000.0,
    max_pos: Optional[List[int]] = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> Tuple[mx.array, mx.array]:
    """Precompute cosine and sine frequencies for RoPE."""
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    n_pos_dims = indices_grid.shape[1]
    indices = generate_freq_grid(theta, n_pos_dims, dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * n_pos_dims
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)

    return cos_freq.astype(out_dtype), sin_freq.astype(out_dtype)
