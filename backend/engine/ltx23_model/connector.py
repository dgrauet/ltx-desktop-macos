"""Embeddings connector for LTX-2.3 text encoder.

1D transformer that refines text embeddings before cross-attention.
Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, rms_norm
from .feed_forward import FeedForward
from .rope import LTXRopeType, precompute_freqs_cis


class BasicTransformerBlock1D(nn.Module):
    """1D transformer block: RMSNorm → self-attention → RMSNorm → FFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.norm_eps = norm_eps
        self.attn1 = Attention(
            query_dim=dim, heads=heads, dim_head=dim_head,
            context_dim=None, rope_type=rope_type, norm_eps=norm_eps,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        pe: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden = rms_norm(hidden_states, eps=self.norm_eps)
        attn_output = self.attn1(norm_hidden, mask=attention_mask, pe=pe)
        hidden_states = hidden_states + attn_output

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden = rms_norm(hidden_states, eps=self.norm_eps)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_output

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    """1D embeddings connector with RoPE and learnable registers.

    Bridges Gemma text encoder output to the diffusion transformer.
    V2 (LTX-2.3): 8 layers, gated attention, 4096-dim for video, 2048-dim for audio.
    V1 (LTX-2.0): 2 layers, 3840-dim.
    """

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[List[int]] = None,
        num_learnable_registers: Optional[int] = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [1]
        self.rope_type = rope_type
        self.norm_eps = norm_eps

        self.transformer_1d_blocks = [
            BasicTransformerBlock1D(
                dim=self.inner_dim, heads=num_attention_heads,
                dim_head=attention_head_dim, rope_type=rope_type,
                norm_eps=norm_eps, apply_gated_attention=apply_gated_attention,
            )
            for _ in range(num_layers)
        ]

        self.num_learnable_registers = num_learnable_registers
        if num_learnable_registers:
            self.learnable_registers = mx.random.uniform(
                low=-1.0, high=1.0,
                shape=(num_learnable_registers, self.inner_dim),
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Replace padded positions with learnable register tokens."""
        seq_len = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        num_duplications = seq_len // self.num_learnable_registers
        tiled_registers = mx.tile(
            self.learnable_registers[None, :, :],
            (batch_size, num_duplications, 1),
        )

        mask_squeezed = attention_mask.squeeze(1).squeeze(1)
        is_valid = (mask_squeezed >= -9000.0)

        idx = mx.arange(seq_len, dtype=mx.int32)[None, :]
        valid_int = is_valid.astype(mx.int32)
        sort_key = (1 - valid_int) * seq_len + idx
        order = mx.argsort(sort_key, axis=1)
        adjusted_hidden_states = mx.take_along_axis(
            hidden_states, order[:, :, None], axis=1
        )

        flipped_mask = is_valid.astype(hidden_states.dtype)[:, ::-1, None]
        hidden_states = (
            flipped_mask * adjusted_hidden_states
            + (1 - flipped_mask) * tiled_registers
        )

        new_mask = mx.zeros_like(attention_mask)
        return hidden_states, new_mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Process embeddings through connector.

        Args:
            hidden_states: [B, T, D]
            attention_mask: Additive mask [B, 1, 1, T]

        Returns:
            (processed_hidden_states, attention_mask)
        """
        if self.num_learnable_registers and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, attention_mask
            )

        seq_len = hidden_states.shape[1]
        indices_grid = mx.arange(seq_len, dtype=mx.float32)[None, None, :]

        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states, eps=self.norm_eps)

        if attention_mask is None:
            attention_mask = mx.zeros((hidden_states.shape[0], 1, 1, hidden_states.shape[1]))

        return hidden_states, attention_mask
