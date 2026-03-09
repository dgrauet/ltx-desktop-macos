"""Attention mechanisms for LTX-2.3 Transformer.

Adapted from Acelogic/LTX-2-MLX. Includes V2 per-head gated attention.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .rope import LTXRopeType, apply_rotary_emb


def rms_norm(x: mx.array, weight: Optional[mx.array] = None, eps: float = 1e-6) -> mx.array:
    """Apply RMS normalization using optimized MLX implementation."""
    return mx.fast.rms_norm(x, weight, eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """Multi-head attention with RMSNorm on Q/K, optional RoPE and V2 gating.

    V2 (LTX-2.3): apply_gated_attention=True adds per-head gating via to_gate_logits.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        context_dim = query_dim if context_dim is None else context_dim

        self.q_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)

        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        pe: Optional[tuple] = None,
        k_pe: Optional[tuple] = None,
    ) -> mx.array:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Reshape for multi-head attention
        b, t_q, _ = q.shape
        _, t_k, _ = k.shape
        q = q.reshape(b, t_q, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, t_k, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t_k, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # Handle mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            mask = mask.astype(q.dtype)

        scale = 1.0 / (self.dim_head ** 0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, t_q, self.heads * self.dim_head)

        # Apply per-head gating if enabled (V2)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            out = out.reshape(b, t_q, self.heads, self.dim_head)
            gates = 2.0 * mx.sigmoid(gate_logits)
            out = out * gates[:, :, :, None]
            out = out.reshape(b, t_q, self.heads * self.dim_head)

        return self.to_out(out)
