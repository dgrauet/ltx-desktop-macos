"""Transformer blocks for LTX-2.3.

Adapted from Acelogic/LTX-2-MLX. Supports V2 features:
- 9-param AdaLN (cross_attention_adaln)
- Per-head gated attention
- Prompt AdaLN for cross-attention KV modulation
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, rms_norm
from .feed_forward import FeedForward
from .rope import LTXRopeType


@dataclass
class TransformerConfig:
    """Configuration for a transformer stream."""
    dim: int
    heads: int
    d_head: int
    context_dim: int
    cross_attention_adaln: bool = False
    apply_gated_attention: bool = False


@dataclass
class TransformerArgs:
    """Arguments passed to transformer blocks during forward pass."""
    x: mx.array
    context: mx.array
    timesteps: mx.array
    positional_embeddings: tuple
    context_mask: Optional[mx.array] = None
    embedded_timestep: Optional[mx.array] = None
    cross_positional_embeddings: Optional[tuple] = None
    cross_scale_shift_timestep: Optional[mx.array] = None
    cross_gate_timestep: Optional[mx.array] = None
    enabled: bool = True
    prompt_timestep: Optional[mx.array] = None

    def replace(self, **kwargs) -> "TransformerArgs":
        return TransformerArgs(
            x=kwargs.get("x", self.x),
            context=kwargs.get("context", self.context),
            timesteps=kwargs.get("timesteps", self.timesteps),
            positional_embeddings=kwargs.get("positional_embeddings", self.positional_embeddings),
            context_mask=kwargs.get("context_mask", self.context_mask),
            embedded_timestep=kwargs.get("embedded_timestep", self.embedded_timestep),
            cross_positional_embeddings=kwargs.get("cross_positional_embeddings", self.cross_positional_embeddings),
            cross_scale_shift_timestep=kwargs.get("cross_scale_shift_timestep", self.cross_scale_shift_timestep),
            cross_gate_timestep=kwargs.get("cross_gate_timestep", self.cross_gate_timestep),
            enabled=kwargs.get("enabled", self.enabled),
            prompt_timestep=kwargs.get("prompt_timestep", self.prompt_timestep),
        )


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN.

    Supports both V1 (6-param AdaLN) and V2 (9-param AdaLN with cross-attention modulation).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        cross_attention_adaln: bool = False,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.norm_eps = norm_eps
        self.cross_attention_adaln = cross_attention_adaln

        adaln_params = 9 if cross_attention_adaln else 6

        self.attn1 = Attention(
            query_dim=dim, heads=num_heads, dim_head=head_dim,
            context_dim=None, rope_type=rope_type, norm_eps=norm_eps,
            apply_gated_attention=apply_gated_attention,
        )
        self.attn2 = Attention(
            query_dim=dim, context_dim=context_dim, heads=num_heads,
            dim_head=head_dim, rope_type=rope_type, norm_eps=norm_eps,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim)
        self.scale_shift_table = mx.zeros((adaln_params, dim), dtype=mx.float32)

        if cross_attention_adaln:
            self.prompt_scale_shift_table = mx.zeros((2, dim), dtype=mx.float32)

    def get_ada_values(self, batch_size, timestep, start, end):
        table_slice = self.scale_shift_table[start:end]
        ada_values = table_slice[None, None, :, :] + timestep[:, :, start:end, :]
        return tuple(ada_values[:, :, i, :] for i in range(end - start))

    def __call__(self, args: TransformerArgs) -> TransformerArgs:
        x = args.x
        batch_size = x.shape[0]

        shift_msa, scale_msa, gate_msa = self.get_ada_values(batch_size, args.timesteps, 0, 3)

        norm_x = rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
        attn_out = self.attn1(norm_x, pe=args.positional_embeddings)
        x = x + attn_out * gate_msa

        # Cross-attention
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(batch_size, args.timesteps, 6, 9)
            kv_mod = self.prompt_scale_shift_table[None, None, :, :] + args.prompt_timestep
            shift_kv = kv_mod[:, :, 0, :]
            scale_kv = kv_mod[:, :, 1, :]
            attn_input = rms_norm(x, eps=self.norm_eps) * (1 + scale_q) + shift_q
            encoder_hs = args.context * (1 + scale_kv) + shift_kv
            cross_out = self.attn2(attn_input, context=encoder_hs, mask=args.context_mask) * gate
        else:
            cross_out = self.attn2(rms_norm(x, eps=self.norm_eps), context=args.context, mask=args.context_mask)
        x = x + cross_out

        shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(batch_size, args.timesteps, 3, 6)
        x_scaled = rms_norm(x, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
        ff_out = self.ff(x_scaled)
        x = x + ff_out * gate_mlp

        return args.replace(x=x)


class BasicAVTransformerBlock(nn.Module):
    """Audio-Video transformer block with cross-modal attention.

    Supports V2 features: 9-param AdaLN, gated attention, prompt AdaLN.
    """

    def __init__(
        self,
        idx: int,
        video_config: Optional[TransformerConfig] = None,
        audio_config: Optional[TransformerConfig] = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        self.cross_attention_adaln = (
            (video_config is not None and video_config.cross_attention_adaln)
            or (audio_config is not None and audio_config.cross_attention_adaln)
        )
        adaln_params = 9 if self.cross_attention_adaln else 6

        # Video components
        if video_config is not None:
            self.attn1 = Attention(
                query_dim=video_config.dim, heads=video_config.heads,
                dim_head=video_config.d_head, context_dim=None, rope_type=rope_type,
                norm_eps=norm_eps, apply_gated_attention=video_config.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video_config.dim, context_dim=video_config.context_dim,
                heads=video_config.heads, dim_head=video_config.d_head,
                rope_type=rope_type, norm_eps=norm_eps,
                apply_gated_attention=video_config.apply_gated_attention,
            )
            self.ff = FeedForward(video_config.dim, dim_out=video_config.dim)
            self.scale_shift_table = mx.zeros((adaln_params, video_config.dim), dtype=mx.float32)

        # Audio components
        if audio_config is not None:
            self.audio_attn1 = Attention(
                query_dim=audio_config.dim, heads=audio_config.heads,
                dim_head=audio_config.d_head, context_dim=None, rope_type=rope_type,
                norm_eps=norm_eps, apply_gated_attention=audio_config.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio_config.dim, context_dim=audio_config.context_dim,
                heads=audio_config.heads, dim_head=audio_config.d_head,
                rope_type=rope_type, norm_eps=norm_eps,
                apply_gated_attention=audio_config.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio_config.dim, dim_out=audio_config.dim)
            self.audio_scale_shift_table = mx.zeros((adaln_params, audio_config.dim), dtype=mx.float32)

        # V2 cross-attention adaln: per-block prompt scale/shift tables
        if self.cross_attention_adaln and video_config is not None:
            self.prompt_scale_shift_table = mx.zeros((2, video_config.dim), dtype=mx.float32)
        if self.cross_attention_adaln and audio_config is not None:
            self.audio_prompt_scale_shift_table = mx.zeros((2, audio_config.dim), dtype=mx.float32)

        # Cross-modal attention (audio <-> video)
        if audio_config is not None and video_config is not None:
            self.audio_to_video_attn = Attention(
                query_dim=video_config.dim, context_dim=audio_config.dim,
                heads=audio_config.heads, dim_head=audio_config.d_head,
                rope_type=rope_type, norm_eps=norm_eps,
                apply_gated_attention=video_config.apply_gated_attention,
            )
            self.video_to_audio_attn = Attention(
                query_dim=audio_config.dim, context_dim=video_config.dim,
                heads=audio_config.heads, dim_head=audio_config.d_head,
                rope_type=rope_type, norm_eps=norm_eps,
                apply_gated_attention=audio_config.apply_gated_attention,
            )
            self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio_config.dim), dtype=mx.float32)
            self.scale_shift_table_a2v_ca_video = mx.zeros((5, video_config.dim), dtype=mx.float32)

    def get_ada_values(self, scale_shift_table, batch_size, timestep, start, end):
        table_slice = scale_shift_table[start:end]
        ada_values = table_slice[None, None, :, :] + timestep[:, :, start:end, :]
        return tuple(ada_values[:, :, i, :] for i in range(end - start))

    def get_av_ca_ada_values(self, scale_shift_table, batch_size,
                             scale_shift_timestep, gate_timestep, num_scale_shift_values=4):
        table_slice = scale_shift_table[:num_scale_shift_values]
        scale_shift_ada = table_slice[None, None, :, :] + scale_shift_timestep
        scale_shift_values = tuple(scale_shift_ada[:, :, i, :] for i in range(num_scale_shift_values))
        gate_table = scale_shift_table[num_scale_shift_values:]
        gate_ada = gate_table[None, None, :, :] + gate_timestep
        gate_values = tuple(gate_ada[:, :, i, :] for i in range(gate_ada.shape[2]))
        return (*scale_shift_values, *gate_values)

    def _apply_text_cross_attention(self, x, context, attn, scale_shift_table,
                                     prompt_scale_shift_table, timestep, prompt_timestep,
                                     context_mask):
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(
                scale_shift_table, x.shape[0], timestep, 6, 9)
            kv_mod = prompt_scale_shift_table[None, None, :, :] + prompt_timestep
            shift_kv = kv_mod[:, :, 0, :]
            scale_kv = kv_mod[:, :, 1, :]
            attn_input = rms_norm(x, eps=self.norm_eps) * (1 + scale_q) + shift_q
            encoder_hs = context * (1 + scale_kv) + shift_kv
            return attn(attn_input, context=encoder_hs, mask=context_mask) * gate
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def __call__(self, video, audio, **kwargs):
        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = vx is not None and video is not None and video.enabled and vx.size > 0
        run_ax = ax is not None and audio is not None and audio.enabled and ax.size > 0
        run_a2v = run_vx and ax is not None and ax.size > 0
        run_v2a = run_ax and vx is not None and vx.size > 0

        # Video self-attention + cross-attention to text
        if run_vx:
            shift_msa, scale_msa, gate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, 0, 3)
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
            attn_out = self.attn1(norm_vx, pe=video.positional_embeddings)
            vx = vx + attn_out * gate_msa

            cross_out = self._apply_text_cross_attention(
                vx, video.context, self.attn2, self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps, video.prompt_timestep, video.context_mask,
            )
            vx = vx + cross_out

        # Audio self-attention + cross-attention to text
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, 0, 3)
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            attn_out = self.audio_attn1(norm_ax, pe=audio.positional_embeddings)
            ax = ax + attn_out * agate_msa

            cross_out = self._apply_text_cross_attention(
                ax, audio.context, self.audio_attn2, self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps, audio.prompt_timestep, audio.context_mask,
            )
            ax = ax + cross_out

        # Audio-Video cross-modal attention
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (scale_ca_audio_a2v, shift_ca_audio_a2v, scale_ca_audio_v2a,
             shift_ca_audio_v2a, gate_out_v2a) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio, ax.shape[0],
                audio.cross_scale_shift_timestep, audio.cross_gate_timestep)

            (scale_ca_video_a2v, shift_ca_video_a2v, scale_ca_video_v2a,
             shift_ca_video_v2a, gate_out_a2v) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video, vx.shape[0],
                video.cross_scale_shift_timestep, video.cross_gate_timestep)

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                vx = vx + self.audio_to_video_attn(
                    vx_scaled, context=ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                ) * gate_out_a2v

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                ax = ax + self.video_to_audio_attn(
                    ax_scaled, context=vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                ) * gate_out_v2a

        # Video feed-forward
        if run_vx:
            shift_mlp, scale_mlp, gate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, 3, 6)
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
            ff_out = self.ff(vx_scaled)
            vx = vx + ff_out * gate_mlp

        # Audio feed-forward
        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, 3, 6)
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ff_out = self.audio_ff(ax_scaled)
            ax = ax + ff_out * agate_mlp

        video_out = video.replace(x=vx) if video is not None else None
        audio_out = audio.replace(x=ax) if audio is not None else None
        return video_out, audio_out
