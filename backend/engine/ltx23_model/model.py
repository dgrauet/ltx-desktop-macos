"""LTX-2.3 Transformer Model for MLX (Unified Video/Audio).

Adapted from Acelogic/LTX-2-MLX (Apache-2.0). Key naming adapted for our
split/quantized weight format.

Key naming differences from upstream:
- linear1/linear2 (ours) vs linear_1/linear_2 (upstream) in PixArtAlphaTextProjection
- ff.proj_in/proj_out (ours) vs ff.project_in.proj/project_out (upstream) in FeedForward
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

from .rope import LTXRopeType, precompute_freqs_cis
from .timestep_embedding import AdaLayerNormSingle
from .transformer import (
    BasicAVTransformerBlock,
    TransformerArgs,
    TransformerConfig,
)


class LTXModelType(Enum):
    """Model type variants."""

    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class PixArtAlphaTextProjection(nn.Module):
    """Projects caption embeddings with GELU activation.

    Weight keys: linear1.{weight,bias}, linear2.{weight,bias}

    Note: Not used for LTX-2.3 (caption_channels=None), but kept for V1
    compatibility.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, out_features, bias=True)

    def __call__(self, caption: mx.array) -> mx.array:
        hidden_states = self.linear1(caption)
        hidden_states = nn.gelu_approx(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


@dataclass
class Modality:
    """Input modality data (video or audio)."""

    latent: mx.array
    context: mx.array
    context_mask: Optional[mx.array]
    timesteps: mx.array
    positions: mx.array
    enabled: bool = True
    sigma: Optional[mx.array] = None


class TransformerArgsPreprocessor:
    """Preprocesses inputs for transformer blocks.

    Handles patchify projection, timestep embedding via AdaLN,
    caption projection, and RoPE computation.
    """

    def __init__(
        self,
        patchify_proj: nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: Optional[PixArtAlphaTextProjection],
        inner_dim: int,
        max_pos: List[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool = True,
        timestep_scale_multiplier: int = 1000,
        positional_embedding_theta: float = 10000.0,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        compute_dtype: mx.Dtype = mx.float32,
        prompt_adaln: Optional[AdaLayerNormSingle] = None,
    ):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.compute_dtype = compute_dtype
        self.prompt_adaln = prompt_adaln

    def _prepare_timestep(
        self,
        timestep: mx.array,
        adaln: AdaLayerNormSingle,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare timestep embeddings."""
        timestep = timestep * self.timestep_scale_multiplier
        emb, embedded_timestep = adaln(timestep.flatten())

        num_embeddings = emb.shape[-1] // self.inner_dim
        emb = emb.reshape(batch_size, -1, num_embeddings, self.inner_dim)
        embedded_timestep = embedded_timestep.reshape(batch_size, -1, self.inner_dim)
        return emb, embedded_timestep

    def _prepare_context(
        self,
        context: mx.array,
        x: mx.array,
    ) -> mx.array:
        """Prepare context (caption) for cross-attention."""
        batch_size = x.shape[0]
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        context = context.reshape(batch_size, -1, x.shape[-1])
        return context

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[mx.array],
        target_dtype: mx.Dtype = mx.float32,
    ) -> Optional[mx.array]:
        """Convert boolean mask to additive mask for softmax."""
        if attention_mask is None:
            return None

        if attention_mask.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return attention_mask

        if target_dtype == mx.float16:
            mask_value = -65504.0
        elif target_dtype == mx.bfloat16:
            mask_value = -3.38e38
        else:
            mask_value = -3.40e38

        mask = (1 - attention_mask.astype(mx.float32)) * mask_value
        mask = mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
        return mask.astype(target_dtype)

    def _prepare_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare RoPE positional embeddings."""
        return precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=mx.float32,
            theta=self.positional_embedding_theta,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
        )

    def prepare(self, modality: Modality) -> TransformerArgs:
        """Prepare all inputs for transformer blocks."""
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]

        timestep_emb, embedded_timestep = self._prepare_timestep(
            modality.timesteps, self.adaln, batch_size
        )

        prompt_timestep = None
        if self.prompt_adaln is not None:
            sigma = modality.sigma if modality.sigma is not None else modality.timesteps
            if sigma.ndim > 1:
                sigma = sigma[:, 0]
            prompt_emb, _ = self._prepare_timestep(sigma, self.prompt_adaln, batch_size)
            prompt_timestep = prompt_emb

        context = self._prepare_context(modality.context, x)
        attention_mask = self._prepare_attention_mask(
            modality.context_mask, target_dtype=self.compute_dtype
        )
        pe = self._prepare_positional_embeddings(modality.positions)

        return TransformerArgs(
            x=x,
            context=context,
            timesteps=timestep_emb,
            positional_embeddings=pe,
            context_mask=attention_mask,
            embedded_timestep=embedded_timestep,
            prompt_timestep=prompt_timestep,
        )


class MultiModalTransformerArgsPreprocessor:
    """Preprocesses inputs for AudioVideo transformer blocks.

    Extends TransformerArgsPreprocessor with cross-modal attention support.
    """

    def __init__(
        self,
        simple_preprocessor: TransformerArgsPreprocessor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        cross_pe_max_pos: int,
        audio_cross_attention_dim: int,
        av_ca_timestep_scale_multiplier: int = 1,
    ):
        self.simple_preprocessor = simple_preprocessor
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def _prepare_cross_positional_embeddings(
        self,
        positions: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare cross-modal positional embeddings (temporal dim only)."""
        temporal_positions = positions[:, 0:1, :]
        return precompute_freqs_cis(
            indices_grid=temporal_positions,
            dim=self.audio_cross_attention_dim,
            out_dtype=mx.float32,
            theta=self.simple_preprocessor.positional_embedding_theta,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            rope_type=self.simple_preprocessor.rope_type,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: mx.array,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare cross-attention timestep embeddings (scale/shift + gate)."""
        scaled_timestep = timestep * self.simple_preprocessor.timestep_scale_multiplier

        scale_shift_emb, _ = self.cross_scale_shift_adaln(scaled_timestep.flatten())
        scale_shift_emb = scale_shift_emb.reshape(
            batch_size, -1, 4, self.simple_preprocessor.inner_dim
        )

        av_ca_factor = (
            self.av_ca_timestep_scale_multiplier
            / self.simple_preprocessor.timestep_scale_multiplier
        )
        gate_emb, _ = self.cross_gate_adaln((scaled_timestep * av_ca_factor).flatten())
        gate_emb = gate_emb.reshape(
            batch_size, -1, 1, self.simple_preprocessor.inner_dim
        )
        return scale_shift_emb, gate_emb

    def prepare(self, modality: Modality) -> TransformerArgs:
        """Prepare all inputs for AudioVideo transformer blocks."""
        args = self.simple_preprocessor.prepare(modality)

        cross_pe = self._prepare_cross_positional_embeddings(modality.positions)

        batch_size = args.x.shape[0]
        cross_scale_shift, cross_gate = self._prepare_cross_attention_timestep(
            modality.timesteps, batch_size
        )

        return args.replace(
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift,
            cross_gate_timestep=cross_gate,
        )


class LTXModel(nn.Module):
    """LTX-2.3 Transformer Model (Unified Video/Audio).

    Supports VideoOnly, AudioOnly, and AudioVideo modes.
    48 transformer blocks with self-attention, cross-attention, FFN.
    AdaLN conditioning on timestep. V2 features: gated attention, 9-param AdaLN.
    """

    AUDIO_ATTENTION_HEADS = 32
    AUDIO_HEAD_DIM = 64
    AUDIO_IN_CHANNELS = 128
    AUDIO_OUT_CHANNELS = 128
    AUDIO_CROSS_PE_MAX_POS = 20

    def __init__(
        self,
        model_type: LTXModelType = LTXModelType.VideoOnly,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        audio_cross_attention_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        caption_channels: Optional[int] = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: int = 1000,
        av_ca_timestep_scale_multiplier: int = 1,
        use_middle_indices_grid: bool = True,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        compute_dtype: mx.Dtype = mx.float32,
        low_memory: bool = False,
        fast_mode: bool = False,
        cross_attention_adaln: bool = False,
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.model_type = model_type
        self.rope_type = rope_type
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.use_middle_indices_grid = use_middle_indices_grid
        self.norm_eps = norm_eps
        self.compute_dtype = compute_dtype
        self.low_memory = low_memory
        self.fast_mode = fast_mode
        self.cross_attention_adaln = cross_attention_adaln

        if fast_mode:
            self._eval_frequency = 0
        elif low_memory:
            self._eval_frequency = 4
        else:
            self._eval_frequency = 8

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [20, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos

        self.num_attention_heads = num_attention_heads
        self.video_inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = self.video_inner_dim
        self.audio_inner_dim = self.AUDIO_ATTENTION_HEADS * self.AUDIO_HEAD_DIM
        self.audio_cross_attention_dim = (
            audio_cross_attention_dim if audio_cross_attention_dim is not None
            else cross_attention_dim
        )

        adaln_num_embeddings = 9 if cross_attention_adaln else 6

        # === VIDEO COMPONENTS ===
        if self.model_type.is_video_enabled():
            self.patchify_proj = nn.Linear(in_channels, self.video_inner_dim, bias=True)

            self.adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=adaln_num_embeddings
            )

            self.prompt_adaln_single = (
                AdaLayerNormSingle(self.video_inner_dim, num_embeddings=2)
                if cross_attention_adaln
                else None
            )

            if caption_channels is not None:
                self.caption_projection = PixArtAlphaTextProjection(
                    in_features=caption_channels, hidden_size=self.video_inner_dim
                )
            else:
                self.caption_projection = None

            self.scale_shift_table = mx.zeros(
                (2, self.video_inner_dim), dtype=mx.float32
            )
            self.norm_out = nn.LayerNorm(
                self.video_inner_dim, affine=False, eps=norm_eps
            )
            self.proj_out = nn.Linear(self.video_inner_dim, out_channels)

        # === AUDIO COMPONENTS ===
        if self.model_type.is_audio_enabled():
            self.audio_patchify_proj = nn.Linear(
                self.AUDIO_IN_CHANNELS, self.audio_inner_dim, bias=True
            )

            self.audio_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=adaln_num_embeddings
            )

            self.audio_prompt_adaln_single = (
                AdaLayerNormSingle(self.audio_inner_dim, num_embeddings=2)
                if cross_attention_adaln
                else None
            )

            if caption_channels is not None:
                self.audio_caption_projection = PixArtAlphaTextProjection(
                    in_features=caption_channels, hidden_size=self.audio_inner_dim
                )
            else:
                self.audio_caption_projection = None

            self.audio_scale_shift_table = mx.zeros(
                (2, self.audio_inner_dim), dtype=mx.float32
            )
            self.audio_norm_out = nn.LayerNorm(
                self.audio_inner_dim, affine=False, eps=norm_eps
            )
            self.audio_proj_out = nn.Linear(
                self.audio_inner_dim, self.AUDIO_OUT_CHANNELS
            )

        # === CROSS-MODAL COMPONENTS ===
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=4
            )
            self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
                self.video_inner_dim, num_embeddings=1
            )
            self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=4
            )
            self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
                self.audio_inner_dim, num_embeddings=1
            )

        # === TRANSFORMER BLOCKS ===
        video_config = None
        if self.model_type.is_video_enabled():
            video_config = TransformerConfig(
                dim=self.video_inner_dim,
                heads=num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                cross_attention_adaln=cross_attention_adaln,
                apply_gated_attention=apply_gated_attention,
            )

        audio_config = None
        if self.model_type.is_audio_enabled():
            audio_config = TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.AUDIO_ATTENTION_HEADS,
                d_head=self.AUDIO_HEAD_DIM,
                context_dim=self.audio_cross_attention_dim,
                cross_attention_adaln=cross_attention_adaln,
                apply_gated_attention=apply_gated_attention,
            )

        self.transformer_blocks = [
            BasicAVTransformerBlock(
                idx=i,
                video_config=video_config,
                audio_config=audio_config,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            for i in range(num_layers)
        ]

        # === PREPROCESSORS ===
        if self.model_type.is_video_enabled():
            video_simple_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.video_inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                compute_dtype=self.compute_dtype,
                prompt_adaln=self.prompt_adaln_single,
            )
            if self.model_type.is_audio_enabled():
                self._video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                    simple_preprocessor=video_simple_preprocessor,
                    cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                    cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                    cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
                    audio_cross_attention_dim=self.audio_inner_dim,
                    av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
                )
            else:
                self._video_args_preprocessor = video_simple_preprocessor

        if self.model_type.is_audio_enabled():
            audio_simple_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=[self.AUDIO_CROSS_PE_MAX_POS],
                num_attention_heads=self.AUDIO_ATTENTION_HEADS,
                use_middle_indices_grid=True,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                compute_dtype=self.compute_dtype,
                prompt_adaln=self.audio_prompt_adaln_single,
            )
            if self.model_type.is_video_enabled():
                self._audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                    simple_preprocessor=audio_simple_preprocessor,
                    cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                    cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                    cross_pe_max_pos=self.AUDIO_CROSS_PE_MAX_POS,
                    audio_cross_attention_dim=self.audio_inner_dim,
                    av_ca_timestep_scale_multiplier=av_ca_timestep_scale_multiplier,
                )
            else:
                self._audio_args_preprocessor = audio_simple_preprocessor

    def set_teacache(self, teacache) -> None:
        """Attach a TeaCacheMLX instance for block-level residual caching."""
        self._teacache = teacache

    def set_step_info(self, step_idx: int, total_steps: int) -> None:
        """Set current denoising step info (called before each forward pass)."""
        self._step_idx = step_idx
        self._total_steps = total_steps

    def _process_transformer_blocks(
        self,
        video_args: Optional[TransformerArgs] = None,
        audio_args: Optional[TransformerArgs] = None,
    ) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        """Process all transformer blocks with periodic materialization for memory.

        When TeaCache is enabled, checks whether the modulated input has changed
        enough to warrant recomputing all 48 blocks. If not, applies the cached
        cumulative residual instead, saving significant compute.
        """
        # TeaCache: check if we can skip all blocks
        teacache = getattr(self, "_teacache", None)
        step_idx = getattr(self, "_step_idx", 0)
        total_steps = getattr(self, "_total_steps", 1)

        if teacache is not None and video_args is not None:
            if teacache.should_skip_blocks(video_args.x, step_idx, total_steps):
                video_out_x, audio_out_x = teacache.apply_cached_residual(
                    video_args.x,
                    audio_args.x if audio_args is not None else None,
                )
                video_args = video_args.replace(x=video_out_x)
                if audio_args is not None and audio_out_x is not None:
                    audio_args = audio_args.replace(x=audio_out_x)
                return video_args, audio_args

        # Save input for residual computation
        video_input = video_args.x if video_args is not None else None
        audio_input = audio_args.x if audio_args is not None else None

        # Run all blocks normally
        # mx.eval is mlx.core.eval — tensor materialization, NOT Python eval()
        materialize = mx.eval  # noqa: S307  — mx.eval materializes MLX tensors, not Python eval
        for i, block in enumerate(self.transformer_blocks):
            video_args, audio_args = block(video_args, audio_args)

            if self._eval_frequency > 0 and (i + 1) % self._eval_frequency == 0:
                if video_args is not None:
                    materialize(video_args.x)
                if audio_args is not None:
                    materialize(audio_args.x)

        # Store residuals for future steps
        if teacache is not None and video_input is not None and video_args is not None:
            teacache.store_residuals(
                video_input,
                video_args.x,
                audio_input,
                audio_args.x if audio_args is not None else None,
            )

        return video_args, audio_args

    def _process_video_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process video transformer output to velocity prediction."""
        scale_shift_values = (
            self.scale_shift_table[None, None, :, :]
            + embedded_timestep[:, :, None, :]
        )
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)
        return x

    def _process_audio_output(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
    ) -> mx.array:
        """Process audio transformer output to velocity prediction."""
        scale_shift_values = (
            self.audio_scale_shift_table[None, None, :, :]
            + embedded_timestep[:, :, None, :]
        )
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = self.audio_norm_out(x)
        x = x * (1 + scale) + shift
        x = self.audio_proj_out(x)
        return x

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass. Returns velocity predictions.

        Returns:
            VideoOnly: video_velocity
            AudioOnly: audio_velocity
            AudioVideo: (video_velocity, audio_velocity)
        """
        # Type casting
        if self.compute_dtype != mx.float32:
            if video is not None:
                video = Modality(
                    latent=video.latent.astype(self.compute_dtype),
                    context=video.context.astype(self.compute_dtype),
                    context_mask=video.context_mask,
                    timesteps=video.timesteps,
                    positions=video.positions,
                    enabled=video.enabled,
                    sigma=video.sigma,
                )
            if audio is not None:
                audio = Modality(
                    latent=audio.latent.astype(self.compute_dtype),
                    context=audio.context.astype(self.compute_dtype),
                    context_mask=audio.context_mask,
                    timesteps=audio.timesteps,
                    positions=audio.positions,
                    enabled=audio.enabled,
                    sigma=audio.sigma,
                )

        # Preprocessing
        video_args = None
        if self.model_type.is_video_enabled():
            if video is None:
                raise ValueError("Video modality required for video-enabled model")
            video_args = self._video_args_preprocessor.prepare(video)

        audio_args = None
        if self.model_type.is_audio_enabled():
            if audio is None:
                batch_size = video_args.x.shape[0] if video_args else 1
                audio = Modality(
                    latent=mx.zeros((batch_size, 0, self.audio_inner_dim)),
                    context=mx.zeros((batch_size, 0, self.audio_inner_dim)),
                    context_mask=None,
                    timesteps=mx.zeros((batch_size,)),
                    positions=mx.zeros((batch_size, 3, 0)),
                    enabled=False,
                )
            if audio.latent.size > 0:
                audio_args = self._audio_args_preprocessor.prepare(audio)
            else:
                audio_args = TransformerArgs(
                    x=mx.zeros(
                        (
                            video_args.x.shape[0] if video_args else 1,
                            0,
                            self.audio_inner_dim,
                        )
                    ),
                    context=mx.zeros((1, 0, self.audio_inner_dim)),
                    timesteps=mx.zeros((1, 0, 6, self.audio_inner_dim)),
                    positional_embeddings=(mx.zeros((1,)), mx.zeros((1,))),
                    enabled=False,
                )

        # Transformer blocks
        video_args, audio_args = self._process_transformer_blocks(
            video_args, audio_args
        )

        # Output processing
        video_out = None
        if self.model_type.is_video_enabled():
            video_out = self._process_video_output(
                video_args.x, video_args.embedded_timestep
            )
            if self.compute_dtype != mx.float32:
                video_out = video_out.astype(mx.float32)

        audio_out = None
        if self.model_type.is_audio_enabled():
            if audio_args.enabled and audio_args.x.size > 0:
                audio_out = self._process_audio_output(
                    audio_args.x, audio_args.embedded_timestep
                )
                if self.compute_dtype != mx.float32:
                    audio_out = audio_out.astype(mx.float32)
            else:
                current_batch_size = (
                    video_out.shape[0]
                    if video_out is not None
                    else (audio.latent.shape[0] if audio else 1)
                )
                audio_out = mx.zeros(
                    (current_batch_size, 0, self.AUDIO_OUT_CHANNELS)
                )

        # Return
        if self.model_type == LTXModelType.VideoOnly:
            return video_out
        elif self.model_type == LTXModelType.AudioOnly:
            return audio_out
        else:
            return video_out, audio_out


class X0Model(nn.Module):
    """Wrapper that converts velocity predictions to denoised outputs.

    x0 = latent - timestep * velocity
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def set_teacache(self, teacache) -> None:
        """Delegate to inner velocity model."""
        self.velocity_model.set_teacache(teacache)

    def set_step_info(self, step_idx: int, total_steps: int) -> None:
        """Delegate to inner velocity model."""
        self.velocity_model.set_step_info(step_idx, total_steps)

    def __call__(
        self,
        video: Optional[Modality] = None,
        audio: Optional[Modality] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """Compute denoised outputs from velocity predictions."""
        output = self.velocity_model(video, audio)

        def denoise(modality: Modality, velocity: mx.array) -> mx.array:
            timesteps = modality.timesteps
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None, None]
            elif timesteps.ndim == 2:
                timesteps = timesteps[:, :, None]
            return modality.latent - timesteps * velocity

        if isinstance(output, tuple):
            video_vel, audio_vel = output
            denoised_video = denoise(video, video_vel)
            if audio is not None and audio.latent.size > 0:
                denoised_audio = denoise(audio, audio_vel)
            else:
                return denoised_video
            return denoised_video, denoised_audio
        else:
            if video is not None:
                return denoise(video, output)
            elif audio is not None:
                return denoise(audio, output)
            return output


# Aliases
LTXAVModel = LTXModel
X0AVModel = X0Model
