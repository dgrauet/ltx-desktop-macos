"""V2 text encoder for LTX-2.3.

Handles the Gemma hidden state → feature extraction → connector pipeline.
Uses separate video (4096-dim) and audio (2048-dim) projections.

Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .connector import Embeddings1DConnector


def norm_and_concat_per_token_rms(
    encoded_text: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    """Per-token RMS normalization for V2 models.

    Args:
        encoded_text: [B, T, D, L] stacked hidden states.
        attention_mask: [B, T] binary mask.

    Returns:
        [B, T, D*L] normalized tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape
    variance = mx.mean(encoded_text ** 2, axis=2, keepdims=True)
    normed = encoded_text * mx.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.astype(mx.bool_)[:, :, None]
    return mx.where(mask_3d, normed, mx.zeros_like(normed))


class GemmaFeaturesExtractorV2(nn.Module):
    """V2 feature extractor for LTX-2.3.

    Uses per-token RMS normalization and dual aggregate embeddings
    that project directly to transformer-native dimensions.

    video_aggregate_embed: (188160 → 4096) with bias
    audio_aggregate_embed: (188160 → 2048) with bias
    """

    def __init__(
        self,
        hidden_dim: int = 3840,
        num_layers: int = 49,
        video_inner_dim: int = 4096,
        audio_inner_dim: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        flat_dim = hidden_dim * num_layers
        self.embedding_dim = hidden_dim

        self.video_aggregate_embed = nn.Linear(flat_dim, video_inner_dim, bias=True)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_inner_dim, bias=True)

    def __call__(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> tuple:
        """Extract features from Gemma hidden states.

        Args:
            hidden_states: List of 49 tensors, each [B, T, 3840].
            attention_mask: [B, T] binary mask.

        Returns:
            (video_features, audio_features) each [B, T, dim].
        """
        stacked = mx.stack(hidden_states, axis=-1)  # [B, T, D, L]
        normed = norm_and_concat_per_token_rms(stacked, attention_mask)
        normed = normed.astype(stacked.dtype)

        v_dim = self.video_aggregate_embed.weight.shape[0]
        a_dim = self.audio_aggregate_embed.weight.shape[0]

        video_features = self.video_aggregate_embed(
            normed * math.sqrt(v_dim / self.embedding_dim)
        )
        audio_features = self.audio_aggregate_embed(
            normed * math.sqrt(a_dim / self.embedding_dim)
        )
        return video_features, audio_features


@dataclass
class TextEncoderOutput:
    """Output from the V2 text encoder."""
    video_encoding: mx.array     # [B, T, 4096]
    audio_encoding: mx.array     # [B, T, 2048]
    attention_mask: mx.array     # [B, T] binary


class AVTextEncoderV2(nn.Module):
    """Audio+Video text encoder for LTX-2.3.

    Pipeline: Gemma hidden states → V2 feature extractor → separate connectors.
    """

    def __init__(
        self,
        feature_extractor: Optional[GemmaFeaturesExtractorV2] = None,
        video_connector: Optional[Embeddings1DConnector] = None,
        audio_connector: Optional[Embeddings1DConnector] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor or GemmaFeaturesExtractorV2()
        self.video_connector = video_connector or Embeddings1DConnector(
            attention_head_dim=128, num_attention_heads=32,
            num_layers=8, apply_gated_attention=True,
        )
        self.audio_connector = audio_connector or Embeddings1DConnector(
            attention_head_dim=64, num_attention_heads=32,
            num_layers=8, apply_gated_attention=True,
        )

    def _convert_to_additive_mask(
        self,
        attention_mask: mx.array,
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """Convert binary mask to additive mask for softmax."""
        if dtype == mx.float16:
            large_value = 65504.0
        elif dtype == mx.bfloat16:
            large_value = 3.38e38
        else:
            large_value = 3.40e38
        additive_mask = (attention_mask.astype(dtype) - 1) * large_value
        return additive_mask.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])

    def encode_from_hidden_states(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> TextEncoderOutput:
        """Encode text from pre-computed Gemma hidden states.

        Args:
            hidden_states: List of 49 tensors from Gemma layers, each [B, T, 3840].
            attention_mask: Binary attention mask [B, T].

        Returns:
            TextEncoderOutput with separate video and audio encodings.
        """
        video_input, audio_input = self.feature_extractor(
            hidden_states, attention_mask, padding_side
        )

        connector_mask = self._convert_to_additive_mask(attention_mask, video_input.dtype)

        video_encoded, output_mask = self.video_connector(video_input, connector_mask)
        binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
        video_encoded = video_encoded * binary_mask[:, :, None]

        audio_encoded, _ = self.audio_connector(audio_input, connector_mask)

        return TextEncoderOutput(
            video_encoding=video_encoded,
            audio_encoding=audio_encoded,
            attention_mask=binary_mask,
        )

    def encode_from_projected(
        self,
        video_features: mx.array,
        audio_features: mx.array,
        attention_mask: mx.array,
    ) -> TextEncoderOutput:
        """Encode from already-projected features (after feature extractor).

        Use when feature extraction was done separately (e.g., chunked).
        """
        connector_mask = self._convert_to_additive_mask(attention_mask, video_features.dtype)

        video_encoded, output_mask = self.video_connector(video_features, connector_mask)
        binary_mask = (output_mask.squeeze(1).squeeze(1) >= -0.5).astype(mx.int32)
        video_encoded = video_encoded * binary_mask[:, :, None]

        audio_encoded, _ = self.audio_connector(audio_features, connector_mask)

        return TextEncoderOutput(
            video_encoding=video_encoded,
            audio_encoding=audio_encoded,
            attention_mask=binary_mask,
        )

    def __call__(
        self,
        hidden_states: List[mx.array],
        attention_mask: mx.array,
        padding_side: str = "left",
    ) -> TextEncoderOutput:
        return self.encode_from_hidden_states(hidden_states, attention_mask, padding_side)
