"""LTX-2.3 transformer model for MLX.

Vendored from Acelogic/LTX-2-MLX (Apache-2.0), adapted for our
split/quantized weight format.

Key naming differences from upstream:
- ff.proj_in (ours) vs ff.project_in.proj (upstream)
- ff.proj_out (ours) vs ff.project_out (upstream)
- linear1/linear2 (ours) vs linear_1/linear_2 (upstream)
"""

from .model import (
    LTXModel,
    LTXModelType,
    Modality,
    PixArtAlphaTextProjection,
    TransformerArgsPreprocessor,
    MultiModalTransformerArgsPreprocessor,
    X0Model,
)
from .transformer import (
    BasicAVTransformerBlock,
    BasicTransformerBlock,
    TransformerArgs,
    TransformerConfig,
)
from .attention import Attention, RMSNorm, rms_norm
from .feed_forward import FeedForward
from .rope import LTXRopeType, apply_rotary_emb, precompute_freqs_cis
from .timestep_embedding import AdaLayerNormSingle
from .loader import load_ltx23_transformer, load_config, create_model_from_config
from .text_encoder import AVTextEncoderV2, GemmaFeaturesExtractorV2, TextEncoderOutput
from .connector import Embeddings1DConnector, BasicTransformerBlock1D
from .vae_decoder import VideoDecoder, load_vae_decoder, decode_video, streaming_decode_to_ffmpeg
from .vae_encoder import VideoEncoder, load_vae_encoder, encode_image
from .audio_decoder import AudioDecoder, load_audio_decoder
from .vocoder import Vocoder, VocoderWithBWE, load_vocoder

__all__ = [
    "LTXModel",
    "LTXModelType",
    "Modality",
    "X0Model",
    "TransformerArgs",
    "TransformerConfig",
    "BasicAVTransformerBlock",
    "Attention",
    "FeedForward",
    "AdaLayerNormSingle",
    "LTXRopeType",
    "load_ltx23_transformer",
    "load_config",
    "create_model_from_config",
    "AVTextEncoderV2",
    "GemmaFeaturesExtractorV2",
    "TextEncoderOutput",
    "Embeddings1DConnector",
    "VideoDecoder",
    "load_vae_decoder",
    "decode_video",
    "streaming_decode_to_ffmpeg",
    "VideoEncoder",
    "load_vae_encoder",
    "encode_image",
    "AudioDecoder",
    "load_audio_decoder",
    "Vocoder",
    "VocoderWithBWE",
    "load_vocoder",
]
