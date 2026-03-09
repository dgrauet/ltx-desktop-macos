"""Two-phase text encoder subprocess for 32GB machines.

Encodes a text prompt into video and audio embeddings using a memory-efficient
two-phase approach that keeps peak GPU memory under 10GB:

Phase 1: Load 4-bit Gemma 3 12B (~6.2GB) → forward pass → save hidden states
          → delete LM → clear GPU cache
Phase 2: Load connector weights (~2.7GB for V1, ~6GB for V2) → compute features
          → connector forward → save final embeddings → exit

Supports both LTX-2.0 (V1) and LTX-2.3 (V2) models, auto-detected from config.json.

Usage::

    python -m engine.encode_text_subprocess \
        --prompt "A cat sitting on a mat" \
        --model-repo /path/to/model \
        --output /tmp/embeddings.npz \
        [--text-encoder-repo mlx-community/gemma-3-12b-it-4bit]

Output .npz contains (V2):
    - prompt_embeds: (1, 1024, 4096) bfloat16  (video)
    - prompt_attention_mask: (1, 1024) int32
    - audio_prompt_embeds: (1, 1024, 2048) bfloat16
    - audio_prompt_attention_mask: (1, 1024) int32

Output .npz contains (V1):
    - video_embeddings: (1, 1024, 3840) bfloat16
    - audio_embeddings: (1, 1024, 3840) bfloat16
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _detect_model_version(model_path: Path) -> str:
    """Detect model version from config.json.

    Returns:
        "2.3" for LTX-2.3, "2.0" for LTX-2.0.
    """
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        version = config.get("model_version", "")
        if version.startswith("2.3") or config.get("is_v2", False):
            return "2.3"
    return "2.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text prompt to embeddings")
    parser.add_argument("--prompt", required=True, help="Text prompt to encode")
    parser.add_argument("--model-repo", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--text-encoder-repo", default=None, help="Text encoder model path")
    args = parser.parse_args()

    model_path = Path(args.model_repo)
    version = _detect_model_version(model_path)
    log.info("Detected model version: %s", version)

    if version == "2.3":
        _encode_v2(args, model_path)
    else:
        _encode_v1(args, model_path)


def _resolve_text_encoder_path(args, model_path: Path) -> str:
    """Resolve the text encoder path from args or defaults."""
    text_encoder_path = args.text_encoder_repo or "google/gemma-3-12b-it"
    te_path_p = Path(str(text_encoder_path))
    if te_path_p.joinpath("text_encoder").is_dir():
        text_encoder_path = str(te_path_p / "text_encoder")
    if not Path(text_encoder_path).exists():
        from mlx_video.utils import get_model_path
        text_encoder_path = str(get_model_path(text_encoder_path))
    return text_encoder_path


def _run_gemma_forward(args, model_path: Path) -> tuple:
    """Phase 1: Run Gemma forward pass and return hidden states + attention mask.

    Returns:
        (all_hidden_states, attention_mask) — list of layer outputs and mask.
    """
    import mlx.core as mx

    text_encoder_path = _resolve_text_encoder_path(args, model_path)

    log.info("Phase 1: Loading language model from %s", text_encoder_path)
    from mlx_video.models.ltx.text_encoder import LanguageModel
    language_model = LanguageModel.from_pretrained(text_encoder_path)
    mx.eval(language_model.parameters())  # noqa: S307
    log.info("LM loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer_path = model_path / "tokenizer"
    if tokenizer_path.exists():
        processor = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    else:
        processor = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
    processor.padding_side = "left"

    inputs = processor(
        args.prompt,
        return_tensors="np",
        max_length=1024,
        truncation=True,
        padding="max_length",
    )
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])

    log.info("Running Gemma forward pass (1024 tokens)...")
    _, all_hidden_states = language_model(
        inputs=input_ids,
        input_embeddings=None,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # Materialize all hidden states before freeing LM
    for hs in all_hidden_states:
        mx.eval(hs)  # noqa: S307
    mx.eval(attention_mask)  # noqa: S307

    # Free LM
    del language_model, input_ids
    del processor, inputs
    gc.collect()
    mx.clear_cache()
    log.info("Phase 1 done, LM freed: %.2f GB active", mx.get_active_memory() / (1024**3))

    return all_hidden_states, attention_mask


def _encode_v2(args, model_path: Path) -> None:
    """LTX-2.3 text encoding pipeline.

    Uses vendored V2 text encoder with:
    - Per-token RMS normalization (not per-layer concat)
    - Dual aggregate embeddings (video: 4096-dim, audio: 2048-dim)
    - 8-layer gated-attention connectors (not 2-layer)
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Phase 1: Gemma forward pass
    all_hidden_states, attention_mask = _run_gemma_forward(args, model_path)

    # Phase 2: V2 Feature extraction + connectors
    log.info("Phase 2 (V2): Loading connectors from %s", model_path)

    from engine.ltx23_model.text_encoder import (
        AVTextEncoderV2,
        GemmaFeaturesExtractorV2,
    )
    from engine.ltx23_model.connector import Embeddings1DConnector

    # Build V2 text encoder
    feature_extractor = GemmaFeaturesExtractorV2(
        hidden_dim=3840,
        num_layers=49,
        video_inner_dim=4096,
        audio_inner_dim=2048,
    )
    video_connector = Embeddings1DConnector(
        attention_head_dim=128,
        num_attention_heads=32,
        num_layers=8,
        apply_gated_attention=True,
    )
    audio_connector = Embeddings1DConnector(
        attention_head_dim=64,
        num_attention_heads=32,
        num_layers=8,
        apply_gated_attention=True,
    )
    text_encoder = AVTextEncoderV2(
        feature_extractor=feature_extractor,
        video_connector=video_connector,
        audio_connector=audio_connector,
    )

    # Load connector weights
    connector_file = model_path / "connector.safetensors"
    if not connector_file.exists():
        raise FileNotFoundError(f"connector.safetensors not found in {model_path}")

    raw_weights = mx.load(str(connector_file))

    # Remap keys from safetensors format to our module structure
    mapped_weights: list[tuple[str, mx.array]] = []
    for k, v in raw_weights.items():
        if not k.startswith("connector."):
            continue
        new_k = k[len("connector."):]

        # text_embedding_projection → feature_extractor
        new_k = new_k.replace("text_embedding_projection.", "feature_extractor.")

        # video/audio connectors
        new_k = new_k.replace("video_embeddings_connector.", "video_connector.")
        new_k = new_k.replace("audio_embeddings_connector.", "audio_connector.")

        # Feed-forward remapping (PyTorch → our naming)
        new_k = new_k.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_k = new_k.replace(".ff.net.2.", ".ff.proj_out.")

        # Attention output remapping
        new_k = new_k.replace(".to_out.0.", ".to_out.")

        mapped_weights.append((new_k, v))

    del raw_weights

    text_encoder.load_weights(mapped_weights, strict=False)
    del mapped_weights
    mx.eval(text_encoder.parameters())  # noqa: S307
    log.info("V2 connectors loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    # Feature extraction — process in chunks to avoid Metal OOM on the
    # huge (1024, 188160) × (188160, 4096) matmul
    chunk_size = 128
    seq_len = all_hidden_states[0].shape[1]

    video_feature_chunks = []
    audio_feature_chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        # Slice each hidden state layer for this chunk
        chunk_hidden = [hs[:, start:end, :] for hs in all_hidden_states]
        chunk_mask = attention_mask[:, start:end]

        v_feat, a_feat = text_encoder.feature_extractor(
            chunk_hidden, chunk_mask, padding_side="left"
        )
        mx.eval(v_feat)  # noqa: S307
        mx.eval(a_feat)  # noqa: S307
        video_feature_chunks.append(v_feat)
        audio_feature_chunks.append(a_feat)
        del chunk_hidden, chunk_mask, v_feat, a_feat

    video_features = mx.concatenate(video_feature_chunks, axis=1)
    audio_features = mx.concatenate(audio_feature_chunks, axis=1)
    mx.eval(video_features)  # noqa: S307
    mx.eval(audio_features)  # noqa: S307
    del video_feature_chunks, audio_feature_chunks, all_hidden_states
    gc.collect()
    mx.clear_cache()
    log.info(
        "V2 features: video=%s audio=%s, active: %.2f GB",
        video_features.shape, audio_features.shape,
        mx.get_active_memory() / (1024**3),
    )

    # Run connectors using encode_from_projected (features already extracted)
    result = text_encoder.encode_from_projected(
        video_features, audio_features, attention_mask
    )
    mx.eval(result.video_encoding)  # noqa: S307
    mx.eval(result.audio_encoding)  # noqa: S307
    mx.eval(result.attention_mask)  # noqa: S307

    log.info(
        "V2 output: video=%s %s, audio=%s %s, mask=%s",
        result.video_encoding.shape, result.video_encoding.dtype,
        result.audio_encoding.shape, result.audio_encoding.dtype,
        result.attention_mask.shape,
    )

    # Save with keys matching generate_v23.py expectations
    # Cast to float32 before numpy conversion (bfloat16 not supported by numpy)
    import numpy as np
    np.savez(
        args.output,
        prompt_embeds=np.array(result.video_encoding.astype(mx.float32)),
        prompt_attention_mask=np.array(result.attention_mask.astype(mx.int32)),
        audio_prompt_embeds=np.array(result.audio_encoding.astype(mx.float32)),
        audio_prompt_attention_mask=np.array(result.attention_mask.astype(mx.int32)),
    )
    log.info("V2 embeddings saved to %s", args.output)
    log.info("Peak memory: %.2f GB", mx.get_peak_memory() / (1024**3))


def _encode_v1(args, model_path: Path) -> None:
    """LTX-2.0 text encoding pipeline (original V1 implementation)."""
    import mlx.core as mx

    text_encoder_path = _resolve_text_encoder_path(args, model_path)

    # Phase 1: Gemma forward pass
    log.info("Phase 1: Loading language model from %s", text_encoder_path)
    from mlx_video.models.ltx.text_encoder import LanguageModel
    language_model = LanguageModel.from_pretrained(text_encoder_path)
    mx.eval(language_model.parameters())  # noqa: S307
    log.info("LM loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    from transformers import AutoTokenizer
    tokenizer_path = model_path / "tokenizer"
    if tokenizer_path.exists():
        processor = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    else:
        processor = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
    processor.padding_side = "left"

    inputs = processor(
        args.prompt,
        return_tensors="np",
        max_length=1024,
        truncation=True,
        padding="max_length",
    )
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])

    log.info("Running Gemma forward pass (1024 tokens)...")
    _, all_hidden_states = language_model(
        inputs=input_ids,
        input_embeddings=None,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    from mlx_video.models.ltx.text_encoder import norm_and_concat_hidden_states
    concat_hidden = norm_and_concat_hidden_states(
        all_hidden_states, attention_mask, padding_side="left"
    )
    mx.eval(concat_hidden)  # noqa: S307
    log.info("Concat hidden states: %s (%.1f MB)",
             concat_hidden.shape,
             concat_hidden.nbytes / (1024**2))

    del language_model, all_hidden_states, input_ids
    del processor, inputs
    gc.collect()
    mx.clear_cache()
    log.info("Phase 1 done, LM freed: %.2f GB active", mx.get_active_memory() / (1024**3))

    # Phase 2: V1 connectors
    log.info("Phase 2: Loading connectors from %s", model_path)
    from mlx_video.models.ltx.text_encoder import (
        GemmaFeaturesExtractor,
        LTX2TextEncoder,
    )

    text_encoder = LTX2TextEncoder()

    connector_file = model_path / "connector.safetensors"
    if not connector_file.exists():
        raise FileNotFoundError(f"connector.safetensors not found in {model_path}")

    connector_weights = mx.load(str(connector_file))

    transformer_weights = {}
    for k, v in connector_weights.items():
        if k.startswith("connector."):
            new_k = "model.diffusion_model." + k[len("connector."):]
            transformer_weights[new_k] = v
        elif k.startswith("text_embedding_projection."):
            transformer_weights[k] = v
    del connector_weights

    if "text_embedding_projection.aggregate_embed.weight" in transformer_weights:
        text_encoder.feature_extractor.aggregate_embed.weight = transformer_weights[
            "text_embedding_projection.aggregate_embed.weight"
        ]

    connector_w = {}
    for key, value in transformer_weights.items():
        if key.startswith("model.diffusion_model.video_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "")
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            connector_w[new_key] = value
    if connector_w:
        text_encoder.video_embeddings_connector.load_weights(
            list(connector_w.items()), strict=False
        )

    audio_connector_w = {}
    for key, value in transformer_weights.items():
        if key.startswith("model.diffusion_model.audio_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.audio_embeddings_connector.", "")
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
            new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            audio_connector_w[new_key] = value
    if audio_connector_w:
        text_encoder.audio_embeddings_connector.load_weights(
            list(audio_connector_w.items()), strict=False
        )
    del transformer_weights, connector_w, audio_connector_w

    mx.eval(text_encoder.feature_extractor.parameters())  # noqa: S307
    mx.eval(text_encoder.video_embeddings_connector.parameters())  # noqa: S307
    mx.eval(text_encoder.audio_embeddings_connector.parameters())  # noqa: S307
    log.info("Connectors loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    chunk_size = 128
    seq_len = concat_hidden.shape[1]
    feature_chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = text_encoder.feature_extractor(concat_hidden[:, start:end, :])
        mx.eval(chunk)  # noqa: S307
        feature_chunks.append(chunk)
    features = mx.concatenate(feature_chunks, axis=1)
    mx.eval(features)  # noqa: S307
    del feature_chunks, concat_hidden
    gc.collect()
    mx.clear_cache()
    log.info("Features computed: %s (%.1f MB), active: %.2f GB",
             features.shape, features.nbytes / (1024**2),
             mx.get_active_memory() / (1024**3))

    additive_mask = (attention_mask - 1).astype(features.dtype)
    additive_mask = additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9

    video_embeddings, _ = text_encoder.video_embeddings_connector(features, additive_mask)
    mx.eval(video_embeddings)  # noqa: S307
    log.info("Video embeddings done, active: %.2f GB", mx.get_active_memory() / (1024**3))

    audio_embeddings, _ = text_encoder.audio_embeddings_connector(features, additive_mask)
    mx.eval(audio_embeddings)  # noqa: S307
    log.info("Audio embeddings done, active: %.2f GB", mx.get_active_memory() / (1024**3))

    log.info("Video embeddings: %s %s", video_embeddings.shape, video_embeddings.dtype)
    log.info("Audio embeddings: %s %s", audio_embeddings.shape, audio_embeddings.dtype)

    mx.savez(
        args.output,
        video_embeddings=video_embeddings,
        audio_embeddings=audio_embeddings,
    )
    log.info("Embeddings saved to %s", args.output)
    log.info("Peak memory: %.2f GB", mx.get_peak_memory() / (1024**3))


if __name__ == "__main__":
    main()
