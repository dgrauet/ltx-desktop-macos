"""Two-phase text encoder subprocess for 32GB machines.

Encodes a text prompt into video and audio embeddings using a memory-efficient
two-phase approach that keeps peak GPU memory under 10GB:

Phase 1: Load 4-bit Gemma 3 12B (~6.2GB) → forward pass → save hidden states
          → delete LM → clear GPU cache
Phase 2: Load connector weights (~2.7GB) → compute features → connector
          forward → save final embeddings → exit

This avoids loading LM + connectors simultaneously (~9GB), which OOMs on
32GB machines when other processes (backend server, macOS UI) use GPU memory.

Usage::

    python -m engine.encode_text_subprocess \
        --prompt "A cat sitting on a mat" \
        --model-repo /path/to/model \
        --output /tmp/embeddings.npz \
        [--text-encoder-repo mlx-community/gemma-3-12b-it-4bit]

Output .npz contains:
    - video_embeddings: (1, 1024, 3840) bfloat16
    - audio_embeddings: (1, 1024, 3840) bfloat16
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text prompt to embeddings")
    parser.add_argument("--prompt", required=True, help="Text prompt to encode")
    parser.add_argument("--model-repo", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--text-encoder-repo", default=None, help="Text encoder model path")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn

    model_path = Path(args.model_repo)

    # Resolve text encoder path
    text_encoder_path = args.text_encoder_repo or "google/gemma-3-12b-it"
    te_path_p = Path(str(text_encoder_path))
    if te_path_p.joinpath("text_encoder").is_dir():
        text_encoder_path = str(te_path_p / "text_encoder")
    if not Path(text_encoder_path).exists():
        from mlx_video.utils import get_model_path
        text_encoder_path = str(get_model_path(text_encoder_path))

    # ----------------------------------------------------------------
    # Phase 1: Gemma forward pass (~6.2GB peak)
    # ----------------------------------------------------------------
    log.info("Phase 1: Loading language model from %s", text_encoder_path)

    from mlx_video.models.ltx.text_encoder import LanguageModel
    language_model = LanguageModel.from_pretrained(text_encoder_path)
    # mx.eval materializes lazy MLX tensors into Metal GPU memory
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

    # Compute norm_and_concat immediately while hidden states are fresh,
    # then free the individual hidden states
    from mlx_video.models.ltx.text_encoder import norm_and_concat_hidden_states
    concat_hidden = norm_and_concat_hidden_states(
        all_hidden_states, attention_mask, padding_side="left"
    )
    mx.eval(concat_hidden)  # noqa: S307 - Force materialization before freeing LM
    log.info("Concat hidden states: %s (%.1f MB)",
             concat_hidden.shape,
             concat_hidden.nbytes / (1024**2))

    # Free LM — this is the critical step that makes phase 2 fit
    del language_model, all_hidden_states, input_ids
    del processor, inputs
    gc.collect()
    mx.clear_cache()
    log.info("Phase 1 done, LM freed: %.2f GB active", mx.get_active_memory() / (1024**3))

    # ----------------------------------------------------------------
    # Phase 2: Connector forward pass (~3GB peak)
    # ----------------------------------------------------------------
    log.info("Phase 2: Loading connectors from %s", model_path)

    # Create the text encoder module (initializes connector architectures)
    from mlx_video.models.ltx.text_encoder import (
        GemmaFeaturesExtractor,
        LTX2TextEncoder,
    )

    # Build a lightweight text encoder with only the connector components
    # (no language model — we already have concat_hidden from phase 1)
    text_encoder = LTX2TextEncoder()

    # Load connector weights from split file
    connector_file = model_path / "connector.safetensors"
    if not connector_file.exists():
        raise FileNotFoundError(f"connector.safetensors not found in {model_path}")

    connector_weights = mx.load(str(connector_file))

    # Map connector weight keys to the expected format
    transformer_weights = {}
    for k, v in connector_weights.items():
        if k.startswith("connector."):
            new_k = "model.diffusion_model." + k[len("connector."):]
            transformer_weights[new_k] = v
        elif k.startswith("text_embedding_projection."):
            transformer_weights[k] = v
    del connector_weights

    # Load feature extractor weight
    if "text_embedding_projection.aggregate_embed.weight" in transformer_weights:
        text_encoder.feature_extractor.aggregate_embed.weight = transformer_weights[
            "text_embedding_projection.aggregate_embed.weight"
        ]

    # Load video connector weights
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

    # Load audio connector weights
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

    # Materialize connector weights
    mx.eval(text_encoder.feature_extractor.parameters())  # noqa: S307
    mx.eval(text_encoder.video_embeddings_connector.parameters())  # noqa: S307
    mx.eval(text_encoder.audio_embeddings_connector.parameters())  # noqa: S307
    log.info("Connectors loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

    # Compute features — the feature_extractor is a huge matmul
    # (1024, 188160) × (188160, 3840). Process in chunks to avoid
    # Metal temp buffer spikes.
    chunk_size = 128
    seq_len = concat_hidden.shape[1]
    feature_chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = text_encoder.feature_extractor(concat_hidden[:, start:end, :])
        mx.eval(chunk)  # noqa: S307 - Force materialization per chunk
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

    # Save to .npz
    mx.savez(
        args.output,
        video_embeddings=video_embeddings,
        audio_embeddings=audio_embeddings,
    )
    log.info("Embeddings saved to %s", args.output)
    log.info("Peak memory: %.2f GB", mx.get_peak_memory() / (1024**3))


if __name__ == "__main__":
    main()
