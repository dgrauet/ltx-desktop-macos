"""Two-phase text encoder subprocess for 32GB machines.

Encodes a text prompt into video and audio embeddings using a memory-efficient
two-phase approach that keeps peak GPU memory under 10GB:

Phase 1: Load 4-bit Gemma 3 12B (~6.2GB) → forward pass → save hidden states
          → delete LM → clear GPU cache
Phase 2: Load connector weights (~6GB) → compute features
          → connector forward → save final embeddings → exit

Usage::

    python -m engine.encode_text_subprocess \
        --prompt "A cat sitting on a mat" \
        --model-repo /path/to/model \
        --output /tmp/embeddings.npz \
        [--text-encoder-repo mlx-community/gemma-3-12b-it-4bit]

Output .npz contains:
    - prompt_embeds: (1, 1024, 4096) bfloat16  (video)
    - prompt_attention_mask: (1, 1024) int32
    - audio_prompt_embeds: (1, 1024, 2048) bfloat16
    - audio_prompt_attention_mask: (1, 1024) int32
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _materialize(tensor) -> None:
    """Materialize an MLX lazy tensor (calls mlx.core.eval, NOT Python eval)."""
    import mlx.core as mx
    mx.eval(tensor)  # noqa: S307 — this is mlx.core.eval (tensor materialization), not Python eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text prompt to embeddings")
    parser.add_argument("--prompt", required=True, help="Text prompt to encode")
    parser.add_argument("--model-repo", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output .npz file path")
    parser.add_argument("--text-encoder-repo", default=None, help="Text encoder model path")
    args = parser.parse_args()

    model_path = Path(args.model_repo)
    _encode(args, model_path)


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
    _materialize(language_model.parameters())
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
        _materialize(hs)
    _materialize(attention_mask)

    # Free LM
    del language_model, input_ids
    del processor, inputs
    gc.collect()
    mx.clear_cache()
    log.info("Phase 1 done, LM freed: %.2f GB active", mx.get_active_memory() / (1024**3))

    return all_hidden_states, attention_mask


def _encode(args, model_path: Path) -> None:
    """LTX-2.3 text encoding pipeline.

    Uses vendored V2 text encoder with:
    - Per-token RMS normalization (not per-layer concat)
    - Dual aggregate embeddings (video: 4096-dim, audio: 2048-dim)
    - 8-layer gated-attention connectors
    """
    import mlx.core as mx

    # Phase 1: Gemma forward pass
    all_hidden_states, attention_mask = _run_gemma_forward(args, model_path)

    # Phase 2: Feature extraction + connectors
    log.info("Phase 2: Loading connectors from %s", model_path)

    from engine.ltx23_model.text_encoder import (
        AVTextEncoderV2,
        GemmaFeaturesExtractorV2,
    )
    from engine.ltx23_model.connector import Embeddings1DConnector

    # Build text encoder
    feature_extractor = GemmaFeaturesExtractorV2(
        hidden_dim=3840,
        num_layers=49,
        video_inner_dim=4096,
        audio_inner_dim=2048,
    )
    from engine.ltx23_model.rope import LTXRopeType
    video_connector = Embeddings1DConnector(
        attention_head_dim=128,
        num_attention_heads=32,
        num_layers=8,
        apply_gated_attention=True,
        positional_embedding_max_pos=[4096],
        rope_type=LTXRopeType.SPLIT,
    )
    audio_connector = Embeddings1DConnector(
        attention_head_dim=64,
        num_attention_heads=32,
        num_layers=8,
        apply_gated_attention=True,
        positional_embedding_max_pos=[4096],
        rope_type=LTXRopeType.SPLIT,
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
    _materialize(text_encoder.parameters())
    log.info("Connectors loaded: %.2f GB active", mx.get_active_memory() / (1024**3))

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
        _materialize(v_feat)
        _materialize(a_feat)
        video_feature_chunks.append(v_feat)
        audio_feature_chunks.append(a_feat)
        del chunk_hidden, chunk_mask, v_feat, a_feat

    video_features = mx.concatenate(video_feature_chunks, axis=1)
    audio_features = mx.concatenate(audio_feature_chunks, axis=1)
    _materialize(video_features)
    _materialize(audio_features)
    del video_feature_chunks, audio_feature_chunks, all_hidden_states
    gc.collect()
    mx.clear_cache()
    log.info(
        "Features: video=%s audio=%s, active: %.2f GB",
        video_features.shape, audio_features.shape,
        mx.get_active_memory() / (1024**3),
    )

    # Run connectors using encode_from_projected (features already extracted)
    result = text_encoder.encode_from_projected(
        video_features, audio_features, attention_mask
    )
    _materialize(result.video_encoding)
    _materialize(result.audio_encoding)
    _materialize(result.attention_mask)

    log.info(
        "Output: video=%s %s, audio=%s %s, mask=%s",
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
    log.info("Embeddings saved to %s", args.output)
    log.info("Peak memory: %.2f GB", mx.get_peak_memory() / (1024**3))


if __name__ == "__main__":
    main()
