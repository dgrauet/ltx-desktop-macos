"""Monkey-patch mlx_video for quantized + split model loading.

Two patches are applied:

1. **LTXModel.load_weights** — detects quantized weights (.scales/.biases)
   and calls ``nn.quantize()`` to replace ``nn.Linear`` with ``nn.QuantizedLinear``
   before loading.

2. **load_unified_weights** — loads from per-component split files
   (e.g. ``transformer.safetensors``) instead of the monolithic
   ``model.safetensors``. This avoids loading the entire 15GB file into
   memory when only a 10GB component is needed — critical for 32GB machines.

Import this module before calling ``mlx_video.generate_av``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

_original_load_weights = None
_original_load_unified = None

# Map weight prefix → split filename
_PREFIX_TO_FILE = {
    "transformer.": "transformer.safetensors",
    "connector.": "connector.safetensors",
    "text_embedding_projection.": "connector.safetensors",
    "vae_decoder.": "vae_decoder.safetensors",
    "vae_encoder.": "vae_encoder.safetensors",
    "vocoder.": "vocoder.safetensors",
    "audio_vae.": "audio_vae.safetensors",
}


# ---------------------------------------------------------------------------
# Patch 1: load_unified_weights → load from split files
# ---------------------------------------------------------------------------

def _patched_load_unified(model_path: Path, prefix: str) -> dict:
    """Load weights from the correct split file instead of model.safetensors.

    Falls back to the original monolithic loading if split files don't exist.
    """
    split_file = _PREFIX_TO_FILE.get(prefix)
    if split_file:
        split_path = model_path / split_file
        if split_path.exists():
            log.info("Loading split file: %s (prefix=%s)", split_file, prefix)
            weights = mx.load(str(split_path))
            return {
                k[len(prefix):]: v
                for k, v in weights.items()
                if k.startswith(prefix)
            }

    # Fallback to original (monolithic file)
    log.info("Falling back to monolithic load for prefix=%s", prefix)
    return _original_load_unified(model_path, prefix)


# ---------------------------------------------------------------------------
# Patch 2: LTXModel.load_weights → handle quantized weights
# ---------------------------------------------------------------------------

def _has_quantized_weights(weight_items: list[tuple[str, mx.array]]) -> dict | None:
    """Check if weight list contains quantized keys (.scales/.biases)."""
    has_scales = any(k.endswith(".scales") for k, _ in weight_items)
    has_biases = any(k.endswith(".biases") for k, _ in weight_items)
    if has_scales and has_biases:
        return {"bits": 4, "group_size": 64}
    return None


def _patched_load_weights(self, weights, strict=False):
    """Patched load_weights that handles quantized weights.

    If quantized keys (.scales, .biases) are detected, calls nn.quantize()
    on the model first to replace nn.Linear with nn.QuantizedLinear,
    then loads the weights normally.
    """
    weight_items = weights if isinstance(weights, list) else list(weights)
    qconfig = _has_quantized_weights(weight_items)

    if qconfig is not None:
        # Infer group_size from weight shapes
        for k, v in weight_items:
            if k.endswith(".scales"):
                base = k.rsplit(".scales", 1)[0]
                for k2, v2 in weight_items:
                    if k2 == base + ".weight":
                        weight_cols = v2.shape[-1]
                        num_groups = v.shape[-1]
                        if num_groups > 0:
                            in_features = weight_cols * 32 // qconfig["bits"]
                            qconfig["group_size"] = in_features // num_groups
                        break
                break

        log.info(
            "Detected quantized weights (bits=%d, group_size=%d). "
            "Converting model layers to QuantizedLinear...",
            qconfig["bits"],
            qconfig["group_size"],
        )
        nn.quantize(self, bits=qconfig["bits"], group_size=qconfig["group_size"])
        log.info("Model layers converted to QuantizedLinear")

    return _original_load_weights(self, weight_items, strict=strict)


# ---------------------------------------------------------------------------
# apply_patch — call once before any model loading
# ---------------------------------------------------------------------------

def apply_patch():
    """Apply quantized + split model patches. Safe to call multiple times."""
    global _original_load_weights, _original_load_unified

    if _original_load_weights is not None:
        return  # Already patched

    # Patch LTXModel.load_weights
    from mlx_video.models.ltx.ltx import LTXModel
    _original_load_weights = LTXModel.load_weights
    LTXModel.load_weights = _patched_load_weights
    log.info("Quantized weight loading patch applied to LTXModel")

    # Patch load_unified_weights
    import mlx_video.generate_av as gen_mod
    _original_load_unified = gen_mod.load_unified_weights
    gen_mod.load_unified_weights = _patched_load_unified
    log.info("Split model loading patch applied to load_unified_weights")

    # Patch is_unified_mlx_model to recognize split models
    _original_is_unified = gen_mod.is_unified_mlx_model

    def _patched_is_unified(model_path: Path) -> bool:
        model_path = Path(model_path)
        if (model_path / "split_model.json").exists():
            return True
        if (model_path / "transformer.safetensors").exists() and (model_path / "config.json").exists():
            return True
        return _original_is_unified(model_path)

    gen_mod.is_unified_mlx_model = _patched_is_unified
    log.info("Split model detection patch applied to is_unified_mlx_model")

    # Patch text encoder .load() to use split connector file
    from mlx_video.models.ltx.text_encoder import LTX2TextEncoder
    _original_te_load = LTX2TextEncoder.load

    def _patched_te_load(self, model_path=None, text_encoder_path="google/gemma-3-12b-it", use_unified=False):
        """Patched text encoder load — reads from connector.safetensors instead of model.safetensors."""
        model_path_p = Path(model_path) if model_path else None

        # If split model exists, load connector weights from split file
        if (
            use_unified
            and model_path_p
            and (model_path_p / "connector.safetensors").exists()
            and not (model_path_p / "model.safetensors").exists()
        ):
            log.info("Loading text encoder + connector from split files")

            # 1. Load the language model
            te_path_p = Path(str(text_encoder_path))
            if te_path_p.joinpath("text_encoder").is_dir():
                text_encoder_path = str(te_path_p / "text_encoder")

            from mlx_video.models.ltx.text_encoder import LanguageModel
            # Resolve HF repo ID to local path if needed
            te_resolved = text_encoder_path
            if not Path(te_resolved).exists():
                from mlx_video.utils import get_model_path
                te_resolved = str(get_model_path(te_resolved))
            self.language_model = LanguageModel.from_pretrained(te_resolved)

            # 2. Load connector weights directly
            connector_weights = mx.load(str(model_path_p / "connector.safetensors"))

            # Build transformer_weights dict matching the expected format
            transformer_weights = {}
            for k, v in connector_weights.items():
                if k.startswith("connector."):
                    new_k = "model.diffusion_model." + k[len("connector."):]
                    transformer_weights[new_k] = v
                elif k.startswith("text_embedding_projection."):
                    transformer_weights[k] = v
            del connector_weights

            # 3. Load feature extractor
            if "text_embedding_projection.aggregate_embed.weight" in transformer_weights:
                self.feature_extractor.aggregate_embed.weight = transformer_weights[
                    "text_embedding_projection.aggregate_embed.weight"
                ]

            # 4. Load video connector weights
            connector_w = {}
            for key, value in transformer_weights.items():
                if key.startswith("model.diffusion_model.video_embeddings_connector."):
                    new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "")
                    new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
                    new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
                    new_key = new_key.replace(".to_out.0.", ".to_out.")
                    connector_w[new_key] = value
            if connector_w:
                self.video_embeddings_connector.load_weights(list(connector_w.items()), strict=False)

            # 5. Load audio connector weights
            audio_connector_w = {}
            for key, value in transformer_weights.items():
                if key.startswith("model.diffusion_model.audio_embeddings_connector."):
                    new_key = key.replace("model.diffusion_model.audio_embeddings_connector.", "")
                    new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
                    new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
                    new_key = new_key.replace(".to_out.0.", ".to_out.")
                    audio_connector_w[new_key] = value
            if audio_connector_w:
                self.audio_embeddings_connector.load_weights(list(audio_connector_w.items()), strict=False)

            # 6. Load tokenizer
            from transformers import AutoTokenizer
            tokenizer_path = model_path_p / "tokenizer"
            if tokenizer_path.exists():
                self.processor = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
            else:
                te_resolved = text_encoder_path
                if not Path(te_resolved).exists():
                    from mlx_video.utils import get_model_path
                    te_resolved = str(get_model_path(te_resolved))
                self.processor = AutoTokenizer.from_pretrained(te_resolved, trust_remote_code=True)
            self.processor.padding_side = "left"

            log.info("Text encoder loaded successfully (split model)")
            # Mark to skip the next mx.eval(text_encoder.parameters()) call
            # that generate_av.py does after load(). This prevents OOM on 32GB.
            self._skip_bulk_eval = True
            # Reduce max_length from 1024 to 256 to avoid Metal OOM during forward pass
            self._use_reduced_max_length = True
            import gc
            gc.collect()
            mx.clear_cache()
            return

        return _original_te_load(self, model_path=model_path, text_encoder_path=text_encoder_path, use_unified=use_unified)

    LTX2TextEncoder.load = _patched_te_load
    log.info("Split model text encoder patch applied")

    # Patch parameters() to skip bulk mx.eval on 32GB machines
    _original_te_parameters = LTX2TextEncoder.parameters

    def _patched_te_parameters(self):
        """Skip bulk parameter materialization when _skip_bulk_eval is set.

        generate_av.py calls mx.eval(text_encoder.parameters()) after load(),
        which materializes ~9GB at once and causes Metal OOM on 32GB machines.
        When _skip_bulk_eval is True, return an empty dict so the bulk eval
        is a no-op. Parameters will materialize lazily during the forward pass.
        """
        if getattr(self, "_skip_bulk_eval", False):
            log.info("Skipping bulk parameter eval for text encoder (32GB mode)")
            self._skip_bulk_eval = False  # Only skip once
            return {}
        return _original_te_parameters(self)

    LTX2TextEncoder.parameters = _patched_te_parameters
    log.info("Text encoder parameters() patch applied (skip bulk eval)")

    # Patch encode() to use shorter max_length on 32GB machines.
    # Default max_length=1024 creates massive activation tensors during
    # Gemma 3 12B forward pass, causing Metal OOM on 32GB.
    # max_length=256 fits comfortably and is enough for video prompts (<200 words).
    _original_te_encode = LTX2TextEncoder.encode

    def _patched_te_encode(self, prompt, max_length=1024, return_audio_embeddings=True):
        if getattr(self, "_use_reduced_max_length", False) and max_length > 256:
            log.info("Reducing text encoder max_length from %d to 256 (32GB mode)", max_length)
            max_length = 256
        return _original_te_encode(self, prompt, max_length=max_length, return_audio_embeddings=return_audio_embeddings)

    LTX2TextEncoder.encode = _patched_te_encode
    log.info("Text encoder max_length patch applied (256 for 32GB)")

    # Patch load_upsampler to handle split models (no model.safetensors)
    from mlx_video.models.ltx import upsampler as upsampler_mod
    _original_load_upsampler = upsampler_mod.load_upsampler

    def _patched_load_upsampler(weights_path: str, use_unified: bool = False):
        """Patched upsampler loader — downloads from Lightricks for split models."""
        path = Path(weights_path)
        if use_unified and path.is_dir() and not (path / "model.safetensors").exists():
            log.info("Split model: downloading upsampler from Lightricks (~1GB)")
            from huggingface_hub import hf_hub_download
            upsampler_file = hf_hub_download(
                repo_id="Lightricks/LTX-2",
                filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            )
            return _original_load_upsampler(upsampler_file, use_unified=False)
        return _original_load_upsampler(weights_path, use_unified=use_unified)

    upsampler_mod.load_upsampler = _patched_load_upsampler
    # Also patch the local reference in generate_av (imported via `from ... import`)
    gen_mod.load_upsampler = _patched_load_upsampler
    log.info("Split model upsampler patch applied")

    # Patch load_vae_decoder similarly
    from mlx_video.models.ltx.video_vae import decoder as decoder_mod
    _original_load_vae_decoder = decoder_mod.load_vae_decoder

    def _patched_load_vae_decoder(weights_path: str, timestep_conditioning=None, use_unified: bool = False):
        """Patched VAE decoder loader — uses split vae_decoder.safetensors."""
        path = Path(weights_path)
        if use_unified and path.is_dir() and not (path / "model.safetensors").exists():
            vae_file = path / "vae_decoder.safetensors"
            if vae_file.exists():
                log.info("Loading VAE decoder from vae_decoder.safetensors")
                # Load via the unified path but with a temp symlink
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    # Create model.safetensors containing only vae_decoder weights
                    vae_weights = mx.load(str(vae_file))
                    mx.save_safetensors(str(tmp_path / "model.safetensors"), vae_weights)
                    del vae_weights
                    return _original_load_vae_decoder(str(tmp_path), timestep_conditioning=timestep_conditioning, use_unified=True)
        return _original_load_vae_decoder(weights_path, timestep_conditioning=timestep_conditioning, use_unified=use_unified)

    decoder_mod.load_vae_decoder = _patched_load_vae_decoder
    # Also patch the local reference in generate_av
    gen_mod.load_vae_decoder = _patched_load_vae_decoder
    log.info("Split model VAE decoder patch applied")
