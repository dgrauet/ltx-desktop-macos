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
_quantize_config: dict | None = None  # Set by apply_patch from quantize_config.json

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
    Also loads quantize_config.json on first call to configure bit width.
    """
    global _quantize_config
    if _quantize_config is None:
        config_path = model_path / "quantize_config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                raw = json.load(f)
            _quantize_config = raw.get("quantization", {})
            log.info("Loaded quantize config: bits=%d, group_size=%d",
                     _quantize_config.get("bits", 0),
                     _quantize_config.get("group_size", 0))

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
    """Check if weight list contains quantized keys (.scales/.biases).

    Returns quantization config dict or None. Reads bits/group_size from
    the module-level _quantize_config (loaded from quantize_config.json).
    """
    has_scales = any(k.endswith(".scales") for k, _ in weight_items)
    has_biases = any(k.endswith(".biases") for k, _ in weight_items)
    if has_scales and has_biases:
        if _quantize_config:
            return {
                "bits": _quantize_config.get("bits", 4),
                "group_size": _quantize_config.get("group_size", 64),
            }
        return {"bits": 4, "group_size": 64}
    return None


def _patched_load_weights(self, weights, strict=False):
    """Patched load_weights that handles quantized weights.

    Only quantizes transformer_blocks (the bulk of the model). Other layers
    like adaln_single, caption_projection, proj_out, patchify_proj are kept
    in bf16 for precision — these are small but critical for correct output.
    """
    weight_items = weights if isinstance(weights, list) else list(weights)
    qconfig = _has_quantized_weights(weight_items)

    if qconfig is not None:
        bits = qconfig["bits"]
        group_size = qconfig["group_size"]
        log.info(
            "Detected quantized weights (bits=%d, group_size=%d). "
            "Converting transformer_blocks to QuantizedLinear...",
            bits,
            group_size,
        )

        # Only quantize Linear layers inside transformer_blocks — other layers
        # (adaln, caption_projection, proj_out, patchify_proj) stay in bf16
        # to avoid precision loss in these critical components.
        def _block_only_predicate(path: str, module: nn.Module) -> bool:
            return isinstance(module, nn.Linear) and "transformer_blocks" in path

        nn.quantize(
            self,
            bits=bits,
            group_size=group_size,
            class_predicate=_block_only_predicate,
        )
        log.info("Transformer blocks converted to QuantizedLinear")

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
            # Instead, we pre-materialize only the language model params (~6GB)
            # which is safe. The full parameters() call materializes ~9GB at once
            # (LM + connectors) which causes Metal OOM.
            self._skip_bulk_eval = True
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
        """Materialize only language model params instead of everything.

        generate_av.py calls mx.eval(text_encoder.parameters()) after load(),
        which materializes ~9GB at once (LM + connectors) causing Metal OOM
        on 32GB machines. Instead, we materialize only the language model
        (~6GB) inside this method and return empty dict so the outer
        mx.eval() is a no-op. Connectors materialize lazily during forward pass.
        """
        if getattr(self, "_skip_bulk_eval", False):
            log.info("Materializing only language model params (32GB mode)")
            self._skip_bulk_eval = False  # Only skip once
            # mx.eval here is mlx.core.eval (tensor materialization)
            mx.eval(self.language_model.parameters())  # noqa: S307
            log.info("Language model params materialized: %.2f GB",
                     mx.get_active_memory() / (1024**3))
            return {}  # outer mx.eval() gets empty dict = no-op
        return _original_te_parameters(self)

    LTX2TextEncoder.parameters = _patched_te_parameters
    log.info("Text encoder parameters() patch applied (skip bulk eval)")

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

    # Patch load_vae_encoder for I2V — custom loader for split models.
    # The original loader uses tree_unflatten which converts numeric keys
    # to list indices, but VideoEncoder.down_blocks is a dict with integer
    # keys. We bypass this by loading weights manually with list→dict fixup.
    from mlx_video.models.ltx.video_vae import encoder as encoder_mod
    _original_load_vae_encoder = encoder_mod.load_vae_encoder

    def _patched_load_vae_encoder(weights_path: str, use_unified: bool = False):
        """Patched VAE encoder loader for split models.

        Handles two issues with the original loader:
        1. Split model: loads from vae_encoder.safetensors instead of model.safetensors
        2. tree_unflatten bug: converts numeric keys to lists, but VideoEncoder
           uses dicts with integer keys for down_blocks. We fix this with a
           custom weight application that converts lists back to dicts.
        """
        path = Path(weights_path)
        is_split = use_unified and path.is_dir() and not (path / "model.safetensors").exists()

        if not is_split:
            return _original_load_vae_encoder(weights_path, use_unified=use_unified)

        vae_file = path / "vae_encoder.safetensors"
        if not vae_file.exists():
            return _original_load_vae_encoder(weights_path, use_unified=use_unified)

        log.info("Loading VAE encoder from vae_encoder.safetensors (custom loader)")

        # 1. Read config from safetensors metadata
        import json as _json
        from safetensors import safe_open
        from mlx_video.models.ltx.video_vae.video_vae import (
            VideoEncoder, LogVarianceType, NormLayerType, PaddingModeType,
        )

        encoder_blocks = []
        norm_layer = NormLayerType.PIXEL_NORM
        latent_log_var = LogVarianceType.UNIFORM
        patch_size = 4

        try:
            with safe_open(str(vae_file), framework="numpy") as f:
                metadata = f.metadata()
                if metadata and "config" in metadata:
                    configs = _json.loads(metadata["config"])
                    vae_config = configs.get("vae", {})
                    for block in vae_config.get("encoder_blocks", []):
                        if isinstance(block, list) and len(block) == 2:
                            encoder_blocks.append((block[0], block[1]))
                    norm_str = vae_config.get("norm_layer", "pixel_norm")
                    norm_layer = (
                        NormLayerType.PIXEL_NORM if norm_str == "pixel_norm"
                        else NormLayerType.GROUP_NORM
                    )
                    var_str = vae_config.get("latent_log_var", "uniform")
                    latent_log_var = {
                        "uniform": LogVarianceType.UNIFORM,
                        "per_channel": LogVarianceType.PER_CHANNEL,
                        "constant": LogVarianceType.CONSTANT,
                    }.get(var_str, LogVarianceType.NONE)
                    patch_size = vae_config.get("patch_size", 4)
        except Exception as e:
            log.warning("Could not read encoder config from metadata: %s", e)

        if not encoder_blocks:
            encoder_blocks = [
                ("res_x", {"num_layers": 4}),
                ("compress_space_res", {"multiplier": 2}),
                ("res_x", {"num_layers": 6}),
                ("compress_time_res", {"multiplier": 2}),
                ("res_x", {"num_layers": 6}),
                ("compress_all_res", {"multiplier": 2}),
                ("res_x", {"num_layers": 2}),
                ("compress_all_res", {"multiplier": 2}),
                ("res_x", {"num_layers": 2}),
            ]

        log.info("Encoder config: %d blocks, patch_size=%d", len(encoder_blocks), patch_size)

        # 2. Create encoder
        encoder = VideoEncoder(
            convolution_dimensions=3,
            in_channels=3,
            out_channels=128,
            encoder_blocks=encoder_blocks,
            patch_size=patch_size,
            norm_layer=norm_layer,
            latent_log_var=latent_log_var,
            encoder_spatial_padding_mode=PaddingModeType.ZEROS,
        )

        # 3. Load weights from split file
        weights = mx.load(str(vae_file))
        log.info("Loaded %d weight keys from vae_encoder.safetensors", len(weights))

        # 4. Strip vae_encoder. prefix and load per-channel stats
        prefix = "vae_encoder."
        encoder_weights = {}
        for key, value in weights.items():
            if not key.startswith(prefix):
                continue
            new_key = key[len(prefix):]

            # Load per-channel statistics directly onto the encoder
            if new_key == "per_channel_statistics._mean_of_means":
                encoder.per_channel_statistics.mean = value
                continue
            if new_key == "per_channel_statistics._std_of_means":
                encoder.per_channel_statistics.std = value
                continue

            # Skip per_channel_statistics sub-keys (handled above)
            if new_key.startswith("per_channel_statistics."):
                continue

            # NO Conv3d transpose — split files are already in MLX format
            encoder_weights[new_key] = value

        del weights
        log.info("Prepared %d encoder weight keys (prefix stripped)", len(encoder_weights))

        # 5. Apply weights using tree_unflatten + list→dict fixup.
        # tree_unflatten converts numeric keys (down_blocks.0.xxx) to list
        # indices, but VideoEncoder.down_blocks is a dict{int: Module}.
        # We fix this by converting lists back to dicts at the down_blocks level.
        from mlx.utils import tree_unflatten

        nested = tree_unflatten(list(encoder_weights.items()))

        def _lists_to_dicts(tree):
            """Recursively convert lists to dicts with integer keys."""
            if isinstance(tree, list):
                return {i: _lists_to_dicts(v) for i, v in enumerate(tree)}
            if isinstance(tree, dict):
                return {k: _lists_to_dicts(v) for k, v in tree.items()}
            return tree

        if "down_blocks" in nested and isinstance(nested["down_blocks"], list):
            nested["down_blocks"] = {
                i: _lists_to_dicts(v) for i, v in enumerate(nested["down_blocks"])
            }
            log.info("Converted down_blocks from list to dict (%d blocks)", len(nested["down_blocks"]))

        encoder.update(nested)
        log.info("VAE encoder loaded successfully (%d blocks)", len(encoder.down_blocks))
        return encoder

    encoder_mod.load_vae_encoder = _patched_load_vae_encoder
    gen_mod.load_vae_encoder = _patched_load_vae_encoder
    log.info("Split model VAE encoder patch applied")

    # Patch text encoder to skip loading entirely when pre-computed embeddings
    # are available via LTX_PRECOMPUTED_EMBEDDINGS env var. This saves ~10GB
    # of text encoder memory in the generation subprocess.
    import os

    def _has_precomputed_embeddings() -> str | None:
        path = os.environ.get("LTX_PRECOMPUTED_EMBEDDINGS")
        if path and Path(path).exists():
            return path
        return None

    # Wrap load() to skip when precomputed embeddings exist
    _wrapped_te_load = LTX2TextEncoder.load

    def _precomputed_te_load(self, *args, **kwargs):
        if _has_precomputed_embeddings():
            log.info("Skipping text encoder load — using pre-computed embeddings")
            self._using_precomputed = True
            return
        return _wrapped_te_load(self, *args, **kwargs)

    LTX2TextEncoder.load = _precomputed_te_load

    # Wrap parameters() to return {} when precomputed
    _wrapped_te_parameters = LTX2TextEncoder.parameters

    def _precomputed_te_parameters(self):
        if getattr(self, "_using_precomputed", False):
            return {}
        return _wrapped_te_parameters(self)

    LTX2TextEncoder.parameters = _precomputed_te_parameters

    # Wrap __call__() to load from file when precomputed
    _original_te_call = LTX2TextEncoder.__call__

    def _patched_te_call(self, *args, **kwargs):
        emb_path = _has_precomputed_embeddings()
        if emb_path:
            log.info("Loading pre-computed embeddings from %s", emb_path)
            data = mx.load(emb_path)
            video_emb = data["video_embeddings"]
            audio_emb = data["audio_embeddings"]
            log.info("Pre-computed video=%s audio=%s", video_emb.shape, audio_emb.shape)
            return video_emb, audio_emb
        return _original_te_call(self, *args, **kwargs)

    LTX2TextEncoder.__call__ = _patched_te_call
    log.info("Pre-computed embeddings patch applied to text encoder")
