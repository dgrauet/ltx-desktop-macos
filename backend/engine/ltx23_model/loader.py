"""Weight loader for LTX-2.3 split/quantized MLX models.

Loads transformer weights from our converted format (split safetensors,
int8/int4 quantized transformer blocks, bf16 non-quantizable layers).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .model import LTXModel, LTXModelType, X0Model
from .rope import LTXRopeType

log = logging.getLogger(__name__)

# Prefix in weight files → strip before loading into model
_WEIGHT_PREFIX = "transformer."


def load_config(model_dir: Path) -> dict:
    """Load model config from config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    with open(config_path) as f:
        return json.load(f)


def create_model_from_config(config: dict, low_memory: bool = True) -> LTXModel:
    """Create an LTXModel instance from config dict.

    Args:
        config: Model configuration from config.json.
        low_memory: If True, use aggressive memory optimization (eval every 4 layers).
    """
    type_map = {
        "AudioVideo": LTXModelType.AudioVideo,
        "VideoOnly": LTXModelType.VideoOnly,
        "AudioOnly": LTXModelType.AudioOnly,
    }
    model_type_str = config.get("model_type", "AudioVideo")
    model_type = type_map.get(model_type_str, LTXModelType.AudioVideo)

    return LTXModel(
        model_type=model_type,
        num_attention_heads=config.get("num_attention_heads", 32),
        attention_head_dim=config.get("attention_head_dim", 128),
        in_channels=config.get("in_channels", 128),
        out_channels=config.get("out_channels", 128),
        num_layers=config.get("num_layers", 48),
        cross_attention_dim=config.get("cross_attention_dim", 4096),
        audio_cross_attention_dim=config.get("audio_cross_attention_dim"),
        norm_eps=config.get("norm_eps", 1e-6),
        caption_channels=config.get("caption_channels"),
        positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
        positional_embedding_max_pos=config.get("positional_embedding_max_pos"),
        timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
        av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1),
        use_middle_indices_grid=config.get("use_middle_indices_grid", True),
        rope_type=LTXRopeType.SPLIT,
        compute_dtype=mx.float32,
        low_memory=low_memory,
        cross_attention_adaln=config.get("cross_attention_adaln", False),
        apply_gated_attention=config.get("apply_gated_attention", False),
    )


def _apply_quantization(model: LTXModel, quant_config: dict) -> None:
    """Apply quantization to model layers based on quantize_config.json.

    Only quantizes transformer_blocks layers (Linear → QuantizedLinear).
    Non-quantizable layers (adaln, proj_out, patchify_proj, etc.) stay bf16.
    """
    bits = quant_config.get("bits", 8)
    group_size = quant_config.get("group_size", 64)
    only_blocks = quant_config.get("only_transformer_blocks", True)

    exclude_audio = quant_config.get("exclude_audio", False)

    if only_blocks:
        # Only quantize nn.Linear modules within transformer_blocks
        def class_predicate(path: str, module: nn.Module) -> bool:
            if not isinstance(module, nn.Linear) or "transformer_blocks" not in path:
                return False
            if exclude_audio and "audio" in path:
                return False
            return True

        nn.quantize(model, bits=bits, group_size=group_size, class_predicate=class_predicate)
    else:
        nn.quantize(model, bits=bits, group_size=group_size)

    log.info(f"Applied {bits}-bit quantization (group_size={group_size})")


def load_transformer_weights(
    model: LTXModel,
    model_dir: Path,
    strict: bool = True,
) -> LTXModel:
    """Load transformer weights from split safetensors into the model.

    Handles quantization: reads quantize_config.json, converts appropriate
    nn.Linear layers to nn.QuantizedLinear, then loads weights.

    Args:
        model: LTXModel instance to load weights into.
        model_dir: Path to the model directory containing safetensors files.
        strict: If True, raise on missing/unexpected keys.

    Returns:
        The model with loaded weights.
    """
    transformer_path = model_dir / "transformer.safetensors"
    if not transformer_path.exists():
        raise FileNotFoundError(f"No transformer.safetensors at {transformer_path}")

    # Apply quantization config if present (converts Linear → QuantizedLinear)
    quant_config_path = model_dir / "quantize_config.json"
    if quant_config_path.exists():
        with open(quant_config_path) as f:
            quant_data = json.load(f)
        quant_config = quant_data.get("quantization", quant_data)
        _apply_quantization(model, quant_config)

    log.info(f"Loading transformer weights from {transformer_path}")

    # Load raw weights
    raw_weights = mx.load(str(transformer_path))

    # Strip 'transformer.' prefix
    weights = {}
    for key, value in raw_weights.items():
        if key.startswith(_WEIGHT_PREFIX):
            new_key = key[len(_WEIGHT_PREFIX):]
            weights[new_key] = value
        else:
            weights[key] = value

    # Load into model
    model.load_weights(list(weights.items()), strict=strict)
    log.info(f"Loaded {len(weights)} transformer weight tensors")

    return model


def load_ltx23_transformer(
    model_dir: str | Path,
    low_memory: bool = True,
    as_x0: bool = True,
) -> LTXModel | X0Model:
    """Load a complete LTX-2.3 transformer from a model directory.

    Args:
        model_dir: Path to model directory with config.json and transformer.safetensors.
        low_memory: Use aggressive memory management (eval every 4 layers).
        as_x0: If True, wrap in X0Model for denoised output.

    Returns:
        LTXModel or X0Model with loaded weights.
    """
    model_dir = Path(model_dir)

    config = load_config(model_dir)
    log.info(
        f"LTX-2.3 model: version={config.get('model_version', 'unknown')}, "
        f"type={config.get('model_type', 'unknown')}, "
        f"layers={config.get('num_layers', '?')}, "
        f"v2={config.get('is_v2', False)}"
    )

    model = create_model_from_config(config, low_memory=low_memory)
    model = load_transformer_weights(model, model_dir)

    # NOTE: mx.compile() disabled — subprocess-per-generation means tracing
    # overhead is paid every invocation and compiled kernels are lost on exit.
    # Would only help with a persistent model server architecture.

    if as_x0:
        return X0Model(model)
    return model
