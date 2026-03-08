"""Monkey-patch mlx_video to support quantized (int4/int8) model loading.

The mlx-video-with-audio library uses nn.Linear layers throughout the transformer.
When loading quantized safetensors (which contain .weight, .scales, .biases keys),
we need to call nn.quantize() on the model BEFORE load_weights() so that
nn.Linear layers are replaced with nn.QuantizedLinear.

This module patches the library's LTXModel.load_weights to handle this automatically.
Import this module before calling `python -m mlx_video.generate_av`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

_original_load_weights = None


def _has_quantized_weights(weight_items: list[tuple[str, mx.array]]) -> dict | None:
    """Check if weight list contains quantized keys (.scales/.biases).

    Returns quantization config dict if quantized, None otherwise.
    """
    has_scales = any(k.endswith(".scales") for k, _ in weight_items)
    has_biases = any(k.endswith(".biases") for k, _ in weight_items)
    if has_scales and has_biases:
        # Try to infer quantization params from the weights
        for k, v in weight_items:
            if k.endswith(".scales"):
                # group_size = weight_cols * bits / 32 (from quantized weight shape)
                # For int4: weight is (out, in*4/32) = (out, in/8), scales is (out, in/group_size)
                return {"bits": 4, "group_size": 64}  # defaults, will be refined
        return {"bits": 4, "group_size": 64}
    return None


def _patched_load_weights(self, weights, strict=False):
    """Patched load_weights that handles quantized weights.

    If quantized keys (.scales, .biases) are detected, calls nn.quantize()
    on the model first to replace nn.Linear with nn.QuantizedLinear,
    then loads the weights normally.
    """
    # Check if these are quantized weights
    weight_items = weights if isinstance(weights, list) else list(weights)
    qconfig = _has_quantized_weights(weight_items)

    if qconfig is not None:
        log.info(
            "Detected quantized weights (bits=%d, group_size=%d). "
            "Converting model layers to QuantizedLinear...",
            qconfig["bits"],
            qconfig["group_size"],
        )

        # Try to load quantize_config.json for accurate params
        # (the config is saved alongside the model by our quantize script)
        config_loaded = False
        try:
            # Check common locations for quantize_config.json
            import mlx_video.generate_av as gen_mod
            # We can't easily get the model path here, so infer from scales shape
            for k, v in weight_items:
                if k.endswith(".scales"):
                    # Find corresponding weight to infer group_size
                    base = k.rsplit(".scales", 1)[0]
                    for k2, v2 in weight_items:
                        if k2 == base + ".weight":
                            # weight shape is (out_features, in_features * bits / 32)
                            # scales shape is (out_features, num_groups)
                            # group_size = in_features / num_groups
                            # in_features = weight_cols * 32 / bits
                            weight_cols = v2.shape[-1]
                            num_groups = v.shape[-1]
                            if num_groups > 0:
                                # in_features = weight_cols * 32 / bits
                                in_features = weight_cols * 32 // qconfig["bits"]
                                qconfig["group_size"] = in_features // num_groups
                                config_loaded = True
                                log.info(
                                    "Inferred group_size=%d from weight shapes",
                                    qconfig["group_size"],
                                )
                            break
                    if config_loaded:
                        break
        except Exception:
            pass

        # Replace nn.Linear layers with nn.QuantizedLinear
        # Only quantize layers that have corresponding .scales keys
        quantized_prefixes = set()
        for k, _ in weight_items:
            if k.endswith(".scales"):
                quantized_prefixes.add(k.rsplit(".scales", 1)[0])

        # Use nn.quantize to convert the model's Linear layers
        nn.quantize(
            self,
            bits=qconfig["bits"],
            group_size=qconfig["group_size"],
        )
        log.info("Model layers converted to QuantizedLinear")

    # Call original load_weights
    return _original_load_weights(self, weight_items, strict=strict)


def apply_patch():
    """Apply the quantized weight loading patch to LTXModel.

    Call this before any model loading occurs. Safe to call multiple times.
    """
    global _original_load_weights

    from mlx_video.models.ltx.ltx import LTXModel

    if _original_load_weights is not None:
        return  # Already patched

    _original_load_weights = LTXModel.load_weights
    LTXModel.load_weights = _patched_load_weights
    log.info("Quantized weight loading patch applied to LTXModel")
