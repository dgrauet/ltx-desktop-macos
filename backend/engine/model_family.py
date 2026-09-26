"""LTX model family detection (2.3 vs 2.5) and per-family feature capabilities.

Detection mirrors ``ltx_pipelines_mlx.utils.generation.is_ltx25_pack`` — a 2.5
pack declares ``ff_bias: false`` in its transformer config, 2.3 packs omit it —
but reads the JSON directly so the API process never imports MLX.
"""

from __future__ import annotations

import json
from pathlib import Path

FAMILY_23 = "2.3"
FAMILY_25 = "2.5"

_CONFIG_FILES = ("embedded_config.json", "config.json")

# Feature availability per family (ltx-2-mlx 0.15.9):
# - TeaCache coefficients are calibrated for 2.3 only (the lib raises on 2.5).
# - 2.5 has no IC-LoRAs yet and the trainer is 2.3-only, so every LoRA the app
#   can import or train targets 2.3 (different latent space — never apply to 2.5).
# - auto-duration (DurationHead), generated keyframe slots and the diffusion
#   video decoder only exist on 2.5 packs.
# - prompt enhancement runs a standalone Gemma 3 in the app, so it works for both.
_CAPABILITIES: dict[str, dict[str, bool]] = {
    FAMILY_23: {
        "enhance": True,
        "loras": True,
        "ic_lora": True,
        "training": True,
        "teacache": True,
        "auto_duration": False,
        "generated_keyframes": False,
        "diffusion_decoder": False,
    },
    FAMILY_25: {
        "enhance": True,
        "loras": False,
        "ic_lora": False,
        "training": False,
        "teacache": False,
        "auto_duration": True,
        "generated_keyframes": True,
        "diffusion_decoder": True,
    },
}


def detect_family(model_dir: str | Path) -> str:
    """Return ``"2.5"`` for an LTX-2.5 pack directory, ``"2.3"`` otherwise.

    Unreadable or missing configs fall back to 2.3, like the lib's loader.
    """
    directory = Path(model_dir)
    for name in _CONFIG_FILES:
        path = directory / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        transformer = config.get("transformer", config)
        if isinstance(transformer, dict) and transformer.get("ff_bias") is False:
            return FAMILY_25
        return FAMILY_23
    return FAMILY_23


def family_from_repo_name(repo: str) -> str:
    """Best-effort family for a repo id that is not on disk yet."""
    return FAMILY_25 if "ltx-2.5" in repo.lower() else FAMILY_23


def capabilities(family: str) -> dict[str, bool]:
    """Feature flags for a model family (copy — safe to mutate)."""
    return dict(_CAPABILITIES.get(family, _CAPABILITIES[FAMILY_23]))
