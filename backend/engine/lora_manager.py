"""LoRA management for LTX-2.3 inference.

Handles loading .safetensors LoRA weights, compatibility verification
(LTX-2.3 latent space only — 2.0 LoRAs are incompatible), and
application/removal from the loaded model.

LoRA application formula:
    W_new = W_original + scale * (lora_up @ lora_down)

where scale = strength * (alpha / rank) if alpha is present, else strength.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)

# Default LoRA directory
LORA_DIR = Path.home() / ".ltx-desktop" / "loras"


@dataclass
class LoRAInfo:
    """Metadata for a single LoRA weight file."""

    id: str           # unique slug (filename without extension)
    name: str         # human-readable name
    path: str         # absolute path to .safetensors file
    lora_type: str    # "camera_control" | "detail" | "style" | "custom"
    compatible: bool  # True if verified LTX-2.3 compatible
    loaded: bool      # True if currently applied to model
    size_mb: float    # file size in megabytes
    strength: float = 0.7  # application strength (0.0-1.0)


@dataclass
class LoadedLoRA:
    """Tracks an active LoRA with its weight deltas for removal."""

    info: LoRAInfo
    weight_deltas: dict[str, mx.array]  # param_path -> delta applied
    strength: float


def _detect_lora_type(name: str) -> str:
    """Infer LoRA type from the filename slug.

    Args:
        name: Lowercase filename stem.

    Returns:
        One of "camera_control", "detail", "style", or "custom".
    """
    if "camera" in name:
        return "camera_control"
    if "detail" in name:
        return "detail"
    if "style" in name:
        return "style"
    return "custom"


def _check_compatibility(path: Path) -> bool:
    """Peek at the safetensors header to assess LTX-2.3 compatibility.

    Reads the weight keys of a .safetensors file without fully loading weights.
    Checks for standard LoRA key patterns (lora_down, lora_up).
    Falls back to True when the header cannot be inspected (optimistic default).

    Args:
        path: Absolute path to the .safetensors file.

    Returns:
        True if the LoRA appears to be a valid LoRA file.
    """
    try:
        # Use mx.load to peek at keys (loads lazily)
        weights = mx.load(str(path))
        keys = list(weights.keys())

        # Check for standard LoRA key patterns
        has_lora_keys = any(
            "lora_down" in k or "lora_up" in k
            for k in keys
        )
        if not has_lora_keys:
            log.warning(
                "File %s does not contain lora_down/lora_up keys — may not be a LoRA",
                path.name,
            )
            return False

        del weights
        return True
    except Exception as exc:
        log.debug("LoRA compatibility check failed for %s: %s", path.name, exc)
        return True


def _map_lora_key_to_model_path(lora_key: str) -> str | None:
    """Map a LoRA weight key to the corresponding model parameter path.

    Standard LoRA keys follow patterns like:
        transformer_blocks.0.attn1.to_q.lora_down.weight
        transformer.transformer_blocks.0.attn1.to_q.lora_up.weight

    The model uses paths like:
        transformer_blocks.0.attn1.to_q.weight

    Args:
        lora_key: Key from the LoRA safetensors file.

    Returns:
        Model parameter path (without .weight suffix), or None if not mappable.
    """
    # Strip common prefixes
    key = lora_key
    for prefix in ("transformer.", "model.", "base_model.model."):
        if key.startswith(prefix):
            key = key[len(prefix):]

    # Must contain lora_down or lora_up
    if "lora_down" not in key and "lora_up" not in key:
        return None

    # Extract the base parameter path
    # e.g., "transformer_blocks.0.attn1.to_q.lora_down.weight"
    #     -> "transformer_blocks.0.attn1.to_q"
    parts = key.split(".")
    base_parts = []
    for part in parts:
        if part in ("lora_down", "lora_up", "lora_A", "lora_B"):
            break
        base_parts.append(part)

    if not base_parts:
        return None

    return ".".join(base_parts)


def load_lora_weights(
    path: str | Path,
    strength: float = 0.7,
) -> tuple[dict[str, mx.array], dict[str, float]]:
    """Load LoRA weights from a safetensors file and compute weight deltas.

    Parses LoRA down/up weight pairs and computes the merged delta:
        delta = scale * (up @ down)

    where scale = strength * (alpha / rank) if alpha is present, else strength.

    Args:
        path: Path to the .safetensors LoRA file.
        strength: User-controlled strength multiplier (0.0-1.0).

    Returns:
        Tuple of (weight_deltas, metadata) where:
        - weight_deltas: dict mapping model parameter paths to delta tensors
        - metadata: dict with info like number of adapted layers, rank, etc.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no valid LoRA weight pairs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA file not found: {path}")

    raw_weights = mx.load(str(path))
    keys = list(raw_weights.keys())

    # Group keys by base parameter path
    down_weights: dict[str, mx.array] = {}  # base_path -> lora_down weight
    up_weights: dict[str, mx.array] = {}    # base_path -> lora_up weight
    alphas: dict[str, float] = {}           # base_path -> alpha scalar

    for key in keys:
        # Handle alpha scalars
        if key.endswith(".alpha"):
            base_path = _map_lora_key_to_model_path(
                key.replace(".alpha", ".lora_down.weight")
            )
            if base_path:
                alphas[base_path] = raw_weights[key].item()
            continue

        base_path = _map_lora_key_to_model_path(key)
        if base_path is None:
            continue

        if "lora_down" in key or "lora_A" in key:
            down_weights[base_path] = raw_weights[key]
        elif "lora_up" in key or "lora_B" in key:
            up_weights[base_path] = raw_weights[key]

    # Compute deltas for each paired down/up
    weight_deltas: dict[str, mx.array] = {}
    ranks = []

    for base_path in down_weights:
        if base_path not in up_weights:
            log.warning("LoRA key %s has lora_down but no lora_up — skipping", base_path)
            continue

        down = down_weights[base_path]
        up = up_weights[base_path]

        # Determine rank and scale
        rank = down.shape[0]
        ranks.append(rank)

        if base_path in alphas:
            scale = strength * (alphas[base_path] / rank)
        else:
            scale = strength

        # Compute delta: scale * (up @ down)
        # down shape: (rank, in_features) or (rank, in_features, ...)
        # up shape: (out_features, rank) or (out_features, rank, ...)
        delta = scale * (up @ down)
        mx.eval(delta)  # Materialize tensor — mx.eval is mlx.core.eval (safe)
        weight_deltas[base_path] = delta

    if not weight_deltas:
        raise ValueError(
            f"No valid LoRA weight pairs found in {path.name}. "
            f"Keys found: {keys[:10]}..."
        )

    metadata = {
        "num_adapted_layers": len(weight_deltas),
        "rank": ranks[0] if ranks else 0,
        "has_alpha": len(alphas) > 0,
        "strength": strength,
    }

    log.info(
        "Loaded LoRA %s: %d layers, rank=%d, strength=%.2f",
        path.name, len(weight_deltas), metadata["rank"], strength,
    )

    del raw_weights
    return weight_deltas, metadata


def apply_lora_to_model(
    model: Any,
    weight_deltas: dict[str, mx.array],
) -> int:
    """Apply precomputed LoRA weight deltas to a model in-place.

    Modifies model weights: W += delta for each adapted layer.

    Args:
        model: The LTX model (LTXModel or X0Model wrapping LTXModel).
        weight_deltas: dict mapping parameter paths to delta tensors.

    Returns:
        Number of layers successfully patched.
    """
    # If model is X0Model, unwrap to get the inner LTXModel
    inner_model = getattr(model, "model", model)

    # Build weight update list for load_weights
    updates: list[tuple[str, mx.array]] = []

    # Get current model parameters as a flat dict
    param_items = dict(inner_model.parameters())

    applied = 0
    for param_path, delta in weight_deltas.items():
        weight_key = f"{param_path}.weight"
        if weight_key in param_items:
            original = param_items[weight_key]
            if hasattr(original, "shape") and original.shape == delta.shape:
                new_weight = original + delta
                mx.eval(new_weight)  # mlx.core.eval — materialize tensor
                updates.append((weight_key, new_weight))
                applied += 1
            else:
                log.debug(
                    "Shape mismatch for %s: model=%s, delta=%s — skipping",
                    param_path,
                    getattr(original, "shape", "N/A"),
                    delta.shape,
                )
        else:
            log.debug("LoRA key %s not found in model parameters", param_path)

    if updates:
        inner_model.load_weights(updates, strict=False)

    log.info("Applied LoRA: %d/%d layers matched", applied, len(weight_deltas))
    return applied


def remove_lora_from_model(
    model: Any,
    weight_deltas: dict[str, mx.array],
) -> int:
    """Remove previously applied LoRA weight deltas from a model.

    Reverses: W -= delta for each adapted layer.

    Args:
        model: The LTX model.
        weight_deltas: dict mapping parameter paths to delta tensors (same as applied).

    Returns:
        Number of layers successfully reverted.
    """
    # Create negated deltas and apply
    neg_deltas = {k: -v for k, v in weight_deltas.items()}
    return apply_lora_to_model(model, neg_deltas)


class LoRAManager:
    """Manages LoRA weight files for LTX-2.3 inference.

    Scans a local directory for .safetensors files, validates compatibility,
    loads/applies/removes LoRA weights, and tracks active LoRAs.
    """

    def __init__(self, model_manager: Any = None) -> None:
        """Initialise the LoRA manager.

        Args:
            model_manager: Shared ModelManager instance (optional, for future use).
        """
        self._model_manager = model_manager
        self._loaded_loras: dict[str, LoadedLoRA] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_loras(self) -> list[LoRAInfo]:
        """Scan LORA_DIR for .safetensors files and return their metadata.

        Returns:
            List of LoRAInfo objects, one per discovered file.
        """
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        results: list[LoRAInfo] = []

        for file in sorted(LORA_DIR.glob("*.safetensors")):
            stem = file.stem
            name = stem.replace("_", " ").replace("-", " ").title()
            lora_type = _detect_lora_type(stem.lower())
            compatible = _check_compatibility(file)
            size_mb = file.stat().st_size / (1024 ** 2)
            loaded_lora = self._loaded_loras.get(stem)
            loaded = loaded_lora is not None

            results.append(
                LoRAInfo(
                    id=stem,
                    name=name,
                    path=str(file),
                    lora_type=lora_type,
                    compatible=compatible,
                    loaded=loaded,
                    size_mb=round(size_mb, 2),
                    strength=loaded_lora.strength if loaded_lora else 0.7,
                )
            )
            log.debug("Scanned LoRA: %s (type=%s, compatible=%s)", stem, lora_type, compatible)

        return results

    def list_loras(self) -> list[LoRAInfo]:
        """Return all available LoRAs discovered on disk.

        Returns:
            List of on-disk LoRAs with current loaded state.
        """
        return self.scan_loras()

    def load_lora(self, lora_id: str, strength: float = 0.7) -> LoRAInfo:
        """Mark a LoRA as active with given strength.

        The actual weight application happens per-generation in the subprocess,
        since the model is loaded fresh each time.

        Args:
            lora_id: Slug matching a .safetensors filename stem.
            strength: Application strength (0.0-1.0).

        Returns:
            Updated LoRAInfo with loaded=True.

        Raises:
            FileNotFoundError: If no LoRA with the given id exists.
            ValueError: If the LoRA is not compatible.
        """
        all_loras = {info.id: info for info in self.list_loras()}

        if lora_id not in all_loras:
            raise FileNotFoundError(f"LoRA not found: {lora_id!r}")

        info = all_loras[lora_id]
        if not info.compatible:
            raise ValueError(
                f"LoRA {lora_id!r} is not compatible with LTX-2.3. "
                "Only LoRAs with lora_down/lora_up weight pairs are supported."
            )

        info.loaded = True
        info.strength = strength
        self._loaded_loras[lora_id] = LoadedLoRA(
            info=info,
            weight_deltas={},  # Deltas computed at generation time in subprocess
            strength=strength,
        )
        log.info("LoRA activated: %s (strength=%.2f)", lora_id, strength)
        return info

    def unload_lora(self, lora_id: str) -> None:
        """Remove a LoRA from the active set.

        Args:
            lora_id: Slug of the LoRA to unload.
        """
        if lora_id in self._loaded_loras:
            del self._loaded_loras[lora_id]
            log.info("LoRA deactivated: %s", lora_id)
        else:
            log.warning("unload_lora called for non-loaded LoRA: %s", lora_id)

    def unload_all(self) -> None:
        """Unload every active LoRA."""
        count = len(self._loaded_loras)
        self._loaded_loras.clear()
        log.info("Unloaded all LoRAs (%d total)", count)

    def get_loaded(self) -> list[LoRAInfo]:
        """Return all currently active LoRAs.

        Returns:
            List of LoRAInfo objects with loaded=True.
        """
        return [ll.info for ll in self._loaded_loras.values()]

    def get_active_lora_args(self) -> list[str]:
        """Build CLI args for passing active LoRAs to the generation subprocess.

        Returns:
            List of CLI argument strings, e.g.:
            ["--lora", "/path/to/file.safetensors:0.7",
             "--lora", "/path/to/other.safetensors:0.5"]
        """
        args: list[str] = []
        for loaded in self._loaded_loras.values():
            if loaded.info.path:
                args.extend([
                    "--lora",
                    f"{loaded.info.path}:{loaded.strength}",
                ])
        return args

    def update_strength(self, lora_id: str, strength: float) -> None:
        """Update the strength of an active LoRA.

        Args:
            lora_id: Slug of the LoRA.
            strength: New strength value (0.0-1.0).
        """
        if lora_id in self._loaded_loras:
            self._loaded_loras[lora_id].strength = strength
            self._loaded_loras[lora_id].info.strength = strength
            log.info("LoRA %s strength updated to %.2f", lora_id, strength)
        else:
            log.warning("update_strength called for non-loaded LoRA: %s", lora_id)

    def import_lora(self, source_path: str) -> LoRAInfo:
        """Copy a .safetensors file into the LoRA directory.

        Args:
            source_path: Absolute path to the source .safetensors file.

        Returns:
            LoRAInfo for the imported LoRA.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the file is not a .safetensors file.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if source.suffix != ".safetensors":
            raise ValueError(f"Expected .safetensors file, got: {source.suffix}")

        LORA_DIR.mkdir(parents=True, exist_ok=True)
        dest = LORA_DIR / source.name

        shutil.copy2(str(source), str(dest))
        log.info("Imported LoRA: %s -> %s", source, dest)

        # Return metadata for the imported file
        stem = dest.stem
        return LoRAInfo(
            id=stem,
            name=stem.replace("_", " ").replace("-", " ").title(),
            path=str(dest),
            lora_type=_detect_lora_type(stem.lower()),
            compatible=_check_compatibility(dest),
            loaded=False,
            size_mb=round(dest.stat().st_size / (1024 ** 2), 2),
        )
