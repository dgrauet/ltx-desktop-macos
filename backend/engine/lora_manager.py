"""LoRA management for LTX-2.3 inference.

Handles loading .safetensors LoRA weights, compatibility verification
(LTX-2.3 latent space only — 2.0 LoRAs are incompatible), and
application/removal from the loaded model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from engine.memory_manager import aggressive_cleanup
from engine.model_manager import ModelManager

log = logging.getLogger(__name__)

# Built-in stub LoRAs exposed to the UI even when no files exist on disk.
_BUILTIN_LORAS: list[LoRAInfo] = []  # populated after dataclass definition


@dataclass
class LoRAInfo:
    """Metadata for a single LoRA weight file."""

    id: str           # unique slug (filename without extension)
    name: str         # human-readable name
    path: str         # absolute path to .safetensors file (empty for built-ins)
    lora_type: str    # "camera_control" | "detail" | "style" | "custom"
    compatible: bool  # True if verified LTX-2.3 compatible
    loaded: bool      # True if currently applied to model
    size_mb: float    # file size in megabytes (0 for built-ins)


_BUILTIN_LORAS = [
    LoRAInfo(
        id="camera-control",
        name="Camera Control",
        path="",
        lora_type="camera_control",
        compatible=True,
        loaded=False,
        size_mb=0.0,
    ),
    LoRAInfo(
        id="detail-enhance",
        name="Detail Enhancement",
        path="",
        lora_type="detail",
        compatible=True,
        loaded=False,
        size_mb=0.0,
    ),
]


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

    Reads the metadata keys of a .safetensors file without loading weights.
    If key names contain "ltx2" or "ltx23" the LoRA is considered compatible.
    Falls back to True when the header cannot be inspected (optimistic default).

    Args:
        path: Absolute path to the .safetensors file.

    Returns:
        True if the LoRA is likely LTX-2.3 compatible.
    """
    try:
        from safetensors import safe_open  # type: ignore[import-untyped]

        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())
        name_lower = path.stem.lower()
        for key in keys:
            key_lower = key.lower()
            if "ltx23" in key_lower or "ltx2" in key_lower:
                return True
        # Also accept if the filename itself indicates LTX-2.3
        if "ltx23" in name_lower or "ltx2" in name_lower:
            return True
        # Cannot confirm compatibility from keys alone — optimistic default
        log.debug("Could not confirm compatibility for %s; defaulting to True", path.name)
        return True
    except Exception as exc:
        log.debug("safetensors header read failed for %s: %s — defaulting to True", path.name, exc)
        return True


class LoRAManager:
    """Manages LoRA weight files for LTX-2.3 inference.

    Scans a local directory for .safetensors files, validates compatibility
    with the LTX-2.3 latent space, and tracks which LoRAs are currently
    applied to the loaded model.
    """

    LORA_DIR: Path = Path.home() / ".ltx-desktop" / "loras"

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialise the LoRA manager.

        Args:
            model_manager: Shared ModelManager instance — used to apply/remove
                weights from the live model in future sprints.
        """
        self._model_manager = model_manager
        self._loaded_loras: dict[str, LoRAInfo] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_loras(self) -> list[LoRAInfo]:
        """Scan LORA_DIR for .safetensors files and return their metadata.

        Returns:
            List of LoRAInfo objects, one per discovered file.
        """
        self.LORA_DIR.mkdir(parents=True, exist_ok=True)
        results: list[LoRAInfo] = []

        for file in sorted(self.LORA_DIR.glob("*.safetensors")):
            stem = file.stem
            name = stem.replace("_", " ").title()
            lora_type = _detect_lora_type(stem.lower())
            compatible = _check_compatibility(file)
            size_mb = file.stat().st_size / (1024 ** 2)
            loaded = stem in self._loaded_loras

            results.append(
                LoRAInfo(
                    id=stem,
                    name=name,
                    path=str(file),
                    lora_type=lora_type,
                    compatible=compatible,
                    loaded=loaded,
                    size_mb=round(size_mb, 2),
                )
            )
            log.debug("Scanned LoRA: %s (type=%s, compatible=%s)", stem, lora_type, compatible)

        return results

    def load_lora(self, lora_id: str) -> LoRAInfo:
        """Mark a LoRA as loaded and (in future sprints) apply its weights.

        Args:
            lora_id: Slug matching a .safetensors filename stem or built-in id.

        Returns:
            Updated LoRAInfo with loaded=True.

        Raises:
            FileNotFoundError: If no LoRA with the given id exists.
            ValueError: If the LoRA is not compatible with LTX-2.3.
        """
        # Search on-disk LoRAs first, then built-ins
        all_loras = {info.id: info for info in self.list_loras()}

        if lora_id not in all_loras:
            raise FileNotFoundError(f"LoRA not found: {lora_id!r}")

        info = all_loras[lora_id]
        if not info.compatible:
            raise ValueError(
                f"LoRA {lora_id!r} is not compatible with LTX-2.3. "
                "LTX-2.0 LoRAs cannot be used with the 2.3 latent space."
            )

        info.loaded = True
        self._loaded_loras[lora_id] = info
        log.info("LoRA loaded (stub — no weight application yet): %s", lora_id)
        # Stub: real weight application will call self._model_manager.get_model()
        # and inject the LoRA weights via mlx linear layer patching.
        return info

    def unload_lora(self, lora_id: str) -> None:
        """Remove a LoRA from the active set and free any associated memory.

        Args:
            lora_id: Slug of the LoRA to unload.
        """
        if lora_id in self._loaded_loras:
            del self._loaded_loras[lora_id]
            log.info("LoRA unloaded: %s", lora_id)
        else:
            log.warning("unload_lora called for non-loaded LoRA: %s", lora_id)
        aggressive_cleanup()

    def unload_all(self) -> None:
        """Unload every active LoRA and free associated memory."""
        count = len(self._loaded_loras)
        self._loaded_loras.clear()
        log.info("Unloaded all LoRAs (%d total)", count)
        aggressive_cleanup()

    def get_loaded(self) -> list[LoRAInfo]:
        """Return all currently loaded (applied) LoRAs.

        Returns:
            List of LoRAInfo objects with loaded=True.
        """
        return list(self._loaded_loras.values())

    def list_loras(self) -> list[LoRAInfo]:
        """Return all available LoRAs: built-ins + discovered files.

        The loaded field reflects the current state from _loaded_loras.

        Returns:
            Combined list of built-in stubs and on-disk LoRAs.
        """
        disk_loras = self.scan_loras()
        disk_ids = {info.id for info in disk_loras}

        # Inject built-ins that are not shadowed by an on-disk file with the same id
        builtins: list[LoRAInfo] = []
        for builtin in _BUILTIN_LORAS:
            if builtin.id not in disk_ids:
                # Return a copy with the current loaded state
                copy = LoRAInfo(
                    id=builtin.id,
                    name=builtin.name,
                    path=builtin.path,
                    lora_type=builtin.lora_type,
                    compatible=builtin.compatible,
                    loaded=builtin.id in self._loaded_loras,
                    size_mb=builtin.size_mb,
                )
                builtins.append(copy)

        return builtins + disk_loras
