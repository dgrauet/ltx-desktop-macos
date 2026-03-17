"""Preset storage for generation parameter bundles.

Stores presets as individual JSON files in ~/.ltx-desktop/presets/.
Provides CRUD operations and ships with built-in default presets.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRESETS_DIR = Path.home() / ".ltx-desktop" / "presets"

_lock = threading.Lock()

# Built-in presets that are created on first run.
BUILTIN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "builtin-quick-preview",
        "name": "Quick Preview",
        "builtin": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "params": {
            "width": 384,
            "height": 256,
            "num_frames": 9,
            "steps": 4,
            "seed": -1,
            "fps": 24,
            "guidance_scale": 1.0,
            "negative_prompt": "",
            "generate_audio": False,
            "ffmpeg_upscale": False,
        },
    },
    {
        "id": "builtin-standard",
        "name": "Standard",
        "builtin": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "params": {
            "width": 768,
            "height": 512,
            "num_frames": 97,
            "steps": 8,
            "seed": -1,
            "fps": 24,
            "guidance_scale": 1.0,
            "negative_prompt": "",
            "generate_audio": False,
            "ffmpeg_upscale": False,
        },
    },
    {
        "id": "builtin-high-quality",
        "name": "High Quality",
        "builtin": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "params": {
            "width": 1280,
            "height": 704,
            "num_frames": 97,
            "steps": 8,
            "seed": -1,
            "fps": 24,
            "guidance_scale": 1.0,
            "negative_prompt": "",
            "generate_audio": False,
            "ffmpeg_upscale": False,
        },
    },
]


def _ensure_dir() -> None:
    """Create the presets directory if it doesn't exist."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def _preset_path(preset_id: str) -> Path:
    """Return the file path for a preset by ID."""
    return PRESETS_DIR / f"{preset_id}.json"


def _read_preset(path: Path) -> dict[str, Any] | None:
    """Read a single preset file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read preset %s: %s", path, exc)
    return None


def _ensure_builtins() -> None:
    """Write built-in presets to disk if they don't exist yet."""
    _ensure_dir()
    for preset in BUILTIN_PRESETS:
        path = _preset_path(preset["id"])
        if not path.exists():
            path.write_text(
                json.dumps(preset, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Created built-in preset: %s", preset["name"])


def list_presets() -> list[dict[str, Any]]:
    """Return all presets, sorted by name (builtins first).

    Returns:
        List of preset dicts.
    """
    with _lock:
        _ensure_builtins()
        presets: list[dict[str, Any]] = []
        for path in PRESETS_DIR.glob("*.json"):
            preset = _read_preset(path)
            if preset:
                presets.append(preset)

    # Sort: builtins first (by name), then user presets by created_at
    def sort_key(p: dict[str, Any]) -> tuple[int, str]:
        is_builtin = 0 if p.get("builtin") else 1
        return (is_builtin, p.get("name", ""))

    presets.sort(key=sort_key)
    return presets


def get_preset(preset_id: str) -> dict[str, Any] | None:
    """Get a single preset by ID.

    Args:
        preset_id: The preset identifier.

    Returns:
        Preset dict or None if not found.
    """
    with _lock:
        _ensure_builtins()
        path = _preset_path(preset_id)
        if path.exists():
            return _read_preset(path)
    return None


def create_preset(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Create a new user preset.

    Args:
        name: Display name for the preset.
        params: Generation parameters to store.

    Returns:
        The created preset dict.
    """
    preset_id = str(uuid.uuid4())
    preset = {
        "id": preset_id,
        "name": name,
        "builtin": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
    }

    with _lock:
        _ensure_dir()
        path = _preset_path(preset_id)
        path.write_text(
            json.dumps(preset, indent=2, default=str),
            encoding="utf-8",
        )

    log.info("Created preset '%s' (%s)", name, preset_id)
    return preset


def update_preset(preset_id: str, name: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Update an existing preset.

    Args:
        preset_id: The preset to update.
        name: New name (optional).
        params: New params (optional).

    Returns:
        Updated preset dict, or None if not found or is a builtin.
    """
    with _lock:
        path = _preset_path(preset_id)
        if not path.exists():
            return None

        preset = _read_preset(path)
        if not preset:
            return None

        if preset.get("builtin"):
            return None  # Cannot modify builtins

        if name is not None:
            preset["name"] = name
        if params is not None:
            preset["params"] = params

        path.write_text(
            json.dumps(preset, indent=2, default=str),
            encoding="utf-8",
        )

    log.info("Updated preset '%s' (%s)", preset.get("name"), preset_id)
    return preset


def delete_preset(preset_id: str) -> bool:
    """Delete a preset by ID.

    Built-in presets cannot be deleted.

    Args:
        preset_id: The preset to delete.

    Returns:
        True if deleted, False if not found or is a builtin.
    """
    with _lock:
        path = _preset_path(preset_id)
        if not path.exists():
            return False

        preset = _read_preset(path)
        if preset and preset.get("builtin"):
            return False  # Cannot delete builtins

        path.unlink()

    log.info("Deleted preset %s", preset_id)
    return True
