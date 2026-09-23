"""User-registered local LTX model packs (directories outside the HF cache).

LTX-2.5 MLX repos are gated and large (47–120 GB); users who converted or
downloaded a pack themselves (e.g. with mlx-forge) register its directory here
instead of re-downloading it. The registry only stores paths — unregistering
never touches the files.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REGISTRY_PATH = Path.home() / ".ltx-desktop" / "local_models.json"

_TRANSFORMER_FILES = (
    "transformer-distilled.safetensors",
    "transformer-dev.safetensors",
    "transformer.safetensors",
)
_CONFIG_FILES = ("embedded_config.json", "config.json")


def validate_pack(path: str | Path) -> Path:
    """Return the resolved pack directory, or raise ValueError explaining what is missing."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    if not any((directory / f).is_file() for f in _CONFIG_FILES):
        raise ValueError(f"No embedded_config.json / config.json in {directory}")
    if not any((directory / f).is_file() for f in _TRANSFORMER_FILES):
        raise ValueError(f"No transformer-*.safetensors in {directory}")
    return directory


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-") or "pack"


def _load() -> list[dict[str, Any]]:
    try:
        data = json.loads(REGISTRY_PATH.read_text())
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Ignoring unreadable local model registry %s: %s", REGISTRY_PATH, e)
        return []
    return [e for e in data if isinstance(e, dict) and "id" in e and "path" in e]


def _save(entries: list[dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(REGISTRY_PATH)


def list_local() -> list[dict[str, Any]]:
    """Registered packs as ``{id, name, path}`` dicts (paths may have gone missing)."""
    return _load()


def register(path: str | Path, name: str | None = None) -> dict[str, Any]:
    """Register a pack directory; re-registering the same path returns the existing entry."""
    directory = validate_pack(path)
    entries = _load()
    for entry in entries:
        if entry["path"] == str(directory):
            return entry
    base = f"local-{_slug(directory.name)}"
    taken = {e["id"] for e in entries}
    model_id, n = base, 2
    while model_id in taken:
        model_id, n = f"{base}-{n}", n + 1
    entry = {"id": model_id, "name": name or directory.name, "path": str(directory)}
    entries.append(entry)
    _save(entries)
    return entry


def unregister(model_id: str) -> bool:
    """Remove a pack from the registry. Returns False if the id is unknown."""
    entries = _load()
    kept = [e for e in entries if e["id"] != model_id]
    if len(kept) == len(entries):
        return False
    _save(kept)
    return True
