"""Persistent history storage for completed generations.

Stores generation metadata as a JSON array in ~/.ltx-desktop/history.json.
Append-only on completion; supports read (newest first) and delete by job_id.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HISTORY_DIR = Path.home() / ".ltx-desktop"
HISTORY_FILE = HISTORY_DIR / "history.json"
MAX_ENTRIES = 100


_lock = threading.Lock()


def _read_entries() -> list[dict[str, Any]]:
    """Read existing history entries from disk."""
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read history file: %s", exc)
    return []


def _write_entries(entries: list[dict[str, Any]]) -> None:
    """Write history entries to disk."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(entries, indent=2, default=str),
        encoding="utf-8",
    )


def add_entry(
    job_id: str,
    prompt: str,
    output_path: str,
    duration_seconds: float,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    generation_type: str,
) -> None:
    """Append a completed generation to the history file.

    Args:
        job_id: Unique job identifier.
        prompt: The text prompt used for generation.
        output_path: Absolute path to the output video file.
        duration_seconds: Wall-clock generation time in seconds.
        width: Output video width in pixels.
        height: Output video height in pixels.
        num_frames: Number of generated frames.
        fps: Frames per second.
        seed: Random seed used.
        generation_type: One of "t2v", "preview", "i2v".
    """
    entry = {
        "job_id": job_id,
        "prompt": prompt,
        "output_path": output_path,
        "duration_seconds": round(duration_seconds, 2),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "seed": seed,
        "generation_type": generation_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with _lock:
        entries = _read_entries()
        entries.insert(0, entry)  # newest first
        entries = entries[:MAX_ENTRIES]
        _write_entries(entries)

    log.info("History: saved entry for job %s (%s)", job_id, generation_type)


def get_entries(limit: int = MAX_ENTRIES) -> list[dict[str, Any]]:
    """Return history entries, newest first.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        List of history entry dicts.
    """
    with _lock:
        entries = _read_entries()
    return entries[:limit]


def seed_from_existing_files() -> int:
    """Populate history with existing MP4 files not already tracked.

    Scans ~/.ltx-desktop/outputs/ and ~/.ltx-desktop/outputs/previews/
    for MP4 files and adds entries for any not already in the history.
    Uses file metadata (creation time, name) since prompt info is unavailable.

    Returns:
        Number of entries added.
    """
    output_dirs = [
        HISTORY_DIR / "outputs",
        HISTORY_DIR / "outputs" / "previews",
    ]

    with _lock:
        entries = _read_entries()
        known_paths = {e.get("output_path") for e in entries}
        known_jobs = {e.get("job_id") for e in entries}
        added = 0

        for output_dir in output_dirs:
            if not output_dir.exists():
                continue
            is_preview = "previews" in str(output_dir)
            for mp4 in sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
                path_str = str(mp4)
                job_id = mp4.stem
                # Strip prefixes like "preview_", "i2v_"
                clean_id = job_id
                for prefix in ("preview_", "i2v_"):
                    if clean_id.startswith(prefix):
                        clean_id = clean_id[len(prefix):]

                if path_str in known_paths or clean_id in known_jobs or job_id in known_jobs:
                    continue

                # Infer generation type from filename
                if is_preview or job_id.startswith("preview_"):
                    gen_type = "preview"
                elif job_id.startswith("i2v_"):
                    gen_type = "i2v"
                else:
                    gen_type = "t2v"

                stat = mp4.stat()
                created = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                entries.append({
                    "job_id": job_id,
                    "prompt": "(imported from existing file)",
                    "output_path": path_str,
                    "duration_seconds": 0,
                    "width": 0,
                    "height": 0,
                    "num_frames": 0,
                    "fps": 24,
                    "seed": 0,
                    "generation_type": gen_type,
                    "created_at": created.isoformat(),
                })
                added += 1

        if added > 0:
            # Sort newest first and cap
            entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
            entries = entries[:MAX_ENTRIES]
            _write_entries(entries)
            log.info("History: seeded %d entries from existing files", added)

    return added


def delete_entry(job_id: str) -> bool:
    """Remove a history entry by job_id.

    Args:
        job_id: The job ID to remove.

    Returns:
        True if the entry was found and removed, False otherwise.
    """
    with _lock:
        entries = _read_entries()
        original_len = len(entries)
        entries = [e for e in entries if e.get("job_id") != job_id]
        if len(entries) == original_len:
            return False
        _write_entries(entries)
    log.info("History: deleted entry for job %s", job_id)
    return True
