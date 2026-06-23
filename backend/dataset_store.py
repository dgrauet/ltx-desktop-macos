"""Disk-backed training datasets: manifest CRUD + media/adequacy validation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from engine.ffmpeg_utils import probe_frame_count, probe_video_info

MIN_CLIPS = 5

TRAINING_DIR: Path = Path.home() / ".ltx-desktop" / "training"


# ---------------------------------------------------------------------------
# Dataset directory helpers
# ---------------------------------------------------------------------------


def dataset_dir(dataset_id: str) -> Path:
    """Return the root directory for the given dataset."""
    return TRAINING_DIR / dataset_id


def clips_dir(dataset_id: str) -> Path:
    """Return the clips subdirectory for the given dataset."""
    return dataset_dir(dataset_id) / "clips"


def captions_dir(dataset_id: str) -> Path:
    """Return the captions subdirectory for the given dataset."""
    return dataset_dir(dataset_id) / "captions"


def precomputed_dir(dataset_id: str) -> Path:
    """Return the hidden precomputed-data subdirectory for the given dataset."""
    return dataset_dir(dataset_id) / ".precomputed"


def create_dataset(dataset_id: str) -> Path:
    """Create a new dataset directory with clips/ and captions/ subdirs.

    Args:
        dataset_id: Unique identifier for the dataset.

    Returns:
        The dataset root directory path.
    """
    root = dataset_dir(dataset_id)
    clips_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    captions_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    return root


def list_datasets() -> list[dict]:
    """List all datasets under TRAINING_DIR.

    Returns:
        List of dicts with keys: id, clip_count, disk_bytes, has_precomputed.
    """
    if not TRAINING_DIR.exists():
        return []
    result: list[dict] = []
    for entry in sorted(TRAINING_DIR.iterdir()):
        if not entry.is_dir():
            continue
        clips = clips_dir(entry.name)
        clip_count = sum(1 for f in clips.iterdir() if f.is_file()) if clips.exists() else 0
        disk_bytes = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        has_precomputed = precomputed_dir(entry.name).exists()
        result.append(
            {
                "id": entry.name,
                "clip_count": clip_count,
                "disk_bytes": disk_bytes,
                "has_precomputed": has_precomputed,
            }
        )
    return result


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset directory and all its contents.

    Args:
        dataset_id: Unique identifier for the dataset.

    Returns:
        True if the dataset was deleted, False if it did not exist.
    """
    root = dataset_dir(dataset_id)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def materialize_captions(dataset_id: str) -> None:
    """Write per-clip .txt caption files from the dataset manifest.

    Reads manifest.json from the dataset directory and writes one
    ``<stem>.txt`` file per entry into captions_dir, as expected by
    ``preprocess_dataset(caption_ext='.txt')``.

    Args:
        dataset_id: Unique identifier for the dataset.
    """
    manifest = read_manifest(str(dataset_dir(dataset_id)))
    cap_dir = captions_dir(dataset_id)
    cap_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        stem = Path(entry["video"]).stem
        (cap_dir / f"{stem}.txt").write_text(entry["caption"])


def validate_clip(path: str) -> list[str]:
    if not Path(path).exists():
        return [f"{path}: file not found"]
    violations: list[str] = []
    w, h, _fps = probe_video_info(path)
    frames = probe_frame_count(path)
    if frames == 0:
        violations.append("could not read video (0 frames) — file may be corrupt")
        return violations
    if w % 32 != 0:
        violations.append(f"width {w} not divisible by 32")
    if h % 32 != 0:
        violations.append(f"height {h} not divisible by 32")
    if frames % 8 != 1:
        nearest_lo = frames - ((frames - 1) % 8)
        violations.append(
            f"frame count {frames} invalid (needs frames % 8 == 1; "
            f"nearest valid: {nearest_lo} or {nearest_lo + 8})"
        )
    return violations


def build_manifest(entries: list[dict]) -> list[dict]:
    manifest: list[dict] = []
    for e in entries:
        caption = (e.get("caption") or "").strip()
        if not caption:
            raise ValueError(f"empty caption for entry {e!r}")
        manifest.append({"caption": caption, "video": e["video"]})
    return manifest


def adequacy_warnings(manifest: list[dict]) -> list[str]:
    warns: list[str] = []
    if len(manifest) < MIN_CLIPS:
        warns.append(
            f"only {len(manifest)} clips (< {MIN_CLIPS}); LoRA quality likely poor"
        )
    return warns


def write_manifest(dataset_dir: str, manifest: list[dict]) -> str:
    p = Path(dataset_dir) / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)


def read_manifest(dataset_dir: str) -> list[dict]:
    p = Path(dataset_dir) / "manifest.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
