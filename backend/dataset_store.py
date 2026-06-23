"""Disk-backed training datasets: manifest CRUD + media/adequacy validation."""
from __future__ import annotations

import json
from pathlib import Path

from engine.ffmpeg_utils import probe_frame_count, probe_video_info

MIN_CLIPS = 5


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
