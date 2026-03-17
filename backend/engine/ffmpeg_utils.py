"""Shared ffmpeg/ffprobe utilities for the LTX Desktop backend.

Consolidates duplicate find/probe logic that was previously scattered
across generate_v23.py, vae_decoder.py, audio_mixer.py, extend.py,
and retake.py.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def find_ffmpeg() -> str:
    """Find the ffmpeg binary on the system.

    Checks ``shutil.which`` first, then common Homebrew installation paths
    on Apple Silicon and Intel Macs.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        FileNotFoundError: If ffmpeg cannot be found anywhere.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffmpeg not found. Install with: brew install ffmpeg")


def find_ffprobe() -> str:
    """Find the ffprobe binary on the system.

    Checks ``shutil.which`` first, then common Homebrew installation paths
    on Apple Silicon and Intel Macs.

    Returns:
        Absolute path to the ffprobe executable.

    Raises:
        FileNotFoundError: If ffprobe cannot be found anywhere.
    """
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffprobe not found. Install with: brew install ffmpeg")


def probe_video_info(video_path: str) -> tuple[int, int, float]:
    """Probe a video file for width, height, and duration using ffprobe.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Tuple of (width, height, duration_seconds). Falls back to
        (768, 512, 4.0) if probing fails.
    """
    try:
        ffprobe_bin = find_ffprobe()
    except FileNotFoundError:
        log.warning("ffprobe not found — using defaults 768x512 @ 4.0s")
        return 768, 512, 4.0

    if not Path(video_path).exists():
        log.warning("Source video not found: %s — using defaults", video_path)
        return 768, 512, 4.0

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            width = int(parts[0])
            height = int(parts[1])
            duration = float(parts[2]) if len(parts) > 2 else 4.0
            return width, height, duration
    except Exception:
        log.warning("ffprobe failed to read info from %s", video_path)
    return 768, 512, 4.0


def has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains an audio stream.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        True if the video contains at least one audio stream, False otherwise.
    """
    try:
        ffprobe_bin = find_ffprobe()
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and "audio" in result.stdout.strip()
    except Exception:
        return False
