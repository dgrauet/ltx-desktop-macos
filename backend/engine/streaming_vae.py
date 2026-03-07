"""Streaming VAE decode — frame-by-frame decode to ffmpeg pipe.

Decodes latents one frame at a time, piping directly to ffmpeg's stdin.
This avoids holding all decoded frames in RAM simultaneously, which is
critical for preventing OOM during the peak memory moment.

Sprint 2: implements the streaming architecture with a stub VAE.
Real MLX VAE will replace the stub when models are loaded.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(p).exists():
                return p
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    return ffmpeg_bin


def streaming_vae_decode(
    latents: Any,
    vae: Any,
    output_path: str | Path,
    width: int,
    height: int,
    fps: int = 24,
    num_frames: int = 97,
    cleanup_interval: int = 8,
) -> None:
    """Decode latents frame-by-frame and pipe directly to ffmpeg.

    Avoids holding all decoded frames in RAM simultaneously.
    Each frame is decoded, written to ffmpeg's stdin, then freed.

    Args:
        latents: Latent tensor (unused in stub, will be MLX array).
        vae: VAE model (unused in stub, will be MLX model).
        output_path: Where to write the output MP4.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        num_frames: Number of frames to decode.
        cleanup_interval: Run aggressive_cleanup every N frames.
    """
    ffmpeg_bin = find_ffmpeg()
    output_path = Path(output_path)

    ffmpeg_proc = subprocess.Popen(
        [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for i in range(num_frames):
            # Stub: generate a solid-color frame that shifts over time
            progress = i / max(num_frames - 1, 1)
            r = int(26 + progress * 60)
            g = int(26 + progress * 80)
            b = int(46 + progress * 120)
            frame_bytes = bytes([r, g, b] * (width * height))

            # In production: frame = vae.decode_frame(latents[i])
            # frame_bytes = frame_to_rgb_bytes(frame)

            ffmpeg_proc.stdin.write(frame_bytes)

            # Free the decoded frame immediately
            # In production: del frame, frame_bytes

            if (i + 1) % cleanup_interval == 0:
                aggressive_cleanup()

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait(timeout=30)

        if ffmpeg_proc.returncode != 0:
            stderr = ffmpeg_proc.stderr.read().decode()
            raise RuntimeError(f"ffmpeg streaming encode failed: {stderr[:500]}")

        log.info("Streaming VAE decode complete: %s (%d frames)", output_path, num_frames)

    except Exception:
        ffmpeg_proc.kill()
        raise
    finally:
        aggressive_cleanup()
