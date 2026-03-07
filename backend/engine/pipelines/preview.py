"""Rapid Preview pipeline — fast low-res preview for validation.

Produces a 384x256, 4-step, single-stage preview video so the user
can validate the prompt direction before launching a full generation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from engine.memory_manager import (
    aggressive_cleanup,
    get_memory_stats,
    reset_peak_memory,
)
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import GenerationResult

log = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "previews"

# Fixed preview settings
PREVIEW_WIDTH = 384
PREVIEW_HEIGHT = 256
PREVIEW_STEPS = 4


class PreviewPipeline:
    """Fast preview pipeline — single-stage, low-res, no upscaler, no audio."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        prompt: str,
        seed: int = 42,
        fps: int = 24,
        num_frames: int = 25,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Run the rapid preview pipeline.

        Args:
            prompt: Text prompt describing the video.
            seed: Random seed for reproducibility.
            fps: Output frames per second.
            num_frames: Number of frames (default 25 for short preview).
            progress_callback: Optional callback(step, total_steps, pct, preview_frame).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        job_id = str(uuid.uuid4())[:8]
        stages: dict[str, float] = {}
        start_time = time.monotonic()

        async def _notify(step: int, total: int, pct: float, frame: str | None = None) -> None:
            if not progress_callback:
                return
            result = progress_callback(step, total, pct, frame)
            if inspect.isawaitable(result):
                await result

        reset_peak_memory()
        aggressive_cleanup()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"preview_{job_id}.mp4"

        total_stages = 3
        current_stage = 0

        # Stage 1: Text encoding
        log.info("[%s] Preview Stage 1: Text encoding", job_id)
        t0 = time.monotonic()
        await asyncio.sleep(0.02)  # Stub: fast
        stages["text_encoding"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 2: Diffusion (single-stage, 4 steps)
        log.info("[%s] Preview Stage 2: Diffusion — %d steps", job_id, PREVIEW_STEPS)
        t0 = time.monotonic()
        for step in range(PREVIEW_STEPS):
            await asyncio.sleep(0.02)  # Stub: fast
            pct = (current_stage + (step + 1) / PREVIEW_STEPS) / total_stages
            await _notify(step + 1, PREVIEW_STEPS, pct)
        stages["diffusion"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1

        # Stage 3: VAE decode + video encoding
        log.info("[%s] Preview Stage 3: VAE decode → ffmpeg", job_id)
        t0 = time.monotonic()
        duration_secs = num_frames / fps
        _generate_stub_video(output_path, PREVIEW_WIDTH, PREVIEW_HEIGHT, duration_secs, fps)
        stages["vae_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, 1.0)

        total_duration = time.monotonic() - start_time
        log.info("[%s] Preview complete in %.2fs", job_id, total_duration)

        return GenerationResult(
            job_id=job_id,
            output_path=str(output_path),
            duration_seconds=total_duration,
            memory_after=get_memory_stats(),
            stages=stages,
        )


def _generate_stub_video(
    output_path: Path, width: int, height: int, duration: float, fps: int
) -> None:
    """Generate a placeholder preview MP4 (green tint to distinguish from full gen)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(p).exists():
                ffmpeg_bin = p
                break
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a2e1a:s={width}x{height}:d={duration:.3f}:r={fps}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    log.info("Preview stub video created: %s", output_path)
