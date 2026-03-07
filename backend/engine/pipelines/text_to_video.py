"""Text-to-Video generation pipeline.

Sprint 1: stubbed inference that produces real MP4 files via ffmpeg
so the API and frontend can integrate against a working pipeline.
Real MLX inference will replace the stubs in Sprint 2+.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from engine.memory_manager import (
    aggressive_cleanup,
    get_memory_stats,
    increment_generation_count,
    periodic_reload_check,
    reset_peak_memory,
)
from engine.model_manager import ModelManager

log = logging.getLogger(__name__)

# Default output directory
OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs"


@dataclass
class GenerationResult:
    """Result of a video generation."""

    job_id: str
    output_path: str
    duration_seconds: float
    memory_after: dict
    stages: dict[str, float] = field(default_factory=dict)


class TextToVideoPipeline:
    """Text-to-Video generation pipeline using MLX (stubbed for Sprint 1)."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        prompt: str,
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        steps: int = 8,
        seed: int = 42,
        guidance_scale: float = 1.0,
        fps: int = 24,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> GenerationResult:
        """Run the full T2V generation pipeline.

        Args:
            prompt: Text prompt describing the video.
            width: Output video width.
            height: Output video height.
            num_frames: Number of frames to generate.
            steps: Number of denoising steps.
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            fps: Output frames per second.
            progress_callback: Optional callback(step, total_steps, pct).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        job_id = str(uuid.uuid4())[:8]
        stages: dict[str, float] = {}
        start_time = time.monotonic()

        async def _notify(step: int, total: int, pct: float) -> None:
            if not progress_callback:
                return
            result = progress_callback(step, total, pct)
            if inspect.isawaitable(result):
                await result

        reset_peak_memory()
        aggressive_cleanup()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.mp4"

        total_stages = 5
        current_stage = 0

        # Stage 1: Text encoding
        log.info("[%s] Stage 1: Text encoding — prompt=%r", job_id, prompt[:80])
        t0 = time.monotonic()
        await asyncio.sleep(0.1)  # Stub: simulate text encoding
        stages["text_encoding"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 2: Diffusion (denoising loop)
        log.info("[%s] Stage 2: Diffusion — %d steps", job_id, steps)
        t0 = time.monotonic()
        for step in range(steps):
            await asyncio.sleep(0.05)  # Stub: simulate denoising step
            pct = (current_stage + (step + 1) / steps) / total_stages
            await _notify(step + 1, steps, pct)
        stages["diffusion"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1

        # Stage 3: Upscale (optional, stubbed)
        log.info("[%s] Stage 3: Upscale (skipped in stub)", job_id)
        t0 = time.monotonic()
        await asyncio.sleep(0.05)
        stages["upscale"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 4: VAE decode + video encoding via ffmpeg
        log.info("[%s] Stage 4: VAE decode → ffmpeg", job_id)
        t0 = time.monotonic()
        duration_secs = num_frames / fps
        _generate_stub_video(output_path, width, height, duration_secs, fps)
        stages["vae_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 5: Audio decode
        log.info("[%s] Stage 5: Audio decode (stub)", job_id)
        t0 = time.monotonic()
        await asyncio.sleep(0.05)
        stages["audio_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, 1.0)

        total_duration = time.monotonic() - start_time
        log.info("[%s] Generation complete in %.2fs", job_id, total_duration)

        increment_generation_count()
        periodic_reload_check(self._model_manager)

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
    """Generate a placeholder MP4 with solid color video and sine wave audio.

    Args:
        output_path: Where to write the MP4.
        width: Video width.
        height: Video height.
        duration: Video duration in seconds.
        fps: Frames per second.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        # Common homebrew location
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(p).exists():
                ffmpeg_bin = p
                break
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    cmd = [
        ffmpeg_bin,
        "-y",
        # Video: solid dark blue
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:s={width}x{height}:d={duration:.3f}:r={fps}",
        # Audio: 440Hz sine wave
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration:.3f}:sample_rate=44100",
        # Encoding
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    log.info("Stub video created: %s", output_path)
