"""Image-to-Video (I2V) generation pipeline.

Takes a reference image that conditions the first frame of the generated video.
Sprint 2: stubbed inference with ffmpeg placeholder videos.
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
    increment_generation_count,
    periodic_reload_check,
    reset_peak_memory,
)
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import GenerationResult, _generate_preview_frame

log = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs"


class ImageToVideoPipeline:
    """Image-to-Video generation pipeline using MLX (stubbed for Sprint 2)."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        prompt: str,
        source_image_path: str,
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        steps: int = 8,
        seed: int = 42,
        guidance_scale: float = 1.0,
        fps: int = 24,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Run the I2V generation pipeline.

        Args:
            prompt: Text prompt describing the video.
            source_image_path: Path to the reference image conditioning the first frame.
            width: Output video width.
            height: Output video height.
            num_frames: Number of frames to generate.
            steps: Number of denoising steps.
            seed: Random seed for reproducibility.
            guidance_scale: Classifier-free guidance scale.
            fps: Output frames per second.
            progress_callback: Optional callback(step, total_steps, pct, preview_frame).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        # Validate source image
        img_path = Path(source_image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_image_path}")

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
        output_path = OUTPUT_DIR / f"i2v_{job_id}.mp4"

        total_stages = 5
        current_stage = 0

        # Stage 1: Image encoding + Text encoding
        log.info("[%s] I2V Stage 1: Image+Text encoding — image=%s", job_id, source_image_path)
        t0 = time.monotonic()
        await asyncio.sleep(0.1)  # Stub: simulate encoding
        stages["image_text_encoding"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 2: Diffusion (image-conditioned denoising loop)
        log.info("[%s] I2V Stage 2: Diffusion — %d steps (image-conditioned)", job_id, steps)
        t0 = time.monotonic()
        for step in range(steps):
            await asyncio.sleep(0.05)  # Stub: simulate denoising step
            preview_frame = None
            if (step + 1) % 4 == 0 or step == steps - 1:
                preview_frame = _generate_preview_frame(width, height, step, steps)
            pct = (current_stage + (step + 1) / steps) / total_stages
            await _notify(step + 1, steps, pct, preview_frame)
        stages["diffusion"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1

        # Stage 3: Upscale
        log.info("[%s] I2V Stage 3: Upscale (skipped in stub)", job_id)
        t0 = time.monotonic()
        await asyncio.sleep(0.05)
        stages["upscale"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 4: VAE decode + video encoding
        log.info("[%s] I2V Stage 4: VAE decode → ffmpeg", job_id)
        t0 = time.monotonic()
        duration_secs = num_frames / fps
        _generate_stub_video(output_path, width, height, duration_secs, fps)
        stages["vae_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 5: Audio decode
        log.info("[%s] I2V Stage 5: Audio decode (stub)", job_id)
        t0 = time.monotonic()
        await asyncio.sleep(0.05)
        stages["audio_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, 1.0)

        total_duration = time.monotonic() - start_time
        log.info("[%s] I2V generation complete in %.2fs", job_id, total_duration)

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
    """Generate a placeholder MP4 (purple tint to distinguish from T2V)."""
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
        "-i", f"color=c=0x2e1a2e:s={width}x{height}:d={duration:.3f}:r={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency=330:duration={duration:.3f}:sample_rate=44100",
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
    log.info("I2V stub video created: %s", output_path)
