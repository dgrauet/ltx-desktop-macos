"""Text-to-Video generation pipeline.

Uses MLX inference via mlx-video-with-audio subprocess for real video generation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from engine.memory_manager import (
    aggressive_cleanup,
    get_memory_stats,
    increment_generation_count,
    periodic_reload_check,
    reset_peak_memory,
)
from engine.mlx_runner import run_mlx_generation
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
    """Text-to-Video generation pipeline using MLX."""

    def __init__(self, model_manager: ModelManager, teacache_enabled: bool = True) -> None:
        self._model_manager = model_manager
        self._teacache_enabled = teacache_enabled

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
        upscale: bool = False,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
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
            progress_callback: Optional callback(step, total_steps, pct, preview_frame).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        job_id = str(uuid.uuid4())[:8]
        stages: dict[str, float] = {}
        start_time = time.monotonic()

        async def _notify(
            step: int, total: int, pct: float, frame: str | None = None,
            *, status: str | None = None
        ) -> None:
            if not progress_callback:
                return
            result = progress_callback(step, total, pct, frame, status=status)
            if inspect.isawaitable(result):
                await result

        reset_peak_memory()
        aggressive_cleanup()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.mp4"

        # Adapt mlx_runner progress to pipeline progress_callback format
        async def _progress_adapter(
            step: int, total_steps: int, stage: int, pct: float,
            *, status: str | None = None
        ) -> None:
            await _notify(step, total_steps, pct, None, status=status)

        # Run MLX inference (handles model loading, text encoding, denoising,
        # VAE decode, audio decode, and ffmpeg muxing internally)
        log.info(
            "[%s] Starting T2V generation: prompt=%r, %dx%d, %d frames, seed=%d",
            job_id, prompt[:80], width, height, num_frames, seed,
        )
        t0 = time.monotonic()

        await run_mlx_generation(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=str(output_path),
            tiling="auto",
            upscale=upscale,
            progress_callback=_progress_adapter,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()

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
