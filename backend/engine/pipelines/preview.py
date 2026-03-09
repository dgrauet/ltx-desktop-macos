"""Rapid Preview pipeline — fast low-res preview for validation.

Produces a 384x256, single-stage preview video so the user
can validate the prompt direction before launching a full generation.
Uses MLX inference via mlx-video-with-audio subprocess.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from engine.memory_manager import (
    aggressive_cleanup,
    get_memory_stats,
    reset_peak_memory,
)
from engine.mlx_runner import run_mlx_generation
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import GenerationResult

log = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "previews"

# Fixed preview settings
PREVIEW_WIDTH = 384
PREVIEW_HEIGHT = 256


class PreviewPipeline:
    """Fast preview pipeline — single-stage, low-res, no upscaler."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        prompt: str,
        seed: int = 42,
        fps: int = 24,
        num_frames: int = 9,
        image: str | None = None,
        image_strength: float = 1.0,
        upscale: bool = False,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Run the rapid preview pipeline.

        Args:
            prompt: Text prompt describing the video.
            seed: Random seed for reproducibility.
            fps: Output frames per second.
            num_frames: Number of frames (default 9 for fast preview).
            image: Optional path to source image for I2V preview.
            image_strength: Strength of image conditioning (0.0-1.0).
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
        output_path = OUTPUT_DIR / f"preview_{job_id}.mp4"

        # Adapt mlx_runner progress to pipeline progress_callback format
        async def _progress_adapter(
            step: int, total_steps: int, stage: int, pct: float,
            *, status: str | None = None
        ) -> None:
            await _notify(step, total_steps, pct, None, status=status)

        # Run MLX inference with preview settings (small resolution, few frames)
        mode = "I2V" if image else "T2V"
        log.info("[%s] Starting %s preview generation: prompt=%r", job_id, mode, prompt[:80])
        t0 = time.monotonic()

        await run_mlx_generation(
            prompt=prompt,
            height=PREVIEW_HEIGHT,
            width=PREVIEW_WIDTH,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=str(output_path),
            image=image,
            image_strength=image_strength,
            tiling="aggressive",
            upscale=False,  # Never upscale previews — low-res by design
            progress_callback=_progress_adapter,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()

        total_duration = time.monotonic() - start_time
        log.info("[%s] Preview complete in %.2fs", job_id, total_duration)

        return GenerationResult(
            job_id=job_id,
            output_path=str(output_path),
            duration_seconds=total_duration,
            memory_after=get_memory_stats(),
            stages=stages,
        )
