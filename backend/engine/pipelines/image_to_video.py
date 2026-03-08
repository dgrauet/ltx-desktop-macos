"""Image-to-Video (I2V) generation pipeline.

Takes a reference image that conditions the first frame of the generated video.
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
    increment_generation_count,
    periodic_reload_check,
    reset_peak_memory,
)
from engine.mlx_runner import run_mlx_generation
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import GenerationResult

log = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs"


class ImageToVideoPipeline:
    """Image-to-Video generation pipeline using MLX."""

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

        async def _notify(
            step: int, total: int, pct: float, frame: str | None = None
        ) -> None:
            if not progress_callback:
                return
            result = progress_callback(step, total, pct, frame)
            if inspect.isawaitable(result):
                await result

        reset_peak_memory()
        aggressive_cleanup()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"i2v_{job_id}.mp4"

        # Adapt mlx_runner progress to pipeline progress_callback format
        async def _progress_adapter(
            step: int, total_steps: int, stage: int, pct: float
        ) -> None:
            await _notify(step, total_steps, pct, None)

        # Run MLX inference with source image conditioning
        log.info(
            "[%s] Starting I2V generation: prompt=%r, image=%s, %dx%d, %d frames",
            job_id, prompt[:80], source_image_path, width, height, num_frames,
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
            image=source_image_path,
            tiling="auto",
            progress_callback=_progress_adapter,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()

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
