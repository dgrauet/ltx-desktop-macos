"""Retake pipeline — regenerate a specific time segment of an existing video.

Uses MLX inference to encode the source video to latent space, create a
temporal region mask, and run masked diffusion to regenerate only the
specified segment while preserving the surrounding context.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from pathlib import Path
from typing import Callable

from engine.ffmpeg_utils import probe_video_info
from engine.memory_manager import (
    aggressive_cleanup,
    build_memory_stats_from_subprocess,
    increment_generation_count,
    periodic_reload_check,
    reset_peak_memory,
)
from engine.mlx_runner import run_mlx_generation
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import GenerationResult

log = logging.getLogger(__name__)

# Output directory for retake results
OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "retakes"

# LTX-2.3 VAE temporal downsampling factor (pixel frames -> latent frames)
# latent_frames = (pixel_frames - 1) / 8 + 1
_VAE_TEMPORAL_FACTOR = 8


class RetakePipeline:
    """Regenerates a specific time segment of an existing video using MLX."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        source_video_path: str,
        prompt: str,
        start_time_s: float,
        end_time_s: float,
        steps: int = 8,
        seed: int = 42,
        fps: int = 24,
        model_repo_id: str | None = None,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Regenerate a specific time segment of a video.

        Encodes the full source video to latent space, masks the retake
        region, runs diffusion with mask blending, then decodes and saves.

        Args:
            source_video_path: Absolute path to the source video file.
            prompt: Text prompt describing the desired content for the segment.
            start_time_s: Start of the retake region in seconds.
            end_time_s: End of the retake region in seconds.
            steps: Number of denoising steps.
            seed: Random seed for reproducibility.
            fps: Frames per second (must match source video).
            model_repo_id: Optional HuggingFace model repo ID.
            progress_callback: Optional callback(step, total_steps, pct, preview_frame).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        # Validate source video
        src_path = Path(source_video_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_video_path}")

        if end_time_s <= start_time_s:
            raise ValueError(
                f"end_time_s ({end_time_s}) must be greater than start_time_s ({start_time_s})"
            )

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
        output_path = OUTPUT_DIR / f"retake_{job_id}.mp4"

        # Probe source video dimensions and frame count
        source_width, source_height, source_num_frames = _probe_video_info(
            source_video_path, fps
        )
        log.info(
            "[%s] Retake: source=%r %dx%d %d frames, segment=%.2fs-%.2fs, prompt=%r",
            job_id, source_video_path, source_width, source_height,
            source_num_frames, start_time_s, end_time_s, prompt[:80],
        )

        # Ensure num_frames follows the 1+8k constraint for the VAE
        num_frames = _round_to_vae_compatible(source_num_frames)
        if num_frames != source_num_frames:
            log.info(
                "[%s] Adjusted num_frames from %d to %d (VAE 1+8k constraint)",
                job_id, source_num_frames, num_frames,
            )

        # Convert time range to latent frame indices
        # pixel_frame = time_s * fps
        # latent_frame = (pixel_frame - 1) / 8 + 1, but frame 0 maps to latent 0
        latent_frames = (num_frames - 1) // _VAE_TEMPORAL_FACTOR + 1
        start_latent = _pixel_time_to_latent_frame(start_time_s, fps)
        end_latent = _pixel_time_to_latent_frame(end_time_s, fps)

        # Clamp to valid latent range
        start_latent = max(0, min(start_latent, latent_frames))
        end_latent = max(start_latent + 1, min(end_latent, latent_frames))

        log.info(
            "[%s] Latent mask: frames %d-%d of %d (regenerate)",
            job_id, start_latent, end_latent, latent_frames,
        )

        # Adapt mlx_runner progress to pipeline progress_callback format
        async def _progress_adapter(
            step: int, total_steps: int, stage: int, pct: float,
            *, status: str | None = None, preview_frame: str | None = None,
        ) -> None:
            await _notify(step, total_steps, pct, preview_frame, status=status)

        # Run MLX inference with retake conditioning
        t0 = time.monotonic()

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=source_height,
            width=source_width,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=str(output_path),
            num_steps=steps,
            retake_source=source_video_path,
            retake_start_frame=start_latent,
            retake_end_frame=end_latent,
            preview_interval=2,
            progress_callback=_progress_adapter,
            model_repo_id=model_repo_id,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()

        total_duration = time.monotonic() - start_time
        log.info("[%s] Retake complete in %.2fs → %s", job_id, total_duration, output_path)

        increment_generation_count()
        periodic_reload_check(self._model_manager)

        subprocess_memory = gen_result.get("subprocess_memory", {})
        memory_after = build_memory_stats_from_subprocess(subprocess_memory)

        return GenerationResult(
            job_id=job_id,
            output_path=str(output_path),
            duration_seconds=total_duration,
            memory_after=memory_after,
            stages=stages,
        )


def _probe_video_info(video_path: str, fps: int) -> tuple[int, int, int]:
    """Probe a video file for dimensions and frame count.

    Delegates to the shared ``probe_video_info`` utility and converts the
    duration to a frame count using the given *fps*.

    Args:
        video_path: Absolute path to the video file.
        fps: Expected FPS (used for frame count estimation from duration).

    Returns:
        Tuple of (width, height, num_frames). Falls back to (768, 512, 97)
        if probing fails.
    """
    width, height, duration = probe_video_info(video_path)
    # Default fallback from probe_video_info is (768, 512, 4.0)
    num_frames = round(duration * fps)
    if num_frames < 1:
        num_frames = 97
    return width, height, num_frames


def _round_to_vae_compatible(num_frames: int) -> int:
    """Round frame count to nearest VAE-compatible value (1 + 8k).

    The LTX-2.3 VAE requires num_frames = 1 + 8*k (e.g. 1, 9, 17, 25...).
    Rounds down to the nearest valid value.
    """
    if num_frames <= 1:
        return 1
    k = (num_frames - 1) // 8
    return 1 + k * 8


def _pixel_time_to_latent_frame(time_s: float, fps: int) -> int:
    """Convert a time in seconds to a latent frame index.

    The VAE downsamples temporally by 8x:
        pixel_frame = time_s * fps
        latent_frame ~ pixel_frame / 8

    Args:
        time_s: Time in seconds.
        fps: Video frames per second.

    Returns:
        Latent frame index (integer).
    """
    pixel_frame = time_s * fps
    # The VAE maps pixel frames [0, 1..8] -> latent frame 0,
    # pixel frames [9..16] -> latent frame 1, etc.
    # More precisely: latent_frame = (pixel_frame) / 8, rounded
    return max(0, round(pixel_frame / _VAE_TEMPORAL_FACTOR))
