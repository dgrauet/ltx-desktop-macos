"""Retake pipeline — regenerate a specific time segment of an existing video.

Sprint 3: stubbed inference that produces real MP4 files via ffmpeg.
Real MLX inference will replace the stubs in a later sprint.
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
from engine.pipelines.text_to_video import GenerationResult

log = logging.getLogger(__name__)

# Output directory for retake results
OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "retakes"


class RetakePipeline:
    """Regenerates a specific time segment of an existing video using MLX (stubbed for Sprint 3)."""

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
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Regenerate a specific time segment of a video.

        Extracts the target segment, runs diffusion conditioned on the
        surrounding context, then muxes the new segment back into the
        original video.

        Args:
            source_video_path: Absolute path to the source video file.
            prompt: Text prompt describing the desired content for the segment.
            start_time_s: Start of the retake region in seconds.
            end_time_s: End of the retake region in seconds.
            steps: Number of denoising steps.
            seed: Random seed for reproducibility.
            fps: Frames per second (must match source video).
            progress_callback: Optional callback(step, total_steps, pct, preview_frame).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
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
        output_path = OUTPUT_DIR / f"retake_{job_id}.mp4"

        total_stages = 5
        current_stage = 0

        segment_duration = end_time_s - start_time_s
        log.info(
            "[%s] Retake: source=%r segment=%.2fs–%.2fs (%.2fs) prompt=%r",
            job_id,
            source_video_path,
            start_time_s,
            end_time_s,
            segment_duration,
            prompt[:80],
        )

        # Stage 1: Source analysis — probe the source video
        log.info("[%s] Stage 1: Source video analysis", job_id)
        t0 = time.monotonic()
        source_width, source_height = _probe_video_dimensions(source_video_path)
        log.info("[%s] Source dimensions: %dx%d", job_id, source_width, source_height)
        stages["source_analysis"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 2: Text encoding
        log.info("[%s] Stage 2: Text encoding — prompt=%r", job_id, prompt[:80])
        t0 = time.monotonic()
        await asyncio.sleep(0.1)  # Stub: simulate text encoding
        stages["text_encoding"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 3: Diffusion (retake region denoising)
        log.info("[%s] Stage 3: Diffusion — %d steps (retake region)", job_id, steps)
        t0 = time.monotonic()
        for step in range(steps):
            await asyncio.sleep(0.05)  # Stub: simulate denoising step
            pct = (current_stage + (step + 1) / steps) / total_stages
            await _notify(step + 1, steps, pct)
        stages["diffusion"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1

        # Stage 4: VAE decode → stub segment video via ffmpeg
        log.info("[%s] Stage 4: VAE decode → ffmpeg (retake segment)", job_id)
        t0 = time.monotonic()
        _generate_stub_retake_video(
            source_video_path=source_video_path,
            output_path=output_path,
            width=source_width,
            height=source_height,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            fps=fps,
        )
        stages["vae_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 5: Mux new segment back into original video
        log.info("[%s] Stage 5: Mux retake segment with original audio", job_id)
        t0 = time.monotonic()
        await asyncio.sleep(0.05)  # Stub: simulate audio mux
        stages["mux"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, 1.0)

        total_duration = time.monotonic() - start_time
        log.info("[%s] Retake complete in %.2fs → %s", job_id, total_duration, output_path)

        increment_generation_count()
        periodic_reload_check(self._model_manager)

        return GenerationResult(
            job_id=job_id,
            output_path=str(output_path),
            duration_seconds=total_duration,
            memory_after=get_memory_stats(),
            stages=stages,
        )


def _probe_video_dimensions(video_path: str) -> tuple[int, int]:
    """Probe a video file for its width and height using ffprobe.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Tuple of (width, height). Falls back to (768, 512) if probing fails.
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        for p in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]:
            if Path(p).exists():
                ffprobe_bin = p
                break
    if not ffprobe_bin or not Path(video_path).exists():
        log.warning("ffprobe not found or source missing — using default 768×512")
        return 768, 512

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            return int(parts[0]), int(parts[1])
    except Exception:
        log.warning("ffprobe failed to read dimensions from %s", video_path)
    return 768, 512


def _generate_stub_retake_video(
    source_video_path: str,
    output_path: Path,
    width: int,
    height: int,
    start_time_s: float,
    end_time_s: float,
    fps: int,
) -> None:
    """Generate a placeholder retake MP4.

    In production this will be replaced by streaming VAE decode piped to
    ffmpeg, then muxed back into the original video. For now we copy the
    original video's non-retake segments and insert a stub solid-colour
    clip for the retake region.

    Args:
        source_video_path: Absolute path to the source video.
        output_path: Where to write the output MP4.
        width: Video width.
        height: Video height.
        start_time_s: Start of the retake region in seconds.
        end_time_s: End of the retake region in seconds.
        fps: Frames per second.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(p).exists():
                ffmpeg_bin = p
                break
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    segment_duration = end_time_s - start_time_s

    # Stub: produce a solid-colour clip of the retake region duration.
    # In production: streaming VAE decode → retake segment → mux with original.
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x2e1a1a:s={width}x{height}:d={segment_duration:.3f}:r={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency=330:duration={segment_duration:.3f}:sample_rate=44100",
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
    log.info("Stub retake video created: %s", output_path)
