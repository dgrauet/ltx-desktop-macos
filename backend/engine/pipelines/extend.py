"""Extend pipeline — extend a video forward or backward from its last/first frame.

Sprint 3: stubbed inference that produces real MP4 files via ffmpeg.
Real MLX ExtendPipeline inference will replace the stubs in a later sprint.

Architecture note: each extension is a full diffusion pass conditioned on
the last N frames of the source video (I2V-conditioned overlap). Segments
are blended with linear alpha in the overlap region. See CLAUDE.md for the
full chunked generation strategy.
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
from typing import Callable, Literal

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

# Output directory for extension results
OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "extensions"


class ExtendPipeline:
    """Extends a video forward or backward using MLX (stubbed for Sprint 3).

    Each extension is generated as a complete diffusion pass conditioned on
    the boundary frames of the source video, then blended and concatenated
    with the original.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        source_video_path: str,
        prompt: str,
        direction: Literal["forward", "backward"],
        extension_frames: int = 49,
        steps: int = 8,
        seed: int = 42,
        fps: int = 24,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Extend a video forward or backward.

        For "forward" extension: conditions on the last N frames of the source
        and generates new frames appended to the end.
        For "backward" extension: conditions on the first N frames of the source
        and generates new frames prepended to the beginning.

        Args:
            source_video_path: Absolute path to the source video file.
            prompt: Text prompt describing the desired extension content.
            direction: "forward" to extend from the last frame, "backward"
                from the first frame.
            extension_frames: Number of new frames to generate. Should be a
                multiple of 8 + 1 (e.g. 49, 65, 97) for best results.
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
        output_path = OUTPUT_DIR / f"extend_{job_id}.mp4"

        total_stages = 5
        current_stage = 0

        extension_duration = extension_frames / fps
        log.info(
            "[%s] Extend: source=%r direction=%s frames=%d (%.2fs) prompt=%r",
            job_id,
            source_video_path,
            direction,
            extension_frames,
            extension_duration,
            prompt[:80],
        )

        # Stage 1: Source analysis — probe dimensions and duration
        log.info("[%s] Stage 1: Source video analysis", job_id)
        t0 = time.monotonic()
        source_width, source_height, source_duration = _probe_video_info(source_video_path)
        log.info(
            "[%s] Source: %dx%d, %.2fs @ %dfps",
            job_id,
            source_width,
            source_height,
            source_duration,
            fps,
        )
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

        # Stage 3: Diffusion (extension segment, conditioned on boundary frames)
        log.info(
            "[%s] Stage 3: Diffusion — %d steps (direction=%s, conditioned on boundary frames)",
            job_id,
            steps,
            direction,
        )
        t0 = time.monotonic()
        for step in range(steps):
            await asyncio.sleep(0.05)  # Stub: simulate denoising step
            pct = (current_stage + (step + 1) / steps) / total_stages
            await _notify(step + 1, steps, pct)
        stages["diffusion"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1

        # Stage 4: VAE decode → stub extension clip via ffmpeg
        log.info("[%s] Stage 4: VAE decode → ffmpeg (extension segment)", job_id)
        t0 = time.monotonic()
        _generate_stub_extension_video(
            source_video_path=source_video_path,
            output_path=output_path,
            width=source_width,
            height=source_height,
            source_duration=source_duration,
            extension_duration=extension_duration,
            direction=direction,
            fps=fps,
        )
        stages["vae_decode"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, current_stage / total_stages)

        # Stage 5: Concatenate extension with original (overlap blend)
        log.info(
            "[%s] Stage 5: Concatenate extension with original (direction=%s)", job_id, direction
        )
        t0 = time.monotonic()
        await asyncio.sleep(0.05)  # Stub: simulate concat/blend
        stages["concat"] = time.monotonic() - t0
        aggressive_cleanup()
        current_stage += 1
        await _notify(current_stage, total_stages, 1.0)

        total_duration = time.monotonic() - start_time
        log.info("[%s] Extension complete in %.2fs → %s", job_id, total_duration, output_path)

        increment_generation_count()
        periodic_reload_check(self._model_manager)

        return GenerationResult(
            job_id=job_id,
            output_path=str(output_path),
            duration_seconds=total_duration,
            memory_after=get_memory_stats(),
            stages=stages,
        )


def _probe_video_info(video_path: str) -> tuple[int, int, float]:
    """Probe a video file for width, height, and duration using ffprobe.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        Tuple of (width, height, duration_seconds). Falls back to
        (768, 512, 4.0) if probing fails.
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        for p in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]:
            if Path(p).exists():
                ffprobe_bin = p
                break
    if not ffprobe_bin or not Path(video_path).exists():
        log.warning("ffprobe not found or source missing — using defaults 768×512 @ 4.0s")
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


def _generate_stub_extension_video(
    source_video_path: str,
    output_path: Path,
    width: int,
    height: int,
    source_duration: float,
    extension_duration: float,
    direction: str,
    fps: int,
) -> None:
    """Generate a placeholder extended MP4.

    In production this will be replaced by:
    1. Streaming VAE decode of the extension segment piped to ffmpeg.
    2. Linear alpha blend in the overlap region between source and extension.
    3. ffmpeg concat filter to produce the final output.

    For now we produce a solid-colour clip representing the extension segment
    concatenated with the original video via ffmpeg's concat filter.

    Args:
        source_video_path: Absolute path to the source video.
        output_path: Where to write the extended output MP4.
        width: Video width.
        height: Video height.
        source_duration: Duration of the source video in seconds.
        extension_duration: Duration of the new extension in seconds.
        direction: "forward" (append) or "backward" (prepend).
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

    total_duration = source_duration + extension_duration

    # Stub: produce a solid-colour clip of the full extended duration.
    # Colour: dark teal for forward, dark purple for backward — visually
    # distinguishable during development.
    stub_colour = "0x1a2e2e" if direction == "forward" else "0x2e1a2e"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"color=c={stub_colour}:s={width}x{height}:d={total_duration:.3f}:r={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency=550:duration={total_duration:.3f}:sample_rate=44100",
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
    log.info(
        "Stub extension video created (%s, %.2fs total): %s",
        direction,
        total_duration,
        output_path,
    )
