"""Extend pipeline — extend a video forward or backward from its boundary frame.

Extend is implemented as I2V generation conditioned on the boundary frame:
- Forward: extract last frame of source -> I2V at frame_idx=0 -> concat source + extension
- Backward: extract first frame of source -> I2V at frame_idx=last -> concat extension + source

Uses the same two-subprocess architecture as I2V (text encoding + generation).
"""

from __future__ import annotations

import inspect
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Literal

from engine.ffmpeg_utils import find_ffmpeg, find_ffprobe, has_audio_stream, probe_video_info
from engine.memory_manager import (
    aggressive_cleanup,
    build_memory_stats_from_subprocess,
    get_memory_stats,
    increment_generation_count,
    periodic_reload_check,
    reset_peak_memory,
)
from engine.mlx_runner import run_mlx_generation
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import GenerationResult

log = logging.getLogger(__name__)

# Output directory for extension results
OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "extensions"


def _find_ffmpeg() -> str:
    """Find ffmpeg binary — delegates to shared utility."""
    return find_ffmpeg()


def _find_ffprobe() -> str:
    """Find ffprobe binary — delegates to shared utility."""
    return find_ffprobe()


def _extract_boundary_frame(
    video_path: str,
    direction: str,
    output_path: str,
) -> None:
    """Extract the boundary frame from a video using ffmpeg.

    For forward extend, extracts the last frame.
    For backward extend, extracts the first frame.

    Args:
        video_path: Path to the source video.
        direction: "forward" (last frame) or "backward" (first frame).
        output_path: Where to save the extracted frame as PNG.
    """
    ffmpeg_bin = _find_ffmpeg()

    if direction == "backward":
        # Extract first frame
        cmd = [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "1",
            output_path,
        ]
    else:
        # Extract last frame: seek to 1s before end, then grab the first
        # frame from that point.  -sseof -0.1 is too tight and produces
        # empty output on many codecs; -1 is safe for any video >= 1s.
        cmd = [
            ffmpeg_bin, "-y",
            "-sseof", "-1",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "1",
            output_path,
        ]

    log.info("Extracting boundary frame (%s): %s", direction, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract boundary frame: {result.stderr[:500]}"
        )

    if not Path(output_path).exists():
        raise RuntimeError(
            f"Boundary frame extraction produced no output: {output_path}"
        )


def _concatenate_videos(
    first_path: str,
    second_path: str,
    output_path: str,
    fps: int,
) -> None:
    """Concatenate two videos using ffmpeg re-encoding for consistency.

    Re-encodes both inputs to ensure matching codec/resolution/fps.

    Args:
        first_path: Path to the first video (plays first).
        second_path: Path to the second video (plays after).
        output_path: Where to write the concatenated output.
        fps: Target frames per second.
    """
    ffmpeg_bin = _find_ffmpeg()

    # Use the concat filter for reliable concatenation with re-encoding.
    # This handles potential codec/resolution mismatches between source and
    # generated extension.
    cmd = [
        ffmpeg_bin, "-y",
        "-i", first_path,
        "-i", second_path,
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
        "-map", "[outv]",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path,
    ]

    # If both inputs have audio, concatenate audio too
    has_audio_0 = _has_audio_stream(first_path)
    has_audio_1 = _has_audio_stream(second_path)

    if has_audio_0 and has_audio_1:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", first_path,
            "-i", second_path,
            "-filter_complex",
            "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-r", str(fps),
            output_path,
        ]
    elif has_audio_0:
        # Only first has audio — take video from both, audio from first only
        cmd = [
            ffmpeg_bin, "-y",
            "-i", first_path,
            "-i", second_path,
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-r", str(fps),
            output_path,
        ]

    log.info("Concatenating videos: %s + %s -> %s", first_path, second_path, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Video concatenation failed: {result.stderr[:500]}")


def _has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains an audio stream — delegates to shared utility."""
    return has_audio_stream(video_path)


class ExtendPipeline:
    """Extends a video forward or backward using MLX I2V generation.

    Each extension generates new frames conditioned on the boundary frame
    of the source video, then concatenates the result.
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

        For "forward" extension: extracts the last frame of the source video,
        generates new frames conditioned on it (I2V at frame_idx=0), and
        concatenates source + extension.

        For "backward" extension: extracts the first frame of the source video,
        generates new frames conditioned on it (I2V at last frame), and
        concatenates extension + source.

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
        # Validate source video
        src_path = Path(source_video_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_video_path}")

        job_id = str(uuid.uuid4())[:8]
        stages: dict[str, float] = {}
        start_time = time.monotonic()

        async def _notify(
            step: int, total: int, pct: float, frame: str | None = None,
            *, status: str | None = None,
        ) -> None:
            if not progress_callback:
                return
            result = progress_callback(step, total, pct, frame, status=status)
            if inspect.isawaitable(result):
                await result

        reset_peak_memory()
        aggressive_cleanup()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"extend_{job_id}.mp4"

        # --- Stage 1: Probe source video ---
        await _notify(0, 0, 0.01, status="Analyzing source video")
        t0 = time.monotonic()
        source_width, source_height, source_duration = probe_video_info(source_video_path)
        log.info(
            "[%s] Source: %dx%d, %.2fs @ %dfps, direction=%s",
            job_id, source_width, source_height, source_duration, fps, direction,
        )
        stages["source_analysis"] = time.monotonic() - t0

        # --- Stage 2: Extract boundary frame ---
        await _notify(0, 0, 0.02, status="Extracting boundary frame")
        t0 = time.monotonic()
        tmp_dir = tempfile.mkdtemp(prefix="ltx_extend_")
        boundary_frame_path = str(Path(tmp_dir) / "boundary_frame.png")

        _extract_boundary_frame(source_video_path, direction, boundary_frame_path)
        log.info("[%s] Boundary frame extracted: %s", job_id, boundary_frame_path)
        stages["frame_extraction"] = time.monotonic() - t0
        aggressive_cleanup()

        # --- Stage 3: Generate extension via I2V ---
        # For forward: condition on frame 0 (default I2V behavior)
        # For backward: condition on the last frame of the new generation
        image_frame_idx = 0 if direction == "forward" else -1

        extension_path = str(Path(tmp_dir) / "extension.mp4")

        # Adapt mlx_runner progress to pipeline progress_callback format
        async def _progress_adapter(
            step: int, total_steps: int, stage: int, pct: float,
            *, status: str | None = None, preview_frame: str | None = None,
        ) -> None:
            # Scale inner progress (0.0-1.0) to overall pipeline range (0.05-0.90)
            scaled_pct = 0.05 + pct * 0.80
            label = status or "Generating extension"
            await _notify(step, total_steps, scaled_pct, preview_frame, status=label)

        log.info(
            "[%s] Starting I2V extension: %dx%d, %d frames, frame_idx=%s",
            job_id, source_width, source_height, extension_frames,
            "last" if image_frame_idx == -1 else image_frame_idx,
        )
        t0 = time.monotonic()

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=source_height,
            width=source_width,
            num_frames=extension_frames,
            seed=seed,
            fps=fps,
            output_path=extension_path,
            image=boundary_frame_path,
            image_strength=1.0,
            image_frame_idx=image_frame_idx,
            num_steps=steps,
            preview_interval=2,
            progress_callback=_progress_adapter,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()
        log.info("[%s] Extension generated: %s", job_id, extension_path)

        # --- Stage 4: Concatenate source + extension ---
        await _notify(0, 0, 0.92, status="Concatenating videos")
        t0 = time.monotonic()

        if direction == "forward":
            # source video first, then extension
            _concatenate_videos(source_video_path, extension_path, str(output_path), fps)
        else:
            # extension first, then source video
            _concatenate_videos(extension_path, source_video_path, str(output_path), fps)

        stages["concatenation"] = time.monotonic() - t0
        aggressive_cleanup()
        log.info("[%s] Videos concatenated: %s", job_id, output_path)

        # Clean up temp files
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

        await _notify(0, 0, 1.0, status="Complete")

        total_duration = time.monotonic() - start_time
        log.info("[%s] Extension complete in %.2fs -> %s", job_id, total_duration, output_path)

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
