"""Retake pipeline -- regenerate a segment of an existing video."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from engine.ffmpeg_utils import probe_video_info
from engine.memory_manager import aggressive_cleanup, reset_peak_memory
from engine.mlx_runner import run_mlx_generation

log = logging.getLogger(__name__)

_VAE_TEMPORAL_FACTOR = 8
_OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "retakes"


@dataclass
class GenerationResult:
    job_id: str
    output_path: str
    duration_seconds: float
    memory_after: dict
    stages: dict[str, float] = field(default_factory=dict)


def _pixel_time_to_latent_frame(time_s: float, fps: int) -> int:
    pixel_frame = int(time_s * fps)
    return pixel_frame // _VAE_TEMPORAL_FACTOR


def _round_to_vae_compatible(num_frames: int) -> int:
    k = max(1, (num_frames - 1) // _VAE_TEMPORAL_FACTOR)
    return 1 + k * _VAE_TEMPORAL_FACTOR


class RetakePipeline:
    def __init__(self, model_manager) -> None:
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
        progress_callback=None,
    ) -> GenerationResult:
        job_id = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        aggressive_cleanup()
        reset_peak_memory()

        info = probe_video_info(source_video_path)
        width = info.get("width", 768)
        height = info.get("height", 512)
        duration = info.get("duration", 4.0)
        num_frames = int(duration * fps)
        num_frames = _round_to_vae_compatible(num_frames)

        start_frame = _pixel_time_to_latent_frame(start_time_s, fps)
        end_frame = _pixel_time_to_latent_frame(end_time_s, fps)

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_OUTPUT_DIR / f"retake_{job_id}.mp4")

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=output_path,
            mode="retake",
            num_steps=steps,
            retake_source=source_video_path,
            retake_start_frame=start_frame,
            retake_end_frame=end_frame,
            model_repo_id=model_repo_id,
            progress_callback=progress_callback,
        )

        aggressive_cleanup()
        elapsed = time.monotonic() - t0

        return GenerationResult(
            job_id=job_id,
            output_path=output_path,
            duration_seconds=elapsed,
            memory_after=gen_result.get("subprocess_memory", {}).get("after_generation", {}),
            stages={"retake": elapsed},
        )
