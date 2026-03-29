"""Extend pipeline -- add frames before or after an existing video."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from engine.memory_manager import aggressive_cleanup, reset_peak_memory
from engine.mlx_runner import run_mlx_generation

log = logging.getLogger(__name__)

_OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "extensions"


@dataclass
class GenerationResult:
    job_id: str
    output_path: str
    duration_seconds: float
    memory_after: dict
    stages: dict[str, float] = field(default_factory=dict)


class ExtendPipeline:
    def __init__(self, model_manager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        source_video_path: str,
        prompt: str,
        direction: str,
        extension_frames: int = 49,
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

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_OUTPUT_DIR / f"extend_{job_id}.mp4")

        # Map "forward"/"backward" to library "after"/"before"
        lib_direction = "after" if direction == "forward" else "before"

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=0,
            width=0,
            num_frames=extension_frames,
            seed=seed,
            fps=fps,
            output_path=output_path,
            mode="extend",
            num_steps=steps,
            extend_source=source_video_path,
            extend_frames=extension_frames,
            extend_direction=lib_direction,
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
            stages={"extend": elapsed},
        )
