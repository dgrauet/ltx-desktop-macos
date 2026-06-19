"""Audio-to-Video (A2V) generation pipeline.

Takes a reference audio track that conditions the generated video. Uses the
library's ``A2VidPipelineTwoStage`` (two-stage Euler + CFG, beta tier).
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

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

OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs"


class AudioToVideoPipeline:
    """Audio-to-Video generation pipeline using MLX."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        prompt: str,
        source_audio_path: str,
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        steps: int = 30,
        seed: int = 42,
        guidance_scale: float = 3.0,
        fps: int = 24,
        audio_start: float = 0.0,
        low_ram: bool = False,
        lora_args: list[str] | None = None,
        model_repo_id: str | None = None,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Run the A2V generation pipeline.

        Args:
            prompt: Text prompt describing the video.
            source_audio_path: Path to the reference audio (WAV/MP3/etc.).
            width: Output video width.
            height: Output video height.
            num_frames: Number of frames to generate.
            steps: Stage-1 denoising steps (A2V is two-stage; default 30).
            seed: Random seed for reproducibility.
            guidance_scale: CFG guidance scale.
            fps: Output frames per second.
            audio_start: Start time in the audio file, in seconds.
            low_ram: Stream DiT blocks from disk to reduce RAM.
            lora_args: Optional LoRA path:strength strings.
            model_repo_id: Optional video model HF repo ID.
            progress_callback: Optional callback(step, total_steps, pct, preview_frame).

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        # Validate source audio
        audio_path = Path(source_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Source audio not found: {source_audio_path}")

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
        output_path = OUTPUT_DIR / f"a2v_{job_id}.mp4"

        # Runner calls: callback(step, total, pct, preview_frame, status=...)
        async def _progress_adapter(
            step: int, total_steps: int, pct: float,
            preview_frame: str | None = None,
            *, status: str | None = None,
        ) -> None:
            await _notify(step, total_steps, pct, preview_frame, status=status)

        log.info(
            "[%s] Starting A2V generation: prompt=%r, audio=%s, %dx%d, %d frames",
            job_id, prompt[:80], source_audio_path, width, height, num_frames,
        )
        t0 = time.monotonic()

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=str(output_path),
            mode="a2v",
            audio=source_audio_path,
            audio_start=audio_start,
            cfg_scale=guidance_scale,
            num_steps=steps,
            low_ram=low_ram,
            lora_args=lora_args,
            progress_callback=_progress_adapter,
            model_repo_id=model_repo_id,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()

        total_duration = time.monotonic() - start_time
        log.info("[%s] A2V generation complete in %.2fs", job_id, total_duration)

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
