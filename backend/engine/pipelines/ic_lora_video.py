"""IC-LoRA controlled generation pipeline.

Conditions generation on a reference control video (depth/pose/edges) via the
library's ``ICLoraPipeline`` (two-stage Euler+CFG, stable tier). Optionally
extracts canny edges from a normal video first.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from engine.ffmpeg_utils import extract_edges, probe_frame_count
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


class IcLoraVideoPipeline:
    """IC-LoRA controlled video generation pipeline using MLX."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        prompt: str,
        source_control_path: str,
        ic_lora_path: str,
        *,
        extract_edges_first: bool = False,
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        steps: int = 30,
        seed: int = 42,
        guidance_scale: float = 3.0,
        fps: int = 24,
        control_strength: float = 1.0,
        ic_lora_strength: float = 1.0,
        conditioning_strength: float = 1.0,
        skip_stage_2: bool = False,
        low_ram: bool = False,
        model_repo_id: str | None = None,
        progress_callback: Callable[[int, int, float, str | None], None] | None = None,
    ) -> GenerationResult:
        """Run the IC-LoRA controlled generation pipeline.

        Args:
            prompt: Text prompt describing the video.
            source_control_path: Path to the control video (or a normal video
                if ``extract_edges_first`` is True).
            ic_lora_path: Path to the IC-LoRA ``.safetensors`` weights.
            extract_edges_first: If True, render canny edges from the source
                video and use that as the control signal.
            (others): generation parameters; ``steps`` is stage-1 step count.

        Returns:
            GenerationResult with output path, timing, and memory stats.
        """
        control_in = Path(source_control_path)
        if not control_in.exists():
            raise FileNotFoundError(f"Control video not found: {source_control_path}")
        if not ic_lora_path:
            raise ValueError("ic_lora_path is required for IC-LoRA generation")
        # IC-LoRA's reference VAE encoder needs a (1 + 8k)-frame input and the
        # library forces a minimum of 9 frames; a shorter control video crashes
        # deep in the VAE with a cryptic reshape error. Reject it up front.
        control_frames = probe_frame_count(source_control_path)
        if control_frames and control_frames < 9:
            raise ValueError(
                f"Control video is too short ({control_frames} frames). "
                "IC-LoRA needs a control video of at least 9 frames."
            )

        job_id = str(uuid.uuid4())[:8]
        stages: dict[str, float] = {}
        start_time = time.monotonic()

        async def _notify(step, total, pct, frame=None, *, status=None):
            if not progress_callback:
                return
            result = progress_callback(step, total, pct, frame, status=status)
            if inspect.isawaitable(result):
                await result

        reset_peak_memory()
        aggressive_cleanup()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        control_video = source_control_path
        if extract_edges_first:
            await _notify(0, 0, 0.0, status="Extracting edges")
            edges_path = OUTPUT_DIR / f"iclora_edges_{job_id}.mp4"
            extract_edges(source_control_path, str(edges_path))
            control_video = str(edges_path)

        output_path = OUTPUT_DIR / f"iclora_{job_id}.mp4"

        async def _progress_adapter(step, total_steps, pct, preview_frame=None, *, status=None):
            await _notify(step, total_steps, pct, preview_frame, status=status)

        log.info(
            "[%s] Starting IC-LoRA generation: prompt=%r, control=%s, %dx%d, %d frames",
            job_id, prompt[:80], control_video, width, height, num_frames,
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
            mode="ic-lora",
            control_video=control_video,
            control_strength=control_strength,
            ic_lora_path=ic_lora_path,
            ic_lora_strength=ic_lora_strength,
            conditioning_strength=conditioning_strength,
            skip_stage_2=skip_stage_2,
            cfg_scale=guidance_scale,
            num_steps=steps,
            low_ram=low_ram,
            progress_callback=_progress_adapter,
            model_repo_id=model_repo_id,
        )

        stages["generation"] = time.monotonic() - t0
        aggressive_cleanup()

        total_duration = time.monotonic() - start_time
        log.info("[%s] IC-LoRA generation complete in %.2fs", job_id, total_duration)

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
