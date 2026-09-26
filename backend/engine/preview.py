"""Progressive diffusion previews for the generation subprocess.

Wraps the library's ``StepwisePreview`` (ltx-pipelines-mlx >= 0.14.21), which
decodes a window of the x0 prediction every N denoising steps and writes it as a
WebP into a directory. After each step we announce every newly written file on
stderr as ``PREVIEW:<path>``; ``mlx_runner`` reads, base64-encodes and deletes it,
then forwards it to the UI over the progress WebSocket.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ltx_pipelines_mlx.utils.stepwise import StepwiseConfig, StepwisePreview

# Decode a single latent frame per preview: one still image, the cheapest point
# on the decoder's cost curve (the UI shows a still, not a motion clip).
PREVIEW_LATENT_FRAMES = 1
# Preview every N denoising steps (the last step is always previewed).
PREVIEW_INTERVAL = 2


class EmittingPreview(StepwisePreview):
    """``StepwisePreview`` that reports each written preview file via ``emit``."""

    def __init__(self, config: StepwiseConfig, emit: Callable[[str], None]) -> None:
        super().__init__(config, verbose=False)
        self._emit = emit
        self._seen: set[Path] = set()

    def bind(self, **kwargs):
        on_step = super().bind(**kwargs)
        if on_step is None:
            return None

        def emitting_on_step(step_idx, num_steps, video_x0, sigma) -> None:
            on_step(step_idx, num_steps, video_x0, sigma)
            for path in sorted(self.config.output_dir.glob("*.webp")):
                if path not in self._seen:
                    self._seen.add(path)
                    self._emit(f"PREVIEW:{path}")

        return emitting_on_step


def make_preview(
    output_dir: str, *, fps: int, seed: int, emit: Callable[[str], None],
) -> EmittingPreview:
    """Build the preview handler to assign to ``pipeline.stepwise``."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config = StepwiseConfig(
        output_dir=directory,
        interval=PREVIEW_INTERVAL,
        frames=PREVIEW_LATENT_FRAMES,
        frame_rate=float(fps),
        seed=seed,
    )
    return EmittingPreview(config, emit)
