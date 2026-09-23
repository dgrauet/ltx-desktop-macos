"""Regression test: the UI's pixel-frame extension count must reach the lib as latent frames.

``RetakePipeline.extend_from_video(extend_frames=...)`` counts *latent* frames
(8 pixel frames each). The API/UI speak pixel frames (e.g. 49 ≈ 2 s at 24 fps);
forwarding them unconverted asked the lib for 49 latent frames ≈ 392 pixel
frames — an ~16 s extension that OOMs on 32 GB.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from engine.pipelines import extend as extend_mod


@pytest.mark.parametrize(("pixel_frames", "latent_frames"), [(9, 1), (25, 3), (49, 6), (97, 12)])
def test_extension_frames_converted_to_latent(monkeypatch, pixel_frames: int, latent_frames: int) -> None:
    run = AsyncMock(return_value={})
    monkeypatch.setattr(extend_mod, "run_mlx_generation", run)

    asyncio.run(extend_mod.ExtendPipeline(None).generate(
        source_video_path="/tmp/in.mp4", prompt="more", direction="forward",
        extension_frames=pixel_frames,
    ))

    assert run.await_args.kwargs["extend_frames"] == latent_frames
