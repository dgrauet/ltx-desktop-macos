"""Regression test: extend must run on the user-selected video model.

``_run_extend`` used to omit ``model_repo_id``, so extend always ran on the
default repo whatever model the user had selected (retake passed it correctly).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import main


def test_extend_forwards_selected_model(monkeypatch) -> None:
    monkeypatch.setattr(main, "selected_video_model", "dgrauet/ltx-2.3-mlx-q4")
    generate = AsyncMock(side_effect=RuntimeError("stop after capture"))
    monkeypatch.setattr(main.extend_pipeline, "generate", generate)
    monkeypatch.setattr(main, "_broadcast_progress", AsyncMock())
    main.jobs["extend-test"] = {"status": "queued"}

    req = main.ExtendRequest(source_video_path="/tmp/in.mp4", prompt="continue the scene")
    asyncio.run(main._run_extend("extend-test", req))

    assert generate.await_args.kwargs["model_repo_id"] == "dgrauet/ltx-2.3-mlx-q4"
