"""T2V/I2V gen kwargs (incl. LTX-2.5 options) must bind to every pipeline's generate_and_save."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from engine.generate_v23 import build_t2v_gen_kwargs, resolve_pipeline


def _args(pipeline_type: str, **over) -> SimpleNamespace:
    base = dict(
        mode="t2v", pipeline_type=pipeline_type, prompt="a fox", output_path="/tmp/o.mp4",
        height=512, width=768, num_frames=97, fps=24, seed=1, num_steps=8,
        cfg_scale=3.0, stg_scale=1.0, image=None, image_strength=1.0,
        auto_duration=False, generated_keyframes=0,
        gemma=None, low_ram=False, ic_lora=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


PIPELINES = ["distilled", "one-stage", "two-stage", "two-stage-hq"]


@pytest.mark.parametrize("pipeline_type", PIPELINES)
@pytest.mark.parametrize(
    "extra",
    [{}, {"mode": "i2v", "image": "/tmp/i.png"}, {"auto_duration": True}, {"generated_keyframes": 2}],
)
def test_kwargs_bind(pipeline_type: str, extra: dict) -> None:
    args = _args(pipeline_type, **extra)
    cls, _ = resolve_pipeline(args)
    kwargs = build_t2v_gen_kwargs(args)
    inspect.signature(cls.generate_and_save).bind(None, **kwargs)


def test_auto_duration_replaces_num_frames() -> None:
    from ltx_pipelines_mlx.utils.types import AutoDuration

    kwargs = build_t2v_gen_kwargs(_args("distilled", auto_duration=True))
    assert isinstance(kwargs["num_frames"], AutoDuration)


def test_keyframes_omitted_when_zero() -> None:
    assert "generated_keyframes" not in build_t2v_gen_kwargs(_args("distilled"))
