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
        auto_duration=False, generated_keyframes=0, segment=None, enable_teacache=False,
        gemma=None, low_ram=False, ic_lora=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


PIPELINES = ["distilled", "one-stage", "two-stage", "two-stage-hq"]


@pytest.mark.parametrize("pipeline_type", PIPELINES)
@pytest.mark.parametrize(
    "extra",
    [
        {}, {"mode": "i2v", "image": "/tmp/i.png"}, {"auto_duration": True}, {"generated_keyframes": 2},
        {"segment": ["a fox walks", "the fox jumps"]},
    ],
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


@pytest.mark.parametrize("pipeline_type", ["two-stage", "two-stage-hq"])
def test_teacache_binds_on_two_stage(pipeline_type: str) -> None:
    args = _args(pipeline_type, enable_teacache=True)
    cls, _ = resolve_pipeline(args)
    kwargs = build_t2v_gen_kwargs(args)
    assert kwargs["enable_teacache"] is True
    inspect.signature(cls.generate_and_save).bind(None, **kwargs)


def test_segments_become_prompt_relay() -> None:
    from ltx_core_mlx.conditioning.prompt_relay import PromptRelayInput

    kwargs = build_t2v_gen_kwargs(_args("distilled", segment=["shot one", "shot two"]))
    relay = kwargs["prompt_relay"]
    assert isinstance(relay, PromptRelayInput)
    assert relay.local_prompts == ["shot one", "shot two"] and relay.segment_lengths is None


def test_optional_kwargs_absent_by_default() -> None:
    kwargs = build_t2v_gen_kwargs(_args("two-stage"))
    assert "prompt_relay" not in kwargs and "enable_teacache" not in kwargs
