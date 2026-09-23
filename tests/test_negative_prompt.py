"""Negative prompt: forwarded to every CFG entry point, refused on the distilled pipeline."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from engine.generate_v23 import build_t2v_gen_kwargs, resolve_pipeline
from tests.test_t2v_kwargs import _args


@pytest.mark.parametrize("pipeline_type", ["one-stage", "two-stage", "two-stage-hq"])
def test_t2v_negative_prompt_binds_on_cfg_pipelines(pipeline_type: str) -> None:
    args = _args(pipeline_type, negative_prompt="blurry, low quality")
    cls, _ = resolve_pipeline(args)
    kwargs = build_t2v_gen_kwargs(args)
    assert kwargs["negative_prompt"] == "blurry, low quality"
    inspect.signature(cls.generate_and_save).bind(None, **kwargs)


def test_empty_negative_prompt_is_forwarded_not_dropped() -> None:
    assert build_t2v_gen_kwargs(_args("two-stage", negative_prompt=""))["negative_prompt"] == ""


def test_none_keeps_lib_default() -> None:
    assert "negative_prompt" not in build_t2v_gen_kwargs(_args("two-stage", negative_prompt=None))


@pytest.mark.parametrize(
    ("cls_name", "method"),
    [
        ("A2VidPipelineTwoStage", "generate_and_save"),
        ("RetakePipeline", "retake_from_video"),
        ("RetakePipeline", "extend_from_video"),
    ],
)
def test_other_cfg_entry_points_accept_negative_prompt(cls_name: str, method: str) -> None:
    import ltx_pipelines_mlx

    params = inspect.signature(getattr(getattr(ltx_pipelines_mlx, cls_name), method)).parameters
    assert "negative_prompt" in params and params["negative_prompt"].default is None


def test_api_refuses_negative_prompt_on_distilled(monkeypatch) -> None:
    import main

    monkeypatch.setattr(main, "_selected_family", lambda: "2.3")
    req = main.T2VRequest(prompt="x", negative_prompt="ugly", pipeline_type="distilled")
    with pytest.raises(main.HTTPException) as exc:
        main._apply_family_rules(req)
    assert exc.value.status_code == 400

    ok = main.T2VRequest(prompt="x", negative_prompt="ugly", pipeline_type="two-stage")
    main._apply_family_rules(ok)
