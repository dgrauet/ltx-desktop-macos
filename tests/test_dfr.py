"""DFR (LTX-2.5 diffusion fidelity rendering): kwargs bind to the lib; API validation."""

from __future__ import annotations

import inspect

import pytest

from engine.generate_v23 import build_t2v_gen_kwargs, resolve_pipeline
from tests.test_t2v_kwargs import _args


@pytest.mark.parametrize(("spatial", "temporal"), [(1, 0), (2, 0), (1, 2), (2, 1)])
@pytest.mark.parametrize("low_ram", [False, True])
def test_dfr_ctor_and_gen_kwargs_bind(spatial: int, temporal: int, low_ram: bool) -> None:
    args = _args("dfr", low_ram=low_ram, dfr_spatial_upscalings=spatial, dfr_temporal_upscalings=temporal,
                 auto_duration=True, model_dir="/tmp/m")
    cls, ctor = resolve_pipeline(args)
    assert cls.__name__ == "DFRPipeline"
    assert ctor["spatial_upscalings"] == spatial and ctor["temporal_upscalings"] == temporal
    inspect.signature(cls.__init__).bind(None, args.model_dir, **ctor)
    gen = build_t2v_gen_kwargs(args)
    assert "cfg_scale" not in gen and gen["stage1_steps"] == args.num_steps
    inspect.signature(cls.generate_and_save).bind(None, **gen)


def _rules(monkeypatch, family: str, **body):
    import main

    monkeypatch.setattr(main, "_selected_family", lambda: family)
    monkeypatch.setattr(main, "_system_ram_gb", lambda: 128.0)
    req = main.T2VRequest(prompt="x", pipeline_type="dfr", **body)
    main._apply_family_rules(req)
    return req


@pytest.mark.parametrize(
    ("family", "body"),
    [
        ("2.3", {}),
        ("2.5", {"generated_keyframes": 2}),
        ("2.5", {"negative_prompt": "blur"}),
        ("2.5", {"segments": ["a", "b"], "dfr_temporal_upscalings": 1}),
        ("2.5", {"enable_teacache": True}),
    ],
)
def test_dfr_refusals(monkeypatch, family: str, body: dict) -> None:
    import main

    with pytest.raises(main.HTTPException) as exc:
        _rules(monkeypatch, family, **body)
    assert exc.value.status_code == 400


def test_dfr_accepted_on_25(monkeypatch) -> None:
    _rules(monkeypatch, "2.5", segments=["a", "b"], dfr_spatial_upscalings=1)
    _rules(monkeypatch, "2.5", dfr_spatial_upscalings=2, dfr_temporal_upscalings=2, auto_duration=True)


def test_output_fps_doubles_per_temporal_round() -> None:
    import main

    assert main._output_fps(main.T2VRequest(prompt="x", pipeline_type="dfr", dfr_temporal_upscalings=2)) == 96
    assert main._output_fps(main.T2VRequest(prompt="x", dfr_temporal_upscalings=2)) == 24
