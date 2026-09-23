"""Regression test: pipeline constructor kwargs must bind to the library signatures.

ltx-2-mlx 0.14.14's RetakePipeline did not accept ``low_ram_streaming``, so every
retake/extend raised TypeError at real generation time — invisible to import and
route smoke tests. This binds the kwargs our subprocess builds against each
pipeline class's actual ``__init__`` signature, for every mode/pipeline_type.
"""

import inspect
from types import SimpleNamespace

import pytest

from engine.generate_v23 import resolve_pipeline

_CASES = [
    ("t2v", "distilled"),
    ("t2v", "one-stage"),
    ("t2v", "two-stage"),
    ("t2v", "two-stage-hq"),
    ("i2v", "distilled"),
    ("a2v", "distilled"),
    ("ic-lora", "distilled"),
    ("retake", "distilled"),
    ("extend", "distilled"),
]


@pytest.mark.parametrize("low_ram", [False, True])
@pytest.mark.parametrize(("mode", "pipeline_type"), _CASES)
def test_ctor_kwargs_bind_to_library_signature(mode: str, pipeline_type: str, low_ram: bool) -> None:
    args = SimpleNamespace(
        mode=mode,
        pipeline_type=pipeline_type,
        gemma=None,
        low_ram=low_ram,
        ic_lora=["/tmp/control-lora.safetensors:1.0"],
        model_dir="/tmp/model",
    )
    cls, kwargs = resolve_pipeline(args)
    # raises TypeError on any unexpected/missing kwarg; model_dir is positional
    inspect.signature(cls.__init__).bind(None, args.model_dir, **kwargs)
