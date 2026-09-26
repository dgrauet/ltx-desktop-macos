"""LTX family detection must agree with the lib's own 2.5 pack detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.model_family import (
    FAMILY_23,
    FAMILY_25,
    capabilities,
    detect_family,
    family_from_repo_name,
)


def _pack(tmp_path: Path, name: str, config: dict | None, filename: str = "embedded_config.json") -> Path:
    d = tmp_path / name
    d.mkdir()
    if config is not None:
        (d / filename).write_text(json.dumps(config))
    return d


@pytest.mark.parametrize(
    ("config", "filename", "family"),
    [
        ({"transformer": {"ff_bias": False}, "model_version": "2.5.0"}, "embedded_config.json", FAMILY_25),
        ({"transformer": {"num_layers": 48}}, "embedded_config.json", FAMILY_23),
        ({"ff_bias": False}, "config.json", FAMILY_25),
        (None, "embedded_config.json", FAMILY_23),
    ],
)
def test_detect_family(tmp_path: Path, config, filename: str, family: str) -> None:
    assert detect_family(_pack(tmp_path, "pack", config, filename)) == family


def test_detect_family_matches_lib(tmp_path: Path) -> None:
    from ltx_pipelines_mlx.utils.generation import is_ltx25_pack

    for i, cfg in enumerate([{"transformer": {"ff_bias": False}}, {"transformer": {}}]):
        pack = _pack(tmp_path, f"p{i}", cfg)
        assert (detect_family(pack) == FAMILY_25) == is_ltx25_pack(pack)


def test_corrupt_config_falls_back_to_23(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / "embedded_config.json").write_text("{not json")
    assert detect_family(d) == FAMILY_23


def test_family_from_repo_name() -> None:
    assert family_from_repo_name("dgrauet/ltx-2.5-mlx-q8") == FAMILY_25
    assert family_from_repo_name("dgrauet/ltx-2.3-mlx-q8") == FAMILY_23


def test_capabilities() -> None:
    c23, c25 = capabilities(FAMILY_23), capabilities(FAMILY_25)
    assert c23["ic_lora"] and c23["training"] and not c23["auto_duration"]
    assert not c25["ic_lora"] and not c25["training"] and c25["auto_duration"]
    assert c23["enhance"] and c25["enhance"]
    c23["ic_lora"] = False
    assert capabilities(FAMILY_23)["ic_lora"]
