"""Local model pack registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import local_models


@pytest.fixture(autouse=True)
def _registry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(local_models, "REGISTRY_PATH", tmp_path / "reg" / "local_models.json")


def _pack(tmp_path: Path, name: str = "ltx-2.5-mlx-q4") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "embedded_config.json").write_text(json.dumps({"transformer": {"ff_bias": False}}))
    (d / "transformer-distilled.safetensors").write_bytes(b"")
    return d


def test_register_list_unregister(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    entry = local_models.register(pack)
    assert entry == {"id": "local-ltx-2.5-mlx-q4", "name": "ltx-2.5-mlx-q4", "path": str(pack.resolve())}
    assert local_models.list_local() == [entry]
    assert local_models.register(pack) == entry  # idempotent
    assert local_models.unregister(entry["id"])
    assert local_models.list_local() == []
    assert pack.is_dir()  # files untouched
    assert not local_models.unregister(entry["id"])


def test_ids_are_unique_for_same_dirname(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _pack(tmp_path / "a", "pack")
    b = _pack(tmp_path / "b", "pack")
    assert local_models.register(a)["id"] == "local-pack"
    assert local_models.register(b)["id"] == "local-pack-2"


@pytest.mark.parametrize("missing", ["config", "transformer", "dir"])
def test_validate_rejects_incomplete_packs(tmp_path: Path, missing: str) -> None:
    pack = _pack(tmp_path)
    if missing == "config":
        (pack / "embedded_config.json").unlink()
    elif missing == "transformer":
        (pack / "transformer-distilled.safetensors").unlink()
    else:
        pack = tmp_path / "nope"
    with pytest.raises(ValueError):
        local_models.register(pack)


def test_corrupt_registry_is_ignored(tmp_path: Path) -> None:
    local_models.REGISTRY_PATH.parent.mkdir(parents=True)
    local_models.REGISTRY_PATH.write_text("{oops")
    assert local_models.list_local() == []
