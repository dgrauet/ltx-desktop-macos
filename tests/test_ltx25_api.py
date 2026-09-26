"""LTX-2.5 API surface: local packs, model family/capabilities, feature gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from engine import local_models

client = TestClient(main.app)


def _pack(tmp_path: Path, name: str, family: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    transformer = {"ff_bias": False} if family == "2.5" else {}
    (d / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    (d / "transformer-distilled.safetensors").write_bytes(b"")
    return d


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(local_models, "REGISTRY_PATH", tmp_path / "local_models.json")
    monkeypatch.setattr(main, "selected_video_model", None)


def _register_and_select(tmp_path: Path, family: str) -> dict:
    pack = _pack(tmp_path, f"ltx-{family}-pack", family)
    r = client.post("/api/v1/models/local", json={"path": str(pack)})
    assert r.status_code == 200, r.text
    model = r.json()
    assert client.post("/api/v1/models/select", json={"model_id": model["id"]}).status_code == 200
    return model


def test_local_pack_listed_with_family_and_capabilities(tmp_path: Path) -> None:
    model = _register_and_select(tmp_path, "2.5")
    assert model["source"] == "local" and model["family"] == "2.5" and model["downloaded"]
    assert model["capabilities"]["auto_duration"] and not model["capabilities"]["ic_lora"]

    listed = {m["id"]: m for m in client.get("/api/v1/models").json()["models"]}
    assert listed[model["id"]]["family"] == "2.5"
    assert listed["ltx-2.5-mlx-q8"]["gated"] and listed["ltx-2.5-mlx-q8"]["family"] == "2.5"
    assert listed["ltx-2.3-mlx-q8"]["family"] == "2.3"
    assert listed["gemma-3-12b-it-4bit"]["capabilities"] is None


def test_register_rejects_non_pack(tmp_path: Path) -> None:
    r = client.post("/api/v1/models/local", json={"path": str(tmp_path)})
    assert r.status_code == 400


def test_local_pack_cannot_be_deleted_only_unregistered(tmp_path: Path) -> None:
    model = _register_and_select(tmp_path, "2.5")
    assert client.delete(f"/api/v1/models/{model['id']}").status_code == 400
    r = client.delete(f"/api/v1/models/local/{model['id']}")
    assert r.status_code == 200
    assert main.selected_video_model is None
    assert Path(model["hf_repo"]).is_dir()


def test_ic_lora_and_training_refused_on_25(tmp_path: Path) -> None:
    _register_and_select(tmp_path, "2.5")
    r = client.post("/api/v1/generate/ic-lora", json={"prompt": "x", "source_control_path": "/tmp/c.mp4"})
    assert r.status_code == 400 and "LTX-2.5" in r.json()["detail"]
    r = client.post("/api/v1/training/runs", json={"dataset_id": "ds", "steps": 10})
    assert r.status_code == 400


@pytest.mark.parametrize(
    "field", [{"auto_duration": True}, {"generated_keyframes": 2}, {"video_decoder": "diffusion"}]
)
def test_25_options_refused_on_23(tmp_path: Path, field: dict) -> None:
    _register_and_select(tmp_path, "2.3")
    r = client.post("/api/v1/generate/text-to-video", json={"prompt": "x", **field})
    assert r.status_code == 400


@pytest.mark.parametrize(
    ("family", "body", "ok"),
    [
        ("2.3", {"enable_teacache": True, "pipeline_type": "two-stage"}, True),
        ("2.3", {"enable_teacache": True, "pipeline_type": "distilled"}, False),
        ("2.5", {"enable_teacache": True, "pipeline_type": "two-stage-hq"}, False),
        ("2.3", {"segments": ["a", " "]}, False),
        ("2.5", {"segments": ["shot one", "shot two"], "generate_audio": False}, True),
    ],
)
def test_relay_audio_teacache_validation(tmp_path: Path, monkeypatch, family, body, ok) -> None:
    _register_and_select(tmp_path, family)
    monkeypatch.setattr(main, "_system_ram_gb", lambda: 128.0)
    req = main.T2VRequest(prompt="x", **body)
    if ok:
        main._apply_family_rules(req)
    else:
        with pytest.raises(main.HTTPException) as exc:
            main._apply_family_rules(req)
        assert exc.value.status_code == 400


def test_loras_refused_on_25(tmp_path: Path) -> None:
    _register_and_select(tmp_path, "2.5")
    r = client.post("/api/v1/generate/text-to-video", json={"prompt": "x", "lora_ids": ["mine"]})
    assert r.status_code == 400 and "LTX-2.3" in r.json()["detail"]


def test_25_forces_low_ram_on_small_macs(tmp_path: Path, monkeypatch) -> None:
    _register_and_select(tmp_path, "2.5")
    monkeypatch.setattr(main, "_system_ram_gb", lambda: 32.0)
    req = main.T2VRequest(prompt="x", auto_duration=True)
    main._apply_family_rules(req)
    assert req.low_ram

    monkeypatch.setattr(main, "_system_ram_gb", lambda: 128.0)
    req = main.T2VRequest(prompt="x")
    main._apply_family_rules(req)
    assert not req.low_ram
