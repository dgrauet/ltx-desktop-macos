"""Tests for training_store — disk-backed training-run persistence."""
from __future__ import annotations

import pytest
from training_store import create_run, update_run, get_run, list_runs, delete_run


def test_run_lifecycle(tmp_path, monkeypatch):
    import training_store as ts
    monkeypatch.setattr(ts, "TRAINING_DIR", tmp_path)
    monkeypatch.setattr(ts, "RUNS_DIR", tmp_path / "runs")
    r = create_run("r1", dataset_id="d1", config_path="/x/c.yaml", created_at="2026-06-23T00:00:00")
    assert r["status"] == "pending"
    update_run("r1", status="training", peak_mem_gb=7.0)
    assert get_run("r1")["status"] == "training"
    assert get_run("r1")["peak_mem_gb"] == 7.0
    assert [x["run_id"] for x in list_runs()] == ["r1"]
    assert delete_run("r1") is True
    assert get_run("r1") is None


def test_corrupt_run_json_is_ignored(tmp_path, monkeypatch):
    import training_store as ts
    monkeypatch.setattr(ts, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs" / "bad").mkdir(parents=True)
    (tmp_path / "runs" / "bad" / "run.json").write_text("{not json")
    assert list_runs() == []
