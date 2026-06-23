"""Training dataset API tests — J4b P1 Task 5.

Uses FastAPI TestClient so no running backend is required. TRAINING_DIR is
monkeypatched to tmp_path to guarantee test isolation.
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient

import dataset_store
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_training_dir(monkeypatch, tmp_path: Path) -> None:
    """Redirect dataset_store.TRAINING_DIR to a temporary directory."""
    monkeypatch.setattr(dataset_store, "TRAINING_DIR", tmp_path)


# ---------------------------------------------------------------------------
# POST /api/v1/training/datasets
# ---------------------------------------------------------------------------


def test_create_dataset_returns_200(monkeypatch, tmp_path):
    """Creating a new dataset returns 200 with the dataset_id."""
    _patch_training_dir(monkeypatch, tmp_path)
    r = client.post("/api/v1/training/datasets", json={"dataset_id": "myds"})
    assert r.status_code == 200
    assert r.json()["dataset_id"] == "myds"


def test_create_dataset_makes_clips_dir(monkeypatch, tmp_path):
    """Creating a dataset materialises clips/ on disk."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "clips-test"})
    assert (tmp_path / "clips-test" / "clips").is_dir()


# ---------------------------------------------------------------------------
# GET /api/v1/training/datasets
# ---------------------------------------------------------------------------


def test_list_datasets_contains_created(monkeypatch, tmp_path):
    """GET list returns the dataset that was just created."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "listme"})
    r = client.get("/api/v1/training/datasets")
    assert r.status_code == 200
    ids = [d["id"] for d in r.json()]
    assert "listme" in ids


def test_list_datasets_empty_when_none(monkeypatch, tmp_path):
    """GET list returns an empty array when no datasets exist."""
    _patch_training_dir(monkeypatch, tmp_path)
    r = client.get("/api/v1/training/datasets")
    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# POST /api/v1/training/datasets/{id}/clips
# ---------------------------------------------------------------------------


def test_upload_clip_saves_file(monkeypatch, tmp_path):
    """Uploading a clip saves it under clips_dir."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "upload-ds"})
    fake_video = b"\x00\x01\x02\x03"  # content doesn't matter for this test
    r = client.post(
        "/api/v1/training/datasets/upload-ds/clips",
        files={"file": ("test.mp4", io.BytesIO(fake_video), "video/mp4")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test.mp4"
    saved = dataset_store.clips_dir("upload-ds") / "test.mp4"
    assert saved.exists()
    assert saved.read_bytes() == fake_video


# ---------------------------------------------------------------------------
# PUT /api/v1/training/datasets/{id}/manifest
# ---------------------------------------------------------------------------


def test_put_manifest_empty_caption_returns_400(monkeypatch, tmp_path):
    """PUT manifest with an empty caption must return HTTP 400."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "manifest-ds"})
    r = client.put(
        "/api/v1/training/datasets/manifest-ds/manifest",
        json={"entries": [{"caption": "", "video": "clip.mp4"}]},
    )
    assert r.status_code == 400


def test_put_manifest_whitespace_only_caption_returns_400(monkeypatch, tmp_path):
    """PUT manifest with whitespace-only caption must return HTTP 400."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "ws-ds"})
    r = client.put(
        "/api/v1/training/datasets/ws-ds/manifest",
        json={"entries": [{"caption": "   ", "video": "clip.mp4"}]},
    )
    assert r.status_code == 400


def test_put_manifest_missing_file_returns_200_with_violation(monkeypatch, tmp_path):
    """PUT manifest with valid caption but missing video returns 200 with violations."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "viol-ds"})
    r = client.put(
        "/api/v1/training/datasets/viol-ds/manifest",
        json={"entries": [{"caption": "a cat", "video": "missing.mp4"}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "violations" in data
    assert "missing.mp4" in data["violations"]


def test_put_manifest_few_clips_warns(monkeypatch, tmp_path):
    """PUT manifest with fewer than MIN_CLIPS returns adequacy warning."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "warn-ds"})
    r = client.put(
        "/api/v1/training/datasets/warn-ds/manifest",
        json={"entries": [{"caption": "a cat", "video": "missing.mp4"}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["warnings"]) > 0


def test_put_manifest_writes_file(monkeypatch, tmp_path):
    """PUT manifest with valid entries writes manifest.json when no hard violations."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "write-ds"})
    # manifest.json is written regardless of per-clip media violations; only an
    # empty/whitespace caption blocks the write (-> HTTP 400). Here the caption is
    # valid, so the manifest must be persisted even though the dummy clip will fail
    # media validation (its violations are reported in the response body).
    clip = dataset_store.clips_dir("write-ds") / "vid.mp4"
    clip.write_bytes(b"\x00")  # file exists; validate_clip will probe it
    r = client.put(
        "/api/v1/training/datasets/write-ds/manifest",
        json={"entries": [{"caption": "a cat walks", "video": str(clip)}]},
    )
    assert r.status_code == 200
    # Manifest file written regardless of media violations
    manifest_path = dataset_store.dataset_dir("write-ds") / "manifest.json"
    assert manifest_path.exists()


# ---------------------------------------------------------------------------
# DELETE /api/v1/training/datasets/{id}
# ---------------------------------------------------------------------------


def test_delete_dataset_returns_200(monkeypatch, tmp_path):
    """Deleting an existing dataset returns 200 with deleted=True."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "del-ds"})
    r = client.delete("/api/v1/training/datasets/del-ds")
    assert r.status_code == 200
    assert r.json()["deleted"] is True


def test_delete_nonexistent_dataset_returns_200_with_false(monkeypatch, tmp_path):
    """Deleting a dataset that does not exist returns 200 with deleted=False."""
    _patch_training_dir(monkeypatch, tmp_path)
    r = client.delete("/api/v1/training/datasets/ghost-ds")
    assert r.status_code == 200
    assert r.json()["deleted"] is False


def test_delete_removes_from_list(monkeypatch, tmp_path):
    """After deletion, the dataset no longer appears in GET list."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "gone-ds"})
    client.delete("/api/v1/training/datasets/gone-ds")
    r = client.get("/api/v1/training/datasets")
    ids = [d["id"] for d in r.json()]
    assert "gone-ds" not in ids


# ---------------------------------------------------------------------------
# Security: path traversal
# ---------------------------------------------------------------------------


def test_clip_upload_rejects_traversal_filename(monkeypatch, tmp_path):
    """Uploading a clip with a traversal filename must return HTTP 400."""
    _patch_training_dir(monkeypatch, tmp_path)
    client.post("/api/v1/training/datasets", json={"dataset_id": "trav-ds"})
    r = client.post(
        "/api/v1/training/datasets/trav-ds/clips",
        files={"file": ("../evil.mp4", io.BytesIO(b"\x00"), "video/mp4")},
    )
    assert r.status_code == 400
    # The escape target must NOT have been written.
    assert not (tmp_path / "evil.mp4").exists()


def test_dataset_id_traversal_rejected(monkeypatch, tmp_path):
    """A traversal dataset_id must be rejected with HTTP 400 on create and delete."""
    _patch_training_dir(monkeypatch, tmp_path)
    r_create = client.post(
        "/api/v1/training/datasets", json={"dataset_id": "../escape"}
    )
    assert r_create.status_code == 400

    # A bare ".." in the path is normalised away by the URL layer before routing,
    # so send it URL-encoded so it reaches (and is rejected by) the handler.
    r_delete = client.delete("/api/v1/training/datasets/%2e%2e")
    assert r_delete.status_code == 400
