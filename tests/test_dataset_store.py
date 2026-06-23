"""P0: dataset_store validates media constraints + caption adequacy."""
import subprocess
from pathlib import Path

import pytest

from dataset_store import build_manifest, validate_clip, MIN_CLIPS
from engine.ffmpeg_utils import find_ffmpeg


def _make_clip(path: Path, frames: int, w: int, h: int):
    # frames at 25fps; testsrc gives exact frame count via -frames:v
    subprocess.run(
        [find_ffmpeg(), "-y", "-f", "lavfi", "-i",
         f"testsrc=size={w}x{h}:rate=25", "-frames:v", str(frames),
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def test_valid_clip_has_no_violations(tmp_path):
    clip = tmp_path / "ok.mp4"
    _make_clip(clip, frames=25, w=704, h=480)  # 25 % 8 == 1, dims % 32 == 0
    assert validate_clip(str(clip)) == []


def test_bad_frame_count_flagged(tmp_path):
    clip = tmp_path / "bad.mp4"
    _make_clip(clip, frames=24, w=704, h=480)  # 24 % 8 != 1
    violations = validate_clip(str(clip))
    assert any("frame" in v.lower() for v in violations)


def test_empty_caption_rejected():
    with pytest.raises(ValueError):
        build_manifest([{"caption": "", "video": "a.mp4"}])


def test_few_clips_warns_not_raises():
    m = build_manifest([{"caption": "a cat", "video": "a.mp4"}])
    assert m[0]["caption"] == "a cat"
    # adequacy warning surfaced separately, not an exception
    assert MIN_CLIPS == 5


def test_missing_clip_flagged():
    violations = validate_clip("/tmp/does_not_exist_xyz.mp4")
    assert violations, "expected at least one violation for missing file"
    assert any("not found" in v for v in violations), f"unexpected message: {violations}"


def test_zero_frame_message_wording(tmp_path):
    """Zero-frame branch produces the corrupt-file message, not a frame-count mismatch."""
    from unittest.mock import patch

    real_clip = tmp_path / "real.mp4"
    _make_clip(real_clip, frames=25, w=704, h=480)

    with patch("dataset_store.probe_frame_count", return_value=0):
        violations = validate_clip(str(real_clip))

    assert violations, "expected violation for 0-frame probe result"
    assert any("corrupt" in v or "could not read" in v for v in violations), (
        f"expected corrupt/unreadable message, got: {violations}"
    )


# ---------------------------------------------------------------------------
# Task 3: dataset directory management + materialize_captions
# ---------------------------------------------------------------------------

import dataset_store as ds  # noqa: E402 — import after module-level fixtures


def test_create_dataset_makes_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    result = ds.create_dataset("myds")
    assert result == ds.dataset_dir("myds")
    assert ds.clips_dir("myds").is_dir()
    assert ds.captions_dir("myds").is_dir()


def test_materialize_captions_writes_per_clip_txt(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    ds.create_dataset("d1")
    (ds.clips_dir("d1") / "a.mp4").write_bytes(b"x")
    ds.write_manifest(str(ds.dataset_dir("d1")), [{"caption": "a cat", "video": "a.mp4"}])
    ds.materialize_captions("d1")
    assert (ds.captions_dir("d1") / "a.txt").read_text() == "a cat"


def test_materialize_captions_multiple_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    ds.create_dataset("d2")
    entries = [
        {"caption": "a dog", "video": "clips/b.mp4"},
        {"caption": "a fish", "video": "sub/c.mp4"},
    ]
    ds.write_manifest(str(ds.dataset_dir("d2")), entries)
    ds.materialize_captions("d2")
    assert (ds.captions_dir("d2") / "b.txt").read_text() == "a dog"
    assert (ds.captions_dir("d2") / "c.txt").read_text() == "a fish"


def test_list_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    ds.create_dataset("ls1")
    ds.create_dataset("ls2")
    (ds.clips_dir("ls1") / "v.mp4").write_bytes(b"video")
    datasets = ds.list_datasets()
    ids = {d["id"] for d in datasets}
    assert {"ls1", "ls2"}.issubset(ids)
    ls1 = next(d for d in datasets if d["id"] == "ls1")
    assert ls1["clip_count"] == 1
    assert ls1["disk_bytes"] > 0
    assert ls1["has_precomputed"] is False


def test_list_datasets_has_precomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    ds.create_dataset("pre1")
    ds.precomputed_dir("pre1").mkdir(parents=True, exist_ok=True)
    datasets = ds.list_datasets()
    pre1 = next(d for d in datasets if d["id"] == "pre1")
    assert pre1["has_precomputed"] is True


def test_delete_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    ds.create_dataset("del1")
    assert ds.dataset_dir("del1").exists()
    result = ds.delete_dataset("del1")
    assert result is True
    assert not ds.dataset_dir("del1").exists()


def test_delete_dataset_nonexistent(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "TRAINING_DIR", tmp_path)
    result = ds.delete_dataset("ghost")
    assert result is False
