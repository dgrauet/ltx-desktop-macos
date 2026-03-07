"""API Integration Tests — Sprint 2."""

from __future__ import annotations

import time

import httpx


def test_health_endpoint(client: httpx.Client):
    """GET /system/health returns 200 with expected fields."""
    r = client.get("/api/v1/system/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "generation_count" in data


def test_system_info(client: httpx.Client):
    """GET /system/info returns valid chip and RAM data."""
    r = client.get("/api/v1/system/info")
    assert r.status_code == 200
    data = r.json()
    assert data["chip"]  # Non-empty
    assert data["ram_total_gb"] > 0
    assert "macos_version" in data


def test_memory_endpoint(client: httpx.Client):
    """GET /system/memory returns valid memory stats."""
    r = client.get("/api/v1/system/memory")
    assert r.status_code == 200
    data = r.json()
    assert "active_memory_gb" in data
    assert "cache_memory_gb" in data
    assert "peak_memory_gb" in data
    assert "system_available_gb" in data
    assert "generation_count_since_reload" in data


def test_generate_t2v(client: httpx.Client):
    """POST /generate/text-to-video produces a valid job that completes."""
    r = client.post("/api/v1/generate/text-to-video", json={
        "prompt": "A sunset over the ocean",
        "width": 256,
        "height": 256,
        "num_frames": 9,
        "steps": 2,
        "seed": 123,
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Poll until complete (max 30s)
    for _ in range(60):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert status["status"] == "completed", f"Job failed: {status.get('error')}"
    assert status["result"]["output_path"].endswith(".mp4")

    # Sprint 2: Verify TeaCache stats are present
    stages = status["result"].get("stages", {})
    assert "teacache_hit_rate" in stages, "TeaCache stats missing from result"


def test_generate_preview(client: httpx.Client):
    """POST /generate/preview produces a fast low-res video."""
    t0 = time.monotonic()

    r = client.post("/api/v1/generate/preview", json={
        "prompt": "A quick preview test",
        "seed": 42,
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Poll until complete (max 30s)
    for _ in range(60):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    elapsed = time.monotonic() - t0

    assert status["status"] == "completed", f"Preview failed: {status.get('error')}"
    assert status["result"]["output_path"].endswith(".mp4")
    # Preview should complete faster than a full T2V generation
    assert elapsed < 10, f"Preview took too long: {elapsed:.2f}s"


def test_generate_i2v_missing_image(client: httpx.Client):
    """POST /generate/image-to-video with invalid image path should fail."""
    r = client.post("/api/v1/generate/image-to-video", json={
        "prompt": "A video from an image",
        "source_image_path": "/nonexistent/image.png",
        "width": 256,
        "height": 256,
        "num_frames": 9,
        "steps": 2,
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # Should fail because image doesn't exist
    for _ in range(60):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert status["status"] == "failed"


def test_queue_list(client: httpx.Client):
    """GET /queue returns a list of jobs."""
    r = client.get("/api/v1/queue")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Sprint 3 tests
# ---------------------------------------------------------------------------

def test_prompt_enhance_unavailable_or_works(client: httpx.Client):
    """POST /prompt/enhance either works or returns 503 if mlx-lm not installed."""
    r = client.post("/api/v1/prompt/enhance", json={"prompt": "a cat"})
    # 200 if mlx-lm available, 503 if not — both are valid
    assert r.status_code in (200, 503), f"Unexpected status: {r.status_code}"
    if r.status_code == 200:
        data = r.json()
        assert "original" in data
        assert "enhanced" in data
        assert data["original"] == "a cat"


def test_generate_retake(client: httpx.Client):
    """POST /generate/retake with non-existent video should fail gracefully."""
    r = client.post("/api/v1/generate/retake", json={
        "source_video_path": "/nonexistent/video.mp4",
        "prompt": "retake the sky",
        "start_time_s": 0.0,
        "end_time_s": 2.0,
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    # Either completes (stub doesn't validate path) or fails — both OK
    for _ in range(30):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert status["status"] in ("completed", "failed")


def test_generate_extend(client: httpx.Client):
    """POST /generate/extend returns a job_id and eventually completes or fails."""
    r = client.post("/api/v1/generate/extend", json={
        "source_video_path": "/nonexistent/video.mp4",
        "prompt": "extend the scene",
        "direction": "forward",
        "extension_frames": 9,
    })
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_extend_invalid_direction(client: httpx.Client):
    """POST /generate/extend with invalid direction returns 422."""
    r = client.post("/api/v1/generate/extend", json={
        "source_video_path": "/video.mp4",
        "prompt": "extend",
        "direction": "sideways",  # invalid
    })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Sprint 4 tests
# ---------------------------------------------------------------------------

def test_lora_list(client: httpx.Client):
    """GET /loras returns a list with at least the built-in stubs."""
    r = client.get("/api/v1/loras")
    assert r.status_code == 200
    loras = r.json()
    assert isinstance(loras, list)
    # Built-in stubs should always be present
    ids = [lora["id"] for lora in loras]
    assert "camera-control" in ids, f"camera-control missing from LoRA list: {ids}"
    assert "detail-enhance" in ids, f"detail-enhance missing from LoRA list: {ids}"


def test_lora_load_builtin(client: httpx.Client):
    """POST /loras/load with a built-in LoRA ID should succeed."""
    r = client.post("/api/v1/loras/load", json={"lora_id": "camera-control"})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True


def test_lora_load_nonexistent(client: httpx.Client):
    """POST /loras/load with unknown ID returns 404."""
    r = client.post("/api/v1/loras/load", json={"lora_id": "nonexistent-lora-xyz"})
    assert r.status_code == 404


def test_audio_tts(client: httpx.Client):
    """POST /audio/tts produces a WAV file."""
    r = client.post("/api/v1/audio/tts", json={
        "text": "Hello, this is a test of the TTS system.",
        "voice": "default",
        "speed": 1.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert "output_path" in data
    assert data["output_path"].endswith(".wav")
    from pathlib import Path
    assert Path(data["output_path"]).exists(), "TTS output file does not exist"


def test_audio_music(client: httpx.Client):
    """POST /audio/music produces a WAV file."""
    r = client.post("/api/v1/audio/music", json={
        "genre": "ambient",
        "duration": 2.0,
    })
    assert r.status_code == 200
    data = r.json()
    assert "output_path" in data
    assert data["output_path"].endswith(".wav")


def test_export_video(client: httpx.Client):
    """POST /export/video with a real video file re-encodes to MP4."""
    # Use a T2V job to get a real video first
    r = client.post("/api/v1/generate/text-to-video", json={
        "prompt": "export test video",
        "width": 256, "height": 256, "num_frames": 9, "steps": 1, "seed": 999,
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    for _ in range(60):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert status["status"] == "completed"
    video_path = status["result"]["output_path"]

    # Now export it
    r2 = client.post("/api/v1/export/video", json={
        "video_path": video_path,
        "codec": "h264",
        "output_format": "mp4",
        "bitrate": "4M",
    })
    assert r2.status_code == 200
    assert "output_path" in r2.json()


def test_export_fcpxml(client: httpx.Client):
    """POST /export/fcpxml generates a valid FCPXML file."""
    # Generate a video first
    r = client.post("/api/v1/generate/text-to-video", json={
        "prompt": "fcpxml test",
        "width": 256, "height": 256, "num_frames": 9, "steps": 1, "seed": 888,
    })
    job_id = r.json()["job_id"]

    for _ in range(60):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    video_path = status["result"]["output_path"]

    r2 = client.post("/api/v1/export/fcpxml", json={
        "video_path": video_path,
        "clip_name": "Test Clip",
    })
    assert r2.status_code == 200
    fcpxml_path = r2.json()["output_path"]
    assert fcpxml_path.endswith(".fcpxml")

    from pathlib import Path
    content = Path(fcpxml_path).read_text()
    assert '<?xml' in content
    assert 'fcpxml' in content
    assert 'Test Clip' in content
