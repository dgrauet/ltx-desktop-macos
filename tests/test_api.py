"""API Integration Tests — Sprint 1."""

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


def test_queue_list(client: httpx.Client):
    """GET /queue returns a list of jobs."""
    r = client.get("/api/v1/queue")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
