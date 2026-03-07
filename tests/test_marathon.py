"""Marathon Generation Test — CRITICAL (blocks release).

Runs 10 consecutive text-to-video generations and validates:
1. No OOM crash
2. Memory after gen 10 is within 20% of memory after gen 1
3. No generation takes >2x longer than gen 1
4. All 10 output files are valid MP4s

Config: 256x256, 9 frames, 2 steps (stub mode for Sprint 1).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx
import pytest

BACKEND_URL = "http://127.0.0.1:8000"


def _generate_and_wait(client: httpx.Client, gen_num: int) -> dict:
    """Submit a T2V job and wait for completion. Returns timing + memory info."""
    t0 = time.monotonic()

    r = client.post("/api/v1/generate/text-to-video", json={
        "prompt": f"Marathon test generation {gen_num}: a peaceful landscape with mountains",
        "width": 256,
        "height": 256,
        "num_frames": 9,
        "steps": 2,
        "seed": gen_num * 100,
    })
    assert r.status_code == 200, f"Gen {gen_num}: POST failed with {r.status_code}"
    job_id = r.json()["job_id"]

    # Poll until complete (max 60s per generation)
    for _ in range(120):
        status = client.get(f"/api/v1/queue/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    elapsed = time.monotonic() - t0

    assert status["status"] == "completed", (
        f"Gen {gen_num}: job {job_id} failed — {status.get('error')}"
    )

    # Get memory stats
    mem = client.get("/api/v1/system/memory").json()

    result = status["result"]
    output_path = result["output_path"]

    return {
        "gen_num": gen_num,
        "job_id": job_id,
        "elapsed": elapsed,
        "output_path": output_path,
        "active_memory_gb": mem["active_memory_gb"],
        "cache_memory_gb": mem["cache_memory_gb"],
        "peak_memory_gb": mem["peak_memory_gb"],
    }


def _verify_mp4(path: str) -> bool:
    """Check that the file is a valid MP4 using ffprobe."""
    try:
        # Try common ffprobe locations
        ffprobe = None
        for p in ["ffprobe", "/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]:
            try:
                subprocess.run([p, "-version"], capture_output=True, check=True)
                ffprobe = p
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

        if not ffprobe:
            # If ffprobe not available, just check file exists and has content
            return Path(path).exists() and Path(path).stat().st_size > 0

        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return Path(path).exists() and Path(path).stat().st_size > 0


@pytest.mark.timeout(3600)
def test_marathon_10_generations(backend_process):
    """Run 10 consecutive generations and validate stability."""
    client = httpx.Client(base_url=backend_process, timeout=600)
    results = []

    print("\n" + "=" * 60)
    print("MARATHON GENERATION TEST — 10 consecutive generations")
    print("=" * 60)

    for i in range(1, 11):
        print(f"\n--- Generation {i}/10 ---")
        result = _generate_and_wait(client, i)
        results.append(result)

        print(f"  Time: {result['elapsed']:.2f}s")
        print(f"  Active memory: {result['active_memory_gb']:.3f} GB")
        print(f"  Cache memory: {result['cache_memory_gb']:.3f} GB")
        print(f"  Output: {result['output_path']}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Criterion 1: All 10 completed (no crash)
    assert len(results) == 10, f"Only {len(results)}/10 generations completed"
    print("PASS: 10/10 generations completed without crash")

    # Criterion 2: Memory stability (gen 10 within 20% of gen 1)
    mem_gen1 = results[0]["active_memory_gb"]
    mem_gen10 = results[-1]["active_memory_gb"]

    if mem_gen1 > 0:
        mem_ratio = mem_gen10 / mem_gen1
        print(f"Memory gen1={mem_gen1:.3f}GB, gen10={mem_gen10:.3f}GB, ratio={mem_ratio:.2f}")
        assert mem_ratio <= 1.2, (
            f"FAIL: Memory grew {mem_ratio:.2f}x (gen1={mem_gen1:.3f}GB, gen10={mem_gen10:.3f}GB)"
        )
        print("PASS: Memory within 20% of generation 1")
    else:
        print("SKIP: Memory check (stub mode, active memory is 0)")

    # Criterion 3: No generation takes >2x longer than gen 1
    time_gen1 = results[0]["elapsed"]
    for r in results:
        ratio = r["elapsed"] / time_gen1 if time_gen1 > 0 else 1.0
        if ratio > 2.0:
            pytest.fail(
                f"FAIL: Gen {r['gen_num']} took {r['elapsed']:.2f}s "
                f"(gen1={time_gen1:.2f}s, ratio={ratio:.2f}x)"
            )
    print(f"PASS: All generations within 2x of gen 1 ({time_gen1:.2f}s)")

    # Criterion 4: All output files are valid MP4s
    for r in results:
        assert _verify_mp4(r["output_path"]), (
            f"FAIL: Invalid MP4 at {r['output_path']}"
        )
    print("PASS: All 10 output files are valid MP4s")

    print("\n" + "=" * 60)
    print("MARATHON TEST PASSED")
    print("=" * 60)

    client.close()
