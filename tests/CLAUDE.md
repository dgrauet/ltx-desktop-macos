# Tests — CLAUDE.md

Testing suite for LTX Desktop macOS. **Agent 4 (QA & DevOps)** owns this directory.

---

## Philosophy

The #1 stability risk is **Metal memory fragmentation across repeated generations**. Every test strategy is oriented around catching this. Unit tests are secondary to the Marathon Generation test.

## Test Hierarchy (by priority)

### 1. 🔴 test_marathon.py — CRITICAL (blocks release)

The single most important test. Must pass before any release.

```python
"""
Marathon Generation Test

Runs 10 consecutive text-to-video generations and validates:
1. No OOM crash
2. Memory after gen 10 is within 20% of memory after gen 1
3. No generation takes >2× longer than gen 1
4. All 10 output files are valid MP4s with correct duration

Config:
- Resolution: 768×512
- Frames: 97
- Model: distilled
- Steps: 8

This test takes ~15-30 minutes depending on hardware.
"""
```

**Pass criteria**:
```
✓ 10/10 generations completed without crash
✓ memory_gen10 <= memory_gen1 * 1.2
✓ time_gen10 <= time_gen1 * 2.0
✓ all output files are valid (ffprobe check)
```

**Run**: `pytest tests/test_marathon.py -v --timeout=3600`

### 2. 🟡 test_memory.py — High priority

```python
"""
Memory Management Tests

- test_aggressive_cleanup_reduces_memory: verify mx.metal cache drops after cleanup
- test_model_load_unload_cycle: load model → check memory → unload → check memory returned
- test_prompt_enhancer_isolation: load Qwen → unload → verify memory freed before LTX load
- test_vae_streaming_peak: verify streaming VAE decode peak < batch VAE decode peak
- test_periodic_reload: run 5 gens → reload → verify memory resets
"""
```

### 3. 🟡 test_inference.py — High priority

```python
"""
Pipeline Tests (require model downloaded)

- test_t2v_basic: 9 frames, 256×256, 1 step → produces valid output
- test_t2v_with_audio: verify audio track present in output
- test_i2v_basic: reference image → video with first frame matching
- test_preview_fast: 384×256, 4 steps → completes in < 30s
- test_retake_segment: generate → retake middle → verify unchanged frames
- test_extend_forward: generate → extend → verify continuous motion
- test_teacache_speedup: same gen with/without TeaCache → verify speedup > 1.3×
- test_teacache_quality: same gen with/without TeaCache → verify PSNR > 25dB
"""
```

### 4. 🟢 test_api.py — Medium priority

```python
"""
API Integration Tests (require running backend)

- test_health_endpoint: GET /system/health → 200
- test_system_info: GET /system/info → valid chip and RAM data
- test_memory_endpoint: GET /system/memory → valid memory stats
- test_generate_t2v: POST /generate/text-to-video → job_id → poll until done
- test_generate_preview: POST /generate/preview → completes faster than full gen
- test_queue_cancel: start gen → cancel → verify stopped
- test_websocket_progress: connect to /ws/progress → receive updates
- test_prompt_enhance: POST /prompt/enhance → enhanced prompt longer than input
- test_export_mp4: POST /export/video → valid MP4 file
- test_export_fcpxml: POST /export/fcpxml → valid XML
- test_model_list: GET /models → at least one model
- test_lora_list: GET /loras → list (possibly empty)
"""
```

### 5. 🟢 test_prompt_enhancer.py — Medium priority

```python
"""
Prompt Enhancement Tests

- test_enhance_short_prompt: "cat" → detailed paragraph > 50 words
- test_enhance_preserves_subject: "red car" → output contains "car"
- test_enhance_lazy_load: verify Qwen3.5-2B loads and unloads cleanly
- test_enhance_memory_freed: after unload, memory returns to pre-load level
"""
```

## Running Tests

```bash
# Full suite (requires models downloaded + backend running)
pytest tests/ -v --timeout=3600

# Quick smoke test (no models needed)
pytest tests/test_api.py::test_health_endpoint -v

# Memory-focused
pytest tests/test_memory.py tests/test_marathon.py -v --timeout=3600

# Pipeline only
pytest tests/test_inference.py -v --timeout=600
```

## Fixtures

```python
# conftest.py

import pytest
import httpx

BACKEND_URL = "http://127.0.0.1:8000"

@pytest.fixture(scope="session")
def backend():
    """Verify backend is running."""
    try:
        r = httpx.get(f"{BACKEND_URL}/api/v1/system/health", timeout=5)
        assert r.status_code == 200
    except Exception:
        pytest.skip("Backend not running. Start with: uvicorn main:app --port 8000")
    return BACKEND_URL

@pytest.fixture(scope="session")
def client(backend):
    """HTTP client for API tests."""
    return httpx.Client(base_url=backend, timeout=600)
```

## Test Data

- Store test images in `tests/fixtures/` (1-2 small JPEGs for I2V tests)
- Store expected outputs metadata in `tests/fixtures/expected/` (JSON with frame count, duration)
- No large model files in tests — tests skip if models not downloaded
