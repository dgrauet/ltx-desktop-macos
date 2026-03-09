# Backend — CLAUDE.md

This is the Python FastAPI backend for LTX Desktop macOS. It handles all MLX inference, model management, API endpoints, and export.

## Two Agent Domains

This directory is worked on by **two agents** with clear boundaries:

- **Agent 1 (Engine)**: `engine/`, `audio/` — MLX pipelines, memory, models, performance
- **Agent 2 (API)**: `api/`, `export/`, `utils/`, `main.py` — FastAPI, endpoints, queue, export

Do NOT cross boundaries. If you need something from the other domain, define an interface (function signature + docstring) and let the other agent implement it.

---

## Tech Stack

- Python 3.12+
- Package manager: `uv`
- API framework: FastAPI + uvicorn
- ML framework: MLX (mlx, mlx-video-with-audio, mlx-lm, mlx-audio)
- Video encoding: ffmpeg (external binary)
- Linter/formatter: ruff

## Conventions

- Mandatory type hints on all functions
- Async for all FastAPI route handlers
- Google-style docstrings
- All engine functions must return memory stats alongside their main output
- Every pipeline stage boundary MUST call `aggressive_cleanup()`

## Critical Rules

### Memory Management (NON-NEGOTIABLE)

```python
import mlx.core as mx
import gc

def aggressive_cleanup():
    gc.collect()
    mx.clear_cache()          # Note: mx.metal.clear_cache() is deprecated since MLX 0.31
    mx.eval(mx.zeros(1))      # Barrier — mx.eval here is mlx.core.eval (tensor materialization)
```

**Call this**:
1. After prompt encoding
2. After Stage 1 diffusion
3. After Stage 2 upscale
4. After VAE decode
5. After audio decode
6. After every completed job
7. After any model unload

### VAE Decode (NON-NEGOTIABLE)

Never decode all frames in memory. Stream frame-by-frame to ffmpeg pipe:

```python
for i in range(num_frames):
    frame = vae.decode_frame(latents[i])
    ffmpeg_proc.stdin.write(frame_to_bytes(frame))
    del frame
    if i % 8 == 0:
        aggressive_cleanup()
```

### Model Loading (NON-NEGOTIABLE)

- Prompt enhancer (Qwen3.5-2B) and video model (LTX-2.3) must NEVER coexist on < 64GB
- Load → use → unload → cleanup → load next
- Periodic full model reload every 5 generations

### Performance Optimizations (implement from day 1)

1. `mx.compile(model.forward)` after loading
2. Kernel warm-up pass (9 frames, 1 step, 256×256) after loading
3. LatentPool pre-allocation for max expected resolution
4. TeaCache (block-output caching, `rel_l1_thresh=0.03`)

## Starting Point

Start with a SINGLE `main.py` file. Extract into modules only as it grows. The initial `main.py` should have:

```python
# main.py — MVP
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="LTX Desktop Backend")

@app.get("/api/v1/system/health")
async def health():
    return {"status": "ok"}

@app.get("/api/v1/system/info")
async def system_info():
    # Detect chip, RAM, etc.
    ...

@app.get("/api/v1/system/memory")
async def memory_stats():
    # mx.get_active_memory() etc. (mx.metal.* deprecated since MLX 0.31)
    ...

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

## Directory Structure

```
backend/
├── pyproject.toml
├── main.py                    # FastAPI entry point (Agent 2)
├── engine/                    # Agent 1 territory
│   ├── __init__.py
│   ├── mlx_inference.py       # Main inference coordinator
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── text_to_video.py
│   │   ├── image_to_video.py
│   │   ├── retake.py
│   │   ├── extend.py
│   │   └── upscaler.py
│   ├── model_manager.py       # Load/unload/download models
│   ├── prompt_enhancer.py     # Qwen3.5-2B via mlx-lm
│   ├── memory_manager.py      # ★ aggressive_cleanup, reload, monitoring
│   ├── lora_manager.py        # LoRA loading and application
│   └── teacache.py            # TeaCache MLX port
├── audio/                     # Agent 1 territory
│   ├── __init__.py
│   ├── tts_engine.py
│   └── audio_mixer.py
├── api/                       # Agent 2 territory
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── generation.py
│   │   ├── models.py
│   │   ├── queue.py
│   │   ├── export.py
│   │   ├── audio.py
│   │   └── health.py
│   └── websockets/
│       ├── __init__.py
│       └── progress.py
├── export/                    # Agent 2 territory
│   ├── __init__.py
│   ├── video_encoder.py
│   └── fcpxml_export.py
└── utils/                     # Agent 2 territory
    ├── __init__.py
    ├── config.py
    └── system_info.py
```

## Dependencies

See `pyproject.toml` in the root CLAUDE.md. Key packages:
- `mlx>=0.31.0`, `mlx-video-with-audio>=0.1.3`, `mlx-lm>=0.31.0`
- `fastapi>=0.115.0`, `uvicorn>=0.32.0`, `websockets>=13.0`
- `mlx-audio>=0.3.0` (optional, for TTS)
