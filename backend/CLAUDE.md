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
- ML framework: MLX (mlx, ltx-core-mlx 0.15.x, ltx-pipelines-mlx 0.15.x — from the dgrauet/ltx-2-mlx monorepo)
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
    mx.synchronize()          # Barrier — wait for all queued GPU work
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

- Library handles staged loading via `low_memory=True` (Gemma → free → transformer+VAE)
- Periodic full model reload every 5 generations

### Performance Notes

- `mx.compile()` and kernel warm-up are not implemented (incompatible with subprocess-per-generation architecture)
- Library (`ltx-pipelines-mlx`) handles inference optimization internally

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
├── main.py                    # FastAPI entry point
├── engine/
│   ├── generate_v23.py        # LTX-2.3 generation subprocess (uses ltx-pipelines-mlx)
│   ├── mlx_runner.py          # Single-subprocess orchestrator
│   ├── memory_manager.py      # ★ aggressive_cleanup, reload, monitoring
│   ├── model_manager.py       # Load/unload/download models
│   ├── lora_manager.py        # LoRA loading and application
│   └── pipelines/
│       ├── text_to_video.py
│       ├── image_to_video.py
│       ├── retake.py
│       └── extend.py
├── audio/                         # Reserved for future TTS/mixing
├── export/
│   ├── video_encoder.py
│   └── fcpxml_export.py
└── utils/
    ├── config.py
    └── system_info.py
```

## Dependencies

See `pyproject.toml`. Key packages:
- `mlx>=0.32.2`, `ltx-core-mlx` / `ltx-pipelines-mlx` / `ltx-trainer-mlx` 0.15.10 (git: dgrauet/ltx-2-mlx, pinned to tag `v0.15.10` — bump the tag explicitly)
- lib 0.14+ API: `frame_rate=` keyword obligatoire sur tous les `generate*`; classes `DistilledPipeline` / `TI2VidOneStagePipeline` / `TI2VidTwoStagesPipeline` / `TI2VidTwoStagesHQPipeline` / `RetakePipeline` (extend inclus); I2V via `image=` sur tous les pipelines
- `fastapi>=0.115.0`, `uvicorn>=0.32.0`, `websockets>=13.0`
- `safetensors>=0.4.0`, `transformers>=4.51.0`, `huggingface-hub>=0.26.0`
- `soundfile>=0.12.0` (for audio WAV output)
