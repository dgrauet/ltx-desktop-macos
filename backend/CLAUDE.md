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
- ML framework: MLX (mlx, mlx-lm); LTX-2.3 model vendored in `engine/ltx23_model/`
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
├── main.py                    # FastAPI entry point
├── engine/
│   ├── generate_v23.py        # LTX-2.3 generation subprocess
│   ├── encode_text_subprocess.py  # Text encoding subprocess (Gemma 3)
│   ├── mlx_runner.py          # Subprocess orchestrator
│   ├── memory_manager.py      # ★ aggressive_cleanup, reload, monitoring
│   ├── model_manager.py       # Load/unload/download models
│   ├── prompt_enhancer.py     # Qwen3.5-2B via mlx-lm
│   ├── lora_manager.py        # LoRA loading and application
│   ├── teacache.py            # TeaCache MLX port
│   ├── ltx23_model/           # Vendored LTX-2.3 architecture
│   │   ├── transformer.py     # DiT blocks (BasicAVTransformerBlock)
│   │   ├── attention.py       # Multi-head attention + RoPE
│   │   ├── feed_forward.py    # Feed-forward network
│   │   ├── model.py           # X0Model, LTXModel wrappers
│   │   ├── loader.py          # Quantized model loading from split files
│   │   ├── pipeline.py        # Diffusion generation loop
│   │   ├── vae_decoder.py     # Video VAE decoder (streaming to ffmpeg)
│   │   ├── vae_encoder.py     # Video VAE encoder (for I2V)
│   │   ├── audio_decoder.py   # Audio VAE decoder (latent → mel)
│   │   ├── vocoder.py         # HiFi-GAN/BigVGAN v2 vocoder + BWE
│   │   ├── text_encoder.py    # Gemma 3 encoder with dual projections
│   │   ├── connector.py       # Embeddings connector
│   │   ├── rope.py            # Rotary position embeddings
│   │   ├── timestep_embedding.py  # AdaLayerNorm
│   │   └── patchifier.py      # Video patchification
│   └── pipelines/
│       ├── text_to_video.py
│       ├── image_to_video.py
│       ├── preview.py
│       ├── retake.py
│       └── extend.py
├── audio/
│   ├── tts_engine.py          # Local TTS (MLX-Audio interface)
│   └── audio_mixer.py         # ffmpeg-based multi-track mixer
├── export/
│   ├── video_encoder.py
│   └── fcpxml_export.py
└── utils/
    ├── config.py
    └── system_info.py
```

## Dependencies

See `pyproject.toml`. Key packages:
- `mlx>=0.31.0`, `mlx-lm>=0.31.0`
- `fastapi>=0.115.0`, `uvicorn>=0.32.0`, `websockets>=13.0`
- `safetensors>=0.4.0`, `transformers>=4.51.0`, `huggingface-hub>=0.26.0`
- `soundfile>=0.12.0` (for audio WAV output)
- `mlx-audio>=0.4.1` (TTS via Kokoro — mlx-lm pin conflict resolved in 0.4.1+)
