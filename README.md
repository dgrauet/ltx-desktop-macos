# LTX Desktop macOS

Native macOS application for local AI video generation using **LTX-2.3** on Apple Silicon via **MLX**. No cloud required — everything runs on-device.

## What It Is

A local alternative to [Lightricks LTX Desktop](https://github.com/Lightricks/ltx-desktop) that replaces NVIDIA/CUDA inference with **MLX on Apple Silicon unified memory**. The official macOS version requires a cloud API — this project eliminates that dependency.

**Stack**: SwiftUI (native macOS app) + Python FastAPI backend + MLX inference engine (LTX-2.3, 22B params).

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | 14.0 Sonoma | 15 Sequoia or later |
| RAM | 32 GB | 64 GB+ |
| Chip | Apple M1 Pro | M3 Max / M4 Ultra |
| Disk | 60 GB free | 100 GB+ free |
| Xcode | 15.0+ | Latest |
| Python | 3.12+ | 3.12 |

> **16 GB machines**: very limited — low resolutions only.

---

## Quick Start

### 1. Install system dependencies

```bash
brew install ffmpeg uv python@3.12
```

### 2. Clone and setup

```bash
git clone <repo-url>
cd ltx-desktop-macos
bash scripts/setup.sh
```

### 3. Download models

```bash
bash scripts/download_models.sh
```

This downloads:
- `dgrauet/ltx-2.3-mlx-distilled-q8` (~28 GB) — pre-converted MLX video generation model
- `mlx-community/gemma-3-12b-it-4bit` (~6 GB) — text encoder
- `mlx-community/Qwen3.5-2B-4bit` (~1.2 GB) — prompt enhancement

### 4. Start the backend

```bash
cd backend
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Open the app

```bash
open app/LTXDesktop.xcodeproj
```

Run from Xcode (Cmd+R). The SwiftUI app auto-connects to the backend on `localhost:8000`.

---

## Architecture

```
SwiftUI App (native macOS)
    |
    | HTTP + WebSocket (localhost:8000)
    |
Python FastAPI Backend
    |
    +-- MLX Engine (DiT inference on Apple Silicon)
    +-- ffmpeg (video encoding / audio mixing)
```

Two-subprocess model for 32GB machines:
- **Subprocess A**: Gemma 3 12B 4-bit text encoding (~8.7GB peak) → exits, frees GPU
- **Subprocess B**: Transformer + VAE generation (~12.6GB peak)

The backend runs as a separate process managed by the SwiftUI app's `ProcessManager`. Crash isolation means an OOM in the ML pipeline kills only the backend — the UI stays alive and offers a restart button.

---

## Features

### Working

- **Text-to-Video (T2V)**: prompt → video with synchronized audio
- **Image-to-Video (I2V)**: drag-and-drop source image conditions the first frame
- **Synchronized audio**: audio generated in single pass (vocoder output, quality WIP)
- **Rapid preview**: 384x256, 4 steps — validates prompt direction in seconds
- **Prompt enhancement**: Qwen3.5-2B rewrites short prompts into detailed descriptions (lazy load/unload)
- **Progressive preview**: intermediate diffusion frames streamed to UI every 2 steps
- **Two-stage upscale**: generate at half resolution, neural 2x upscale (single safetensors file from Lightricks/LTX-2.3)
- **Batch generation queue**: priority-based job queue with cancel support
- **History view**: video archive with thumbnails, metadata, and delete
- **Model management**: download/delete models from Settings, disk usage tracking
- **Memory monitor**: real-time Metal memory stats with color-coded warnings
- **Streaming VAE decode**: frames decoded one-by-one into ffmpeg pipe (never all in RAM)
- **Export**: MP4 with H.264 encoding + AAC audio

### Not Yet Working

- **LoRA**: UI exists (scan, toggle, strength slider) but untested with real LTX-2.3 LoRA models
- **TTS voiceover**: sine-wave placeholder — interface ready for MLX-Audio (Kokoro, Dia, CSM)
- **Audio mixing**: ffmpeg-based mixer stub
- **Retake / Extend**: API endpoints exist but pipelines are stubs
- **FCPXML export**: endpoint exists but untested

---

## Models

| Model | Repo | Role | Size |
|-------|------|------|------|
| LTX-2.3 Distilled (int8) | `dgrauet/ltx-2.3-mlx-distilled-q8` | Video generation (transformer + VAE + audio + vocoder) | ~28 GB |
| Gemma 3 12B IT (4-bit) | `mlx-community/gemma-3-12b-it-4bit` | Text encoder | ~6 GB |
| Qwen3.5-2B (4-bit) | `mlx-community/Qwen3.5-2B-4bit` | Prompt enhancement | ~1.2 GB |
| LTX-2.3 Spatial Upscaler | `Lightricks/LTX-2.3` (single file) | Neural 2x upscale | ~1 GB |

The pipeline uses `mlx-video-with-audio` (Acelogic) as reference implementation for conditioning, denoising loop, and VAE encoding.

---

## Memory Management

Metal memory fragmentation is the #1 stability risk. The backend implements:

- **`aggressive_cleanup()`** at every pipeline stage boundary
- **Streaming VAE decode** to ffmpeg pipe — never all frames in RAM
- **Periodic model reload** every 5 generations to reclaim fragmented Metal buffers
- **Prompt enhancer isolation**: loaded, used, and unloaded before the video model (critical on 32 GB)

---

## Performance

Measured on Apple M2 Pro 32 GB (int8 quantized distilled model, 8 steps):

| Resolution | Frames | Time |
|-----------|--------|------|
| 384x256 | 9 | ~42s |
| 768x512 | 97 | ~495s (~8 min) |
| 1280x704 | 97 | ~1650s (~27 min) |

Marathon test: 10 consecutive 97-frame generations at 768x512 — no OOM, stable timing (484-507s per gen).

---

## Prompting Guide

LTX-2.3 responds best to structured, detailed prompts:

1. **Subject**: appearance, clothing, expression
2. **Action**: specific, chronological movements
3. **Environment**: location, lighting, atmosphere
4. **Camera**: angle and motion (dolly, pan, static, tracking)
5. **Style**: cinematic, realistic, animated, color grading
6. **Audio** (optional): sounds, music, dialogue

Example:
> A young woman with short brown hair wearing a white linen shirt walks along a sun-drenched coastal path. She pauses to look at the sea, smiling slightly. Gentle ocean breeze moves her hair. Camera slowly tracks alongside her. Warm golden hour light. Soft ambient waves and distant seagulls. Cinematic, shallow depth of field.

Use the Enhance button (Cmd+E) to auto-expand short prompts.

---

## Roadmap

- Audio quality tuning (vocoder output noisy)
- Real TTS voiceover via MLX-Audio (Kokoro, Dia, CSM)
- Background music generation
- LoRA testing with real LTX-2.3 compatible models
- Video retake (regenerate time segment) and extension (forward/backward)
- FCPXML export for Final Cut Pro / DaVinci Resolve
- Performance: persistent model server for mx.compile() and TeaCache (currently disabled — incompatible with subprocess-per-gen architecture)
- Simple timeline editor (if demand warrants it)
- V2V, A2V, keyframe interpolation

---

## Known Limitations

- **~8 min per generation** at 768x512 — TeaCache (0% hit rate with 8-step distilled) and mx.compile() (tracing overhead lost per subprocess) are both disabled
- **Audio quality**: vocoder output can sound noisy
- **32 GB RAM**: text encoder and video model cannot coexist — two-subprocess architecture is mandatory
- **No 16 GB support**: model weights alone are ~21 GB

---

## License

Apache-2.0 — same as [LTX Desktop](https://github.com/Lightricks/ltx-desktop) by Lightricks.

The LTX-2.3 model weights are subject to the [LTX Model License](https://ltx.io/model/license).
