# LTX Desktop macOS

Native macOS application for local AI video generation using **LTX-2.3** on Apple Silicon via **MLX**. No cloud required — everything runs on-device.

## What It Is

LTX Desktop replicates the features of [Lightricks LTX Desktop](https://github.com/Lightricks/ltx-desktop) but replaces NVIDIA/CUDA inference with 100% local inference on Apple Silicon unified memory. The official macOS version only supports cloud API — this project eliminates that dependency.

**Stack**: SwiftUI (native macOS app) + Python FastAPI backend + MLX inference engine.

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

> **16 GB machines**: very limited — low resolutions only, cloud text encoding recommended.

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

`setup.sh` verifies macOS version, installs dependencies, and creates the Python virtual environment.

### 3. Download models

```bash
bash scripts/download_models.sh
```

Downloads to `~/.cache/huggingface/`:
- `notapalindrome/ltx2-mlx-av` (~42 GB) — main video generation model (MLX-native, distilled + audio)
- `mlx-community/Qwen3.5-2B-4bit` (~1.2 GB) — prompt enhancement model

### 4. Start the backend

```bash
bash scripts/dev.sh
```

Or manually:

```bash
cd backend
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000
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
    +-- MLX-Audio (optional local TTS)
```

The backend runs as a separate process managed by the SwiftUI app's `ProcessManager`. Crash isolation means an OOM in the ML pipeline kills only the backend — the UI stays alive and offers a restart button.

---

## Features by Sprint

### Sprint 1 — Foundation
- FastAPI backend with health, system info, and memory endpoints
- Text-to-video generation pipeline (stubbed MLX, real ffmpeg output)
- WebSocket real-time progress streaming
- SwiftUI app: prompt field, generate button, progress bar
- `ProcessManager`: auto-starts backend, polls `/health` until ready, shows "Preparing engine..." splash
- Memory manager: `aggressive_cleanup()`, periodic model reload every 5 generations
- Marathon generation test (10 consecutive generations, memory stability check)

### Sprint 2 — Core Features
- Rapid preview: 384x256, 4 steps — validates prompt direction in seconds before full render
- Progressive diffusion display: intermediate frames streamed via WebSocket during generation
- Image-to-Video (I2V): drag-and-drop source image, conditions first frame
- Streaming VAE decode to ffmpeg pipe (never decodes all frames into RAM simultaneously)
- TeaCache MLX port: block-output caching, ~1.6-2.1x speedup, `rel_l1_thresh=0.03`
- Memory monitor panel in Settings: active/cache/peak/available with color-coded warnings

### Sprint 3 — Enhancement
- Prompt enhancement: Qwen3.5-2B-4bit via mlx-lm, lazy load/unload, never coexists with video model
- Retake pipeline: regenerate a specific time segment of an existing video
- Extend pipeline: extend a video forward or backward from its endpoint
- HistoryView: LazyVGrid of video thumbnails (AVFoundation async), detail panel with AVPlayer
- Settings: General tab (enhance toggle, output directory) + Models tab
- Enhance button in GenerationView (Cmd+E, sparkles icon, spinner during enhancement)

### Sprint 4 — LoRA, Audio, Export
- LoRA manager: scan `~/.ltx-desktop/loras/` for `.safetensors`, compatibility check, built-in stubs
- LoRAView: toggle switches, incompatibility warnings, empty state
- Audio TTS engine: MLX-Audio stub (sine-wave placeholder, mlx_audio-ready interface)
- Audio mixer: ffmpeg filter graph with amix, volume ducking (music ducks to 20% with TTS active)
- Export: H.264/H.265/ProRes, MP4/MOV via ffmpeg re-encode
- FCPXML export: FCPXML 1.11 for Final Cut Pro, compatible with Premiere/DaVinci via XML
- ExportSheet in HistoryView: format/codec/bitrate pickers, Reveal in Finder
- FPS picker (24/30) and word count label (orange >150 words, red >200 words) in GenerationView

### Sprint 5 — Stability & Release
- Full API test suite covering all endpoints (Sprint 1-4)
- Marathon test: 10 consecutive generations, memory stability, TeaCache validation
- README and release preparation

---

## Memory Management

Metal memory fragmentation is the #1 stability risk for repeated video generation on Apple Silicon. The backend implements mandatory mitigations:

- **`aggressive_cleanup()`** called at every pipeline stage boundary: after text encoding, after diffusion, after upscale, after VAE decode, after audio decode, after every job
- **Streaming VAE decode**: frames decoded one-by-one into an ffmpeg pipe — never all frames in RAM simultaneously
- **Periodic model reload**: full unload + reload every 5 generations to reclaim fragmented Metal buffers
- **Prompt enhancer isolation**: Qwen3.5-2B is loaded, used, and unloaded before the video model loads (critical on 32 GB machines)

Memory thresholds (shown in the Settings memory panel):

| Condition | Severity |
|-----------|----------|
| Cache > 2x active memory | Warning |
| Peak > 85% total RAM | Critical |
| System available < 4 GB | Critical |

---

## Performance

Four optimizations are applied from the start:

1. **Kernel warm-up** (`warmup_pipeline()`): forces Metal kernel compilation at startup so the first real generation does not pay the JIT cost. Shows "Preparing engine..." splash during warm-up.
2. **`mx.compile()` on model forward pass**: fuses element-wise ops into single Metal kernels, reducing dispatch overhead (~10-15% speedup).
3. **Latent pool pre-allocation** (`LatentPool`): pre-allocates a GPU buffer at max expected shape to reduce Metal allocator pressure.
4. **TeaCache**: block-output caching for the DiT — skips recomputing transformer blocks whose output barely changes between timesteps. Proven 1.6-2.1x speedup on LTX-Video at `rel_l1_thresh=0.03`.

Approximate generation times (with real MLX inference, estimated):

| Hardware | 768x512, 97 frames, distilled |
|----------|-------------------------------|
| M2 Max 32 GB | ~35-45s (with TeaCache) |
| M3 Max 64 GB | ~25-32s |
| M4 Ultra 128 GB | ~15-20s |

---

## API Reference

The backend exposes a REST + WebSocket API on `http://localhost:8000`.

### Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/generate/text-to-video` | Full T2V generation |
| POST | `/api/v1/generate/image-to-video` | I2V with reference image |
| POST | `/api/v1/generate/preview` | Fast preview (384x256, 4 steps) |
| POST | `/api/v1/generate/retake` | Regenerate a time segment |
| POST | `/api/v1/generate/extend` | Extend video forward or backward |

### Queue & Progress

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/queue` | List all jobs |
| GET | `/api/v1/queue/{job_id}` | Get job status and result |
| POST | `/api/v1/queue/{job_id}/cancel` | Cancel a job |
| WS | `/ws/progress/{job_id}` | Real-time progress + preview frames |

### Prompt Enhancement

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/prompt/enhance` | Enhance prompt via Qwen3.5-2B (503 if mlx-lm absent) |

### LoRA

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/loras` | List available LoRAs |
| POST | `/api/v1/loras/load` | Load (activate) a LoRA |
| POST | `/api/v1/loras/unload/{id}` | Unload a LoRA |

### Audio

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/audio/tts` | Text-to-speech (local stub) |
| POST | `/api/v1/audio/music` | Generate background music (stub) |
| POST | `/api/v1/audio/mix` | Mix audio tracks into video |

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/export/video` | Re-encode with H.264/H.265/ProRes |
| POST | `/api/v1/export/fcpxml` | FCPXML 1.11 for Final Cut Pro |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/system/health` | Health check |
| GET | `/api/v1/system/info` | Chip, RAM, macOS version |
| GET | `/api/v1/system/memory` | Real-time Metal memory stats |

---

## Prompting Guide

LTX-2.3 responds best to structured, detailed prompts:

1. **Subject**: appearance, clothing, expression
2. **Action**: specific, chronological movements
3. **Environment**: location, lighting, atmosphere
4. **Camera**: angle and motion (dolly, pan, static, tracking)
5. **Style**: cinematic, realistic, animated, color grading
6. **Audio** (optional): sounds, music, dialogue in quotes

Example:
> A young woman with short brown hair wearing a white linen shirt walks along a sun-drenched coastal path. She pauses to look at the sea, smiling slightly. Gentle ocean breeze moves her hair. Camera slowly tracks alongside her. Warm golden hour light. Soft ambient waves and distant seagulls. Cinematic, shallow depth of field.

Use the Enhance button (Cmd+E) to auto-expand short prompts via Qwen3.5-2B.

---

## LoRA Support

Place `.safetensors` LoRA files in `~/.ltx-desktop/loras/`. The app scans this directory on launch.

> **Important**: LoRAs must be trained for the LTX-2.3 latent space. LTX-2.0 LoRAs are incompatible and will be flagged in the LoRAView.

Built-in stubs (always visible, download not yet implemented):
- **Camera Control** — dolly in/out, jib, static shots
- **Detail Enhancement** — sharpens fine texture and detail

---

## File Layout

```
ltx-desktop-macos/
├── CLAUDE.md                     # Root project spec and architecture
├── AGENTS.md                     # Agent team roles and sprint plan
├── README.md                     # This file
├── app/                          # SwiftUI macOS app (Xcode project)
│   └── LTXDesktop/
│       ├── Views/                # GenerationView, HistoryView, LoRAView, ExportSheet, SettingsView
│       ├── ViewModels/           # GenerationVM, HistoryVM, LoRAVM, MemoryVM
│       ├── Models/               # VideoItem, ModelInfo, LoRAInfo, GenerationJob, etc.
│       └── Services/             # BackendService (HTTP+WS), ProcessManager, MemoryMonitor
├── backend/                      # Python FastAPI backend
│   ├── main.py                   # All API routes (single-file MVP)
│   ├── engine/
│   │   ├── memory_manager.py     # aggressive_cleanup, periodic reload, memory stats
│   │   ├── model_manager.py      # MLX model load/unload lifecycle
│   │   ├── prompt_enhancer.py    # Qwen3.5-2B lazy load/unload
│   │   ├── lora_manager.py       # LoRA scan, load, compatibility check
│   │   ├── teacache.py           # TeaCache MLX port (block-output caching)
│   │   └── pipelines/
│   │       ├── text_to_video.py  # Full T2V pipeline
│   │       ├── image_to_video.py # I2V pipeline
│   │       ├── preview.py        # Rapid preview (384x256, 4 steps)
│   │       ├── retake.py         # Segment regeneration
│   │       └── extend.py         # Forward/backward extension
│   └── audio/
│       ├── tts_engine.py         # Local TTS (MLX-Audio interface)
│       └── audio_mixer.py        # ffmpeg-based multi-track mixer
├── tests/
│   ├── conftest.py               # Backend fixture (auto-start/stop)
│   ├── test_api.py               # API integration tests (Sprint 1-4)
│   └── test_marathon.py          # 10-gen stability test (release gate)
├── scripts/
│   ├── setup.sh                  # Full installation
│   ├── download_models.sh        # HuggingFace model download
│   └── dev.sh                    # Dev launch (backend + Xcode hint)
└── models/                       # Gitignored — ~42 GB+
```

---

## Running Tests

```bash
# Full API test suite (requires running backend)
cd /path/to/ltx-desktop-macos
backend/.venv/bin/pytest tests/test_api.py -v

# Marathon stability test (10 consecutive generations)
backend/.venv/bin/pytest tests/test_marathon.py -v -s

# Both
backend/.venv/bin/pytest tests/ -v -s
```

The marathon test is the release gate. Pass criteria:
- 10/10 generations complete without crash
- Memory after generation 10 within 20% of generation 1
- No generation takes more than 2x longer than generation 1
- All 10 output files are valid MP4s

---

## What Is Not Yet Implemented

These are planned but not yet built:

- **Real MLX inference**: all pipelines are currently stubs that produce ffmpeg-generated placeholder videos. The pipeline structure, memory management, and API are fully in place — plugging in the real `mlx-video-with-audio` calls is the next step.
- **Real prompt enhancement**: mlx-lm integration is wired up — once `mlx-lm` is installed, the `/prompt/enhance` endpoint uses Qwen3.5-2B automatically.
- **Real TTS**: `tts_engine.py` generates a sine-wave placeholder. The interface is designed for MLX-Audio (Kokoro, CSM).
- **Model download UI**: the Models tab in Settings shows model status but download buttons are not yet wired to the backend.
- **Timeline editor**: deliberately deferred — export to FCPXML for Final Cut Pro is the recommended workflow instead.
- **Batch generation queue**: coming in a future sprint.

---

## License

Apache-2.0 — same as [LTX Desktop](https://github.com/Lightricks/ltx-desktop) by Lightricks.

The LTX-2.3 model weights are subject to the [LTX Model License](https://ltx.io/model/license).
