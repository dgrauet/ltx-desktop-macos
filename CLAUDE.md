# CLAUDE.md — LTX Desktop macOS (Apple Silicon + MLX Local Inference) — Final

## Project Vision

Build a native macOS application that replicates the features of **LTX Desktop** (the open-source video editor by Lightricks built on LTX-2.3), but replacing NVIDIA/CUDA inference with **100% local inference on Apple Silicon via MLX**. The official macOS version of LTX Desktop only uses the cloud API — our goal is to eliminate that dependency and run entirely locally.

**Differentiation**: no existing app combines a full AI video generation + editing workflow running locally on Apple Silicon. The ltx-video-mac proof of concept exists but is a simple generator — not an editor.

---

## Technical Context

### LTX-2.3 — The Model
- **Architecture**: Diffusion Transformer (DiT) 19B parameters, synchronized audio+video generation in a single pass
- **Capabilities**: text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, retake (regenerate a specific time region)
- **Resolutions**: up to native 1080p (1920×1080), native portrait (1080×1920), landscape, 16:9, 9:16
- **FPS**: up to 30 fps output (50 fps supported by the full model)
- **Audio**: built-in vocoder, synchronized generation (dialogue, ambience, music)
- **Variants**:
  - `ltx-2.3-dev` — full bf16 model (~42GB)
  - `ltx-2.3-distilled` + LoRA — fast inference (8 steps stage 1, 4 steps stage 2)
  - `ltx-2.3-distilled-fp8` — reduced memory footprint
  - Spatial and temporal latent upscalers
- **Text encoder**: Gemma 3 (12B) — encodes prompts into dense embeddings. **Note**: this is for video generation text encoding, NOT for prompt enhancement (see Prompt Enhancement section).
- **VAE**: rebuilt architecture for LTX-2.3, better texture and fine detail preservation
- **HuggingFace checkpoints**: `Lightricks/LTX-2.3`

### MLX on Apple Silicon — Inference Stack
- **Framework**: MLX (Apple, open-source MIT) — array framework optimized for Apple Silicon
- **Key advantage**: unified CPU/GPU memory — no data copying between RAM and VRAM
- **APIs**: Python, Swift, C++, C
- **Existing packages to leverage**:
  - `mlx` — core framework (pip install mlx)
  - `mlx-video-with-audio` (PyPI v0.1.3) — LTX-2 pipeline ported to MLX, T2V and I2V with synchronized audio
  - `mlx-audio` — TTS, STT, STS optimized for Apple Silicon
  - `mlx-lm` — LLM inference (for prompt enhancement and text encoding)
  - `mlx-vlm` — Vision Language Models
  - DiffusionKit — diffusion models on Apple Silicon (Core ML + MLX)
  - MFLUX — MLX port of FLUX (reference for porting diffusion models)
- **Pre-converted MLX model**: `notapalindrome/ltx2-mlx-av` (~42GB, distilled, audio+video)
- **Weight conversion**: PyTorch → MLX via `mlx_video.convert`
- **Quantization**: int4, int8 support via MLX to reduce memory footprint

### Existing Reference — ltx-video-mac
The `james-see/ltx-video-mac` repo is an existing native SwiftUI macOS app that runs LTX-2 inference via MLX. It serves as a **proof of concept** but does NOT cover LTX Desktop's video editing features. Current capabilities:
- Text-to-video with synchronized audio
- Generation queue with real-time progress tracking
- Generated video history, parameter presets
- Prompt enhancement via Gemma, voiceover (ElevenLabs cloud or local MLX-Audio)
- Background music (54 genre presets), auto-installation of missing Python packages

### Target Reference — LTX Desktop (Lightricks)
The `Lightricks/ltx-desktop` repo is the official app:
- **Frontend**: TypeScript + React (Electron renderer)
- **Backend**: Python FastAPI (localhost:8000)
- **Shell**: Electron (lifecycle, OS integration, ffmpeg, Python backend management)
- **Local inference**: Windows only (NVIDIA GPU)
- **macOS**: API-only mode (no local inference)
- **License**: Apache-2.0

---

## Application Architecture

### Chosen Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│              SwiftUI Frontend (native macOS)             │
│   Generation · Preview · Settings · File management     │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP + WebSocket (localhost:8000)
                           ▼
┌─────────────────────────────────────────────────────────┐
│       Python Backend (FastAPI, separate process)        │
│   Pipelines · Queue · Model management · Export         │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  MLX Engine  │  │   ffmpeg     │  │  MLX-Audio   │
│  (DiT+VAE   │  │  (Encode/    │  │  (TTS/STT    │
│  Inference)  │  │   Export)    │  │   local)     │
└──────────────┘  └──────────────┘  └──────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│     Apple Silicon (Metal GPU)        │
│     Unified Memory (shared RAM)      │
└──────────────────────────────────────┘
```

### Why FastAPI in a Separate Process (not PythonKit)

**PythonKit was considered** (calling Python directly from Swift) to reduce complexity. However, FastAPI in a separate process is preferred because:

1. **Crash isolation**: an OOM in MLX kills the backend process, not the UI. Video generation can consume 30-60GB RAM — crashes are expected.
2. **GIL blocking**: a generation takes 2-10 minutes. PythonKit would block the Swift thread or require complex cross-language threading.
3. **Real-time progress**: WebSocket is the cleanest way to stream generation progress to the UI during multi-minute operations.
4. **Proven pattern**: both LTX Desktop (Electron → FastAPI) and ltx-video-mac (SwiftUI → Python subprocess) use this exact architecture.
5. **Debug & iterate**: the backend can be restarted independently, tested via curl/httpie, and developed separately.

**Simplification rule**: start with a single `main.py` file, not the full folder structure. Extract into modules only as complexity grows.

### Tauri Option (alternative for faster bootstrapping)
If capitalizing on LTX Desktop's existing React frontend is preferred:
```
Frontend: React + TypeScript (fork of LTX Desktop frontend)
Shell: Tauri 2.0 (Rust, native WKWebView webview on macOS)
Backend: Python FastAPI (identical)
Advantage: reuse of existing UI code, faster to market
Drawback: less native than pure SwiftUI, Rust dependency
```

---

## Prompt Enhancement — Qwen 3.5 (NOT Gemma)

### Why Not Gemma 12B
The original plan used Gemma 3 12B for prompt enhancement. This is **not viable** — loading Gemma 12B (~10-12GB) alongside LTX-2.3 (~42GB) exceeds the RAM of most machines.

### Recommended: Qwen3.5-2B (released March 2, 2026)

**Qwen3.5-2B** is the ideal replacement:

| Property | Gemma 3 12B | Qwen3.5-2B (4-bit) |
|----------|-------------|---------------------|
| RAM usage | ~10-12GB | **~1.2GB** |
| Instruction following | Good | Excellent (scaled RL training) |
| MLX support | Via mlx-lm | **Native mlx-lm, confirmed working** |
| License | Google terms | **Apache 2.0** |
| Context window | 8K | **262K tokens** |
| Architecture | Dense | Hybrid (Gated Delta Networks + MoE) |
| Multilingual | 100+ langs | **201 languages** |
| Release date | 2024 | **March 2, 2026** |

**Available on HuggingFace**: `Qwen/Qwen3.5-2B` and MLX-quantized versions on `mlx-community`.

### Lazy Load Strategy (critical for memory)

The prompt enhancer and the video model should **never be loaded simultaneously**:

```
1. User writes prompt
2. Load Qwen3.5-2B (~1.2GB) → enhance prompt → unload Qwen3.5-2B
3. Free memory: mx.clear_cache()
4. Load LTX-2.3 (~42GB) → generate video → keep loaded for next generation
5. If user wants to enhance another prompt → unload LTX-2.3 first (or use cached model)
```

For machines with 64GB+ RAM, both can coexist. For 32GB machines, sequential loading is mandatory.

### Fallback: Cloud text encoding
For machines with very low RAM (16-32GB), offer the option to use the LTX API for text encoding (free on LTX Console). The API key is stored locally.

---

## Features to Implement

### Phase 1 — AI Video Generator (MVP)

> **Product 1**: A polished video generator. No timeline. Prompt → Video → Export.
> This is what the majority of users want. Ship this first.

#### 1.1 Text-to-Video (T2V)
- Text prompt field with word counter (max 200 words recommended)
- Adjustable parameters:
  - Resolution: 768×512, 512×768, 1280×704, 704×1280, 1920×1080, 1080×1920
  - Frame count: 25, 49, 65, 97, 121, 177, 257 (multiples of 8 + 1)
  - FPS: 24, 30
  - Guidance scale, inference steps (8 distilled, 20-40 full)
  - Seed (reproducibility), negative prompt
- Two-stage pipeline:
  1. Stage 1: low-resolution generation (768×512 or 512×768)
  2. Stage 2: 2× spatial latent upscale via the upscaler model
- Synchronized audio generated automatically in a single pass
- Real-time progress bar via WebSocket

#### 1.2 Rapid Preview (critical UX feature)
- **Before launching full generation**, produce a fast low-res preview:
  - Resolution: 384×256
  - Steps: 4 (distilled single-stage pipeline)
  - Time: a few seconds
- User validates the direction, then launches full render
- Uses `TI2VidOneStagePipeline` or `DistilledPipeline`

#### 1.3 Progressive Diffusion Display
- During denoising, extract intermediate latents and display partially decoded frames
- Show the video "forming" in real time — makes long generation feel interactive
- MLX lazy evaluation + unified memory makes this low-overhead
- Decode every N steps (e.g., every 4th step) to avoid excessive VAE overhead

#### 1.4 Image-to-Video (I2V)
- Upload a reference image (drag & drop or file picker)
- The image conditions the first frame of the video
- Same parameters as T2V + source image
- 2.3 improvement: less freezing, more real motion (reduced Ken Burns effect)

#### 1.5 Prompt Enhancement (Qwen3.5-2B)
- Rewrites short prompts into detailed LTX-2.3-optimized prompts
- Preview enhanced prompt before generation — user can edit
- Lazy load/unload to preserve memory for video generation
- Toggle on/off in settings
- Built-in prompting guide in UI

#### 1.6 Model Management
- Automatic checkpoint download from HuggingFace on first launch
- Model choice: distilled (fast) vs full dev (max quality)
- Disk space indicator (~42GB), download progress
- Support for pre-converted MLX models (`notapalindrome/ltx2-mlx-av`)
- Cache in `~/.cache/huggingface/`

#### 1.7 Video Retake & Extension
- **Retake**: regenerate a specific time segment of an existing video (keep the rest)
- **Extension**: extend a video forward or backward from its last/first frame
- Both are high-value features that don't require a timeline

#### 1.8 LoRA Support
- Camera Control LoRA: dolly in/out, jib up/down, static shots
- Detail Enhancement LoRA
- Custom style LoRAs (.safetensors loading)
- IC-LoRA: video-to-video and image-to-video transformations
- LoRA selection UI with activation toggles
- **Note**: LoRAs must be compatible with the 2.3 latent space (2.0 LoRAs won't work)

#### 1.9 Audio
- Built-in synchronized audio from LTX-2.3 (automatic, no config needed)
- Optional local TTS voiceover via MLX-Audio (Kokoro, Dia, CSM)
- Optional background music generation (54+ genre presets)
- Automatic mixing: music at 30% volume, ducked to 20% with voiceover
- Add audio to previously generated videos (right-click → Add Audio)

#### 1.10 Export
- Video export: MP4 (H.264, H.265/HEVC), MOV
- Separate audio export: WAV, AAC
- Export settings: codec, bitrate, final resolution
- ffmpeg integration for final encoding

#### 1.11 History & Projects
- Video Archive: browse, preview, manage all generated videos
- Parameter preset saving and loading
- Metadata: prompt, seed, parameters, date, generation duration
- Thumbnail grid view

### Phase 1.5 — Professional Export (no timeline needed)

#### FCPXML Export
- Export generated clips as **FCPXML** (Final Cut Pro XML) — the native macOS editing format
- Also support Adobe Premiere Pro XML and DaVinci Resolve XML
- This lets users bring AI-generated clips into their existing editing workflow
- **Avoids months of NLE development** while providing professional interoperability

#### Batch Generation
- Queue multiple prompts for sequential generation
- Estimated time remaining per job and total
- Background generation while browsing history
- Priority management

### Phase 2 — Advanced Workflows (if product finds traction)

#### 2.1 Simple Timeline (optional)
- Non-linear timeline with video and audio tracks
- 2 video tracks, 5 audio tracks
- Basic operations: trim, split, duplicate, reorder, insert
- Playback preview with audio/video sync
- **Only build this if Phase 1 proves user demand**

#### 2.2 Advanced Generation Modes
- **Video-to-Video (V2V)**: transform an existing video via prompt
- **Audio-to-Video (A2V)**: generate video conditioned on an audio file
- **Keyframe Interpolation**: interpolate between keyframe images
- **Multi-keyframe conditioning**: multiple reference images at different timeline points

#### 2.3 Native Portrait Resolution (9:16)
- Native 1080×1920 for TikTok, Instagram Reels, YouTube Shorts
- Trained on vertical data (not cropped from landscape)
- Aspect ratio selector in the UI

#### 2.4 Latent Cache System
- Save video and audio latents to disk after generation
- Enables instant retake (no full re-generation)
- Enables faster extension (only generate new segment)
- LRU cache with configurable disk budget (default: 50GB)
- Stored alongside generated videos in project folder

#### 2.5 Depth & Edge Conditioning
- Depth map conditioning support
- Canny edge conditioning
- Control adapters for precise motion control

#### 2.6 Intelligent Model Management
- Lazy loading: only load modules when needed
- Dynamic unloading: free model when idle for N minutes
- Automatic memory pressure detection via `os_proc_available_memory()`
- Preemptive unloading when system memory pressure is high
- Display memory usage per loaded component in settings

#### 2.7 Chunked Generation for Long Videos (Temporal Sliding Window)

For generating videos **longer than what fits in a single pass** (e.g., pushing from 97 to 177+ frames on a 32GB machine), use overlapping temporal chunks with blending.

**Important architectural constraint**: LTX-2.3 is a Diffusion Transformer (DiT) with **global temporal attention** — every frame attends to every other frame during diffusion. This means you **cannot** arbitrarily window the diffusion loop for a single clip. The model was trained with full temporal context and will produce degraded output if you slice frames mid-diffusion.

**What DOES work**: generating **separate overlapping segments** as complete generation passes, then blending them:

```
Segment 1: generate frames 1-97     (full diffusion pass, all 97 frames in attention)
Segment 2: generate frames 73-169   (conditioned on frame 73 from segment 1 as I2V)
Blend:     frames 73-97 = alpha-blend between segment 1 and segment 2
Result:    seamless 169-frame video
```

Implementation approach:
- Use the **ExtendPipeline** (forward extension) which is purpose-built for this
- The extend pipeline conditions on the last N frames of the previous segment
- Overlap region (e.g., 24 frames) uses linear alpha blending for smooth transitions
- Audio is extended similarly, with crossfade in the overlap region
- Each segment is a full diffusion pass → no quality degradation

Memory impact:
```
Without chunking (177 frames):  model (38GB) + latents+buffers (~4GB) = ~42GB → needs 64GB Mac
With chunking (2 × 97 frames):  model (38GB) + latents+buffers (~2GB) = ~40GB → fits on 48GB Mac
```

The savings come from smaller per-pass latent/buffer allocations, not from the model weights (which dominate memory regardless). The real benefit is enabling **longer total video duration on memory-constrained hardware**.

**Where this does NOT help**: reducing memory for a single 97-frame generation. The model weights (~38GB) are the dominant cost, and the latent tensor for 97 frames at compressed resolution is only ~150-300MB. The attention computation uses efficient attention (not materializing the full N² matrix), so windowing the attention wouldn't save significant memory anyway.

---

## Project Structure

```
ltx-desktop-macos/
├── CLAUDE.md                          # This file
├── README.md
├── LICENSE                            # Apache-2.0
│
├── app/                               # SwiftUI Application (Xcode project)
│   ├── LTXDesktop.xcodeproj
│   ├── LTXDesktop/
│   │   ├── App.swift                  # Entry point
│   │   ├── ContentView.swift          # Main layout
│   │   ├── Views/
│   │   │   ├── GenerationView.swift   # Prompt + parameters + generate
│   │   │   ├── PreviewView.swift      # Video preview + progressive display
│   │   │   ├── HistoryView.swift      # Video archive grid
│   │   │   ├── SettingsView.swift     # App + model settings + memory monitor
│   │   │   └── LoRAView.swift         # LoRA browser and activation
│   │   ├── Models/
│   │   │   ├── GenerationJob.swift    # Generation job state
│   │   │   ├── VideoItem.swift        # Generated video metadata
│   │   │   └── AppSettings.swift      # Persistent settings (UserDefaults)
│   │   ├── Services/
│   │   │   ├── BackendService.swift   # HTTP + WebSocket to Python backend
│   │   │   ├── ProcessManager.swift   # Start/stop/monitor Python backend
│   │   │   └── MemoryMonitor.swift    # Metal memory: active/cache/peak + fragmentation alerts
│   │   └── Utils/
│   │       └── FFmpegWrapper.swift    # ffmpeg shell commands
│   └── Resources/
│       └── Assets.xcassets
│
├── backend/                           # Python FastAPI Backend
│   ├── pyproject.toml                 # uv config
│   ├── main.py                        # FastAPI app — START HERE, single file MVP
│   ├── engine/
│   │   ├── mlx_inference.py          # Main MLX inference pipeline
│   │   ├── pipelines/
│   │   │   ├── text_to_video.py      # T2V pipeline (two-stage + single-stage preview)
│   │   │   ├── image_to_video.py     # I2V pipeline
│   │   │   ├── retake.py             # Retake pipeline (segment regeneration)
│   │   │   ├── extend.py             # Video extension (forward/backward)
│   │   │   └── upscaler.py           # 2× spatial latent upscale
│   │   ├── model_manager.py          # MLX model loading/unloading with lazy load
│   │   ├── prompt_enhancer.py        # Qwen3.5-2B via mlx-lm (lazy load/unload)
│   │   ├── memory_manager.py         # ★ CRITICAL: aggressive Metal cleanup, periodic model reload, VAE streaming
│   │   └── lora_manager.py           # LoRA loading and application
│   ├── audio/
│   │   ├── tts_engine.py             # Local TTS via MLX-Audio
│   │   └── audio_mixer.py            # Multi-track audio mixing
│   ├── export/
│   │   ├── video_encoder.py          # Final encoding via ffmpeg
│   │   └── fcpxml_export.py          # FCPXML + Premiere XML export
│   └── utils/
│       ├── config.py                 # Global configuration
│       └── system_info.py            # Apple Silicon hardware detection
│
├── models/                            # Models folder (gitignored, ~42GB+)
│   └── README.md                     # Download instructions
│
├── scripts/
│   ├── setup.sh                      # Full installation (uv, python, deps, ffmpeg)
│   ├── download_models.sh            # Model download from HF
│   └── dev.sh                        # Dev launch (backend + open Xcode)
│
└── tests/
    ├── test_inference.py             # MLX pipeline tests
    ├── test_api.py                   # FastAPI endpoint tests
    ├── test_memory.py                # Memory management tests
    └── test_marathon.py              # ★ CRITICAL: 10 consecutive generations stability test
```

---

## Python Dependencies (backend)

```toml
[project]
name = "ltx-desktop-macos-backend"
requires-python = ">=3.12"
dependencies = [
    # MLX stack
    "mlx>=0.31.0",
    "mlx-video-with-audio>=0.1.3",
    "mlx-lm>=0.24.0",

    # API
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "websockets>=13.0",
    "python-multipart>=0.0.12",

    # ML / Tensor
    "numpy>=1.26.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.26.0",
    "transformers>=4.51.0",

    # Video/Audio processing
    "opencv-python>=4.10.0",
    "tqdm>=4.66.0",
    "pillow>=10.4.0",
    "soundfile>=0.12.0",

    # Audio (optional, for local TTS)
    "mlx-audio>=0.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27.0",
    "ruff>=0.7.0",
]
```

---

## Hardware Configuration and Limits

### Apple Silicon Compatibility Matrix

| Chip | RAM | Supported Model | Max Resolution | Max Frames | With Chunked Extend | Notes |
|------|-----|-----------------|----------------|------------|---------------------|-------|
| M1 | 16GB | Distilled int4 | 768×512 | 49 | 97 | Very limited, consider cloud text encoding |
| M1 Pro/Max | 32GB | Distilled int8 | 1280×704 | 97 | 177 | Good for prototyping |
| M1 Ultra | 64GB | Distilled bf16 | 1280×704 | 177 | 257+ | Comfortable |
| M2 Ultra | 64-192GB | Full bf16 | 1920×1080 | 257 | 500+ | Production ready |
| M3 Max | 36-128GB | Distilled bf16+ | 1920×1080 | 177 | 257+ | Good GPU performance |
| M3 Ultra | 128-512GB | Full bf16 + upscaler | 1920×1080 | 257+ | unlimited | Optimal |
| M4 Max | 36-128GB | Distilled bf16+ | 1920×1080 | 177 | 257+ | Best perf/watt |
| M4 Ultra | 128-512GB | Full pipeline | 1920×1080+ | 257+ | unlimited | Best case |
| M5 (2026) | 24-128GB | Distilled bf16 | 1920×1080 | 177 | 257+ | Neural Accelerators, ~2-4× speedup |

*"With Chunked Extend" column uses overlapping segment generation via ExtendPipeline (Phase 2 feature). Quality is near-identical to single-pass generation thanks to I2V-conditioned segments and overlap blending.*

### Automatic Safe Mode
On launch, detect:
- Chip model via `sysctl -n machdep.cpu.brand_string` or IOKit
- Total RAM via `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')`
- Available RAM via `os_proc_available_memory()` (macOS API)
- Automatically limit resolution and frame count based on available RAM
- Show warning banner if RAM < 32GB with recommendation to use cloud text encoding

### Memory Budget Breakdown (32GB machine example)
```
macOS system overhead:         ~4GB
LTX-2.3 distilled (int8):    ~21GB   ← MODEL WEIGHTS: the dominant cost
Video latents + buffers:       ~3GB   ← See detailed breakdown below
Metal cache + fragmentation:   ~2-4GB ← GROWS over time without cleanup!
Prompt enhancer (when active): ~1.2GB (Qwen3.5-2B 4-bit, unloaded during generation)
Available headroom:            ~1-3GB
```
**Without aggressive cleanup, the "Metal cache + fragmentation" row grows unbounded across generations, eventually consuming all headroom and causing OOM.**

### Where Memory Actually Goes (common misconception)

A frequent mistake is thinking that video latents are the main memory consumer. They are not — **model weights dominate by 10×**:

```
Component breakdown for 97 frames @ 768×512 distilled:

Model weights (19B params, int8):      ~21,000 MB  ← 85% of total
Text encoder (Gemma connector):         ~1,200 MB
Upscaler model:                          ~800 MB
─────────────────────────────────────────────────
Video latent tensor (97×128×96×64, bf16):  ~150 MB  ← only 0.6% of total
Audio latent tensor:                        ~20 MB
Diffusion intermediates (noise pred, etc): ~300 MB
Attention KV cache (efficient attention):  ~500 MB
VAE decode peak (if not streaming):      ~1,500 MB  ← streaming eliminates this
─────────────────────────────────────────────────
Total latent/buffer:                     ~2,500 MB  ← 10% of total
```

**Key insight**: techniques that reduce latent memory (like temporal windowing) save ~2GB at best. Techniques that manage model weights (quantization int8→int4) save ~10GB. Techniques that manage Metal cache (aggressive cleanup) prevent unbounded growth. Prioritize accordingly.

---

## Backend API — Main Endpoints

### Generation
```
POST /api/v1/generate/text-to-video      # Full T2V generation
POST /api/v1/generate/image-to-video     # I2V with reference image
POST /api/v1/generate/preview            # Fast low-res preview (384×256, 4 steps)
POST /api/v1/generate/retake             # Regenerate a time segment
POST /api/v1/generate/extend             # Extend video forward/backward
```

### Queue & Progress
```
GET  /api/v1/queue                       # List jobs
POST /api/v1/queue/{job_id}/cancel       # Cancel a job
WS   /ws/progress/{job_id}              # Real-time progress + intermediate frames
```

### Models
```
GET  /api/v1/models                      # List installed models
POST /api/v1/models/download             # Download a model from HF
GET  /api/v1/loras                       # List available LoRAs
POST /api/v1/loras/load                  # Load a LoRA
```

### Audio
```
POST /api/v1/audio/tts                   # Local text-to-speech
POST /api/v1/audio/music                 # Generate background music
POST /api/v1/audio/mix                   # Mix audio tracks onto video
```

### Export
```
POST /api/v1/export/video                # Final export with ffmpeg
POST /api/v1/export/fcpxml               # FCPXML export for Final Cut Pro
POST /api/v1/export/premiere-xml         # XML export for Premiere Pro / DaVinci
```

### System
```
GET  /api/v1/system/info                 # Hardware info (chip, RAM, GPU)
GET  /api/v1/system/health               # Health check
GET  /api/v1/system/memory               # Real-time RAM monitoring
POST /api/v1/prompt/enhance              # Prompt enhancement via Qwen3.5-2B
```

---

## MLX Inference Pipeline — Technical Detail

### Two-Stage Pipeline (production quality)

```python
# Stage 1: Low-resolution generation
# Uses the distilled model with 8 predefined sigmas
# Resolution: 768×512 (landscape) or 512×768 (portrait)
# Generates video + audio latents simultaneously

# Stage 2: 2× spatial latent upscale
# Uses the separate upscaler model
# Applies 4 additional denoising steps
# Result: 1536×1024 or 1024×1536

# Final VAE decoding
# Video: latent decoding → pixel frames
# Audio: latent decoding → waveform via vocoder
```

### Rapid Preview Pipeline (single-stage)

```python
# Uses DistilledPipeline with 8 predefined sigmas
# Resolution: 384×256 (half of base)
# Steps: 4
# No upscaler stage
# Result: low-res preview in seconds
# User validates → launch full two-stage render
```

### Progressive Diffusion Display

```python
# During denoising loop:
# Every 4th step, extract current latent
# Partial VAE decode → JPEG frame
# Send via WebSocket to frontend for display
# Cost: ~200ms per intermediate decode on M3 Max
# UX: user sees the video "forming" in real-time
```

### ⚠️ CRITICAL: Metal Memory Fragmentation and Cache Management

**This is the #1 stability risk of the entire project.** Almost every MLX video project underestimates this. Understanding and handling it correctly is the difference between a demo that works once and a production app that runs for hours.

#### The Problem: Unified Memory ≠ Infinite Memory

On Apple Silicon, CPU and GPU share the same physical memory. This sounds simple but creates a dangerous illusion. In practice:

- MLX tensors live in Metal GPU memory
- MLX maintains an internal GPU cache
- Metal keeps its own buffer allocations
- **Memory is NOT returned to the system immediately after `del tensor`**

The result: after 2-3 consecutive video generations, memory usage grows monotonically even though each generation in isolation fits in RAM:

```
Generation 1: 32GB → 45GB (OK)
Generation 2: 45GB → 55GB (slow)
Generation 3: 55GB → 65GB (OOM crash)
```

This is **not** because the model is too large. It's because of **Metal buffer fragmentation and MLX cache accumulation**.

#### Why Video Makes This Worse

Video latents are enormous compared to image or text workloads. A typical generation:

```
97 frames × 768×512 latent × multiple channels = several GB of temporary buffers
```

And the pipeline involves multiple memory-heavy stages (diffusion, upscaler, VAE decode, audio decode), each creating large intermediate allocations that fragment the Metal memory pool.

#### The Classic Symptom

```
Generation 1 → OK (fast)
Generation 2 → OK (slightly slower)
Generation 3 → noticeable slowdown (system starts memory compression)
Generation 4 → OOM crash or kernel panic
```

Users blame the model size. The real cause is **accumulated Metal cache and fragmented buffers**.

#### Mandatory Solution: Aggressive Cleanup Between Stages

```python
import mlx.core as mx
import gc

def aggressive_cleanup():
    """Force-free Metal memory between pipeline stages.
    MUST be called between every major stage and between every generation."""
    gc.collect()                    # Python garbage collector
    mx.clear_cache()                # MLX Metal cache (mx.metal.clear_cache deprecated since 0.31)
    # Force synchronization to ensure all GPU work is complete
    mx.eval(mx.zeros(1))           # Barrier: wait for all pending GPU ops

# Required cleanup points in the generation pipeline:
#
# 1. After prompt encoding (text encoder)
# 2. After Stage 1 diffusion (before upscaler)
# 3. After Stage 2 upscaler (before VAE)
# 4. After VAE decode (before audio decode)
# 5. After audio decode (before ffmpeg encode)
# 6. After job completion (before next job)
```

#### Mandatory Solution: Periodic Model Reload

Some Metal buffers are **never freed** even with `clear_cache()` — they are tied to the model's lifetime. The proven fix used by production MLX apps:

```python
# Every N generations (e.g., every 5), fully unload and reload the model
generation_count = 0
MAX_GENERATIONS_BEFORE_RELOAD = 5

async def generate_video(params):
    global generation_count
    generation_count += 1

    if generation_count >= MAX_GENERATIONS_BEFORE_RELOAD:
        log.info("Periodic model reload to reclaim fragmented Metal memory")
        model_manager.unload_all()
        aggressive_cleanup()
        model_manager.load_model(params.model_id)
        generation_count = 0

    # ... proceed with generation
```

#### Mandatory Solution: Streaming VAE Decode

The VAE decode step is the **peak memory moment** — it transforms compressed latents into full RGB frames. Decoding all frames at once can double RAM usage temporarily.

**Never decode all frames in memory.** Stream frame-by-frame directly to ffmpeg:

```python
import subprocess

def streaming_vae_decode(latents, vae, output_path, fps=24):
    """Decode latents frame-by-frame and pipe directly to ffmpeg.
    Avoids holding all decoded frames in RAM simultaneously."""

    ffmpeg_proc = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        output_path
    ], stdin=subprocess.PIPE)

    for i in range(num_frames):
        # Decode single frame
        frame = vae.decode_frame(latents[i])
        frame_bytes = frame_to_rgb_bytes(frame)

        # Write to ffmpeg pipe
        ffmpeg_proc.stdin.write(frame_bytes)

        # Free the decoded frame immediately
        del frame, frame_bytes
        if i % 8 == 0:  # Cleanup every 8 frames
            aggressive_cleanup()

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    aggressive_cleanup()
```

#### Complete Generation Pipeline with Cleanup and Optimizations

```
app_start
  │
  ├─ load models
  ├─ mx.compile(model.forward)         ← perf: compile forward pass
  ├─ create LatentPool(max_shape)      ← perf: pre-allocate buffers
  ├─ warmup_pipeline(pipe)             ← perf: force kernel compilation
  ├─ ★ aggressive_cleanup()
  │
  ├─ [ready for user generations — "Preparing engine..." splash ends here]
  │
job_start
  │
  ├─ load prompt enhancer (Qwen3.5-2B)
  ├─ enhance prompt
  ├─ unload prompt enhancer
  ├─ ★ aggressive_cleanup()
  │
  ├─ encode text prompt (Gemma connector, using compiled model)
  ├─ ★ aggressive_cleanup()
  │
  ├─ Stage 1: diffusion (denoise loop using compiled_forward + pooled latents)
  ├─ ★ aggressive_cleanup()
  │
  ├─ [optional] send Stage 1 preview to UI via WebSocket
  │
  ├─ Stage 2: latent upscale
  ├─ ★ aggressive_cleanup()
  │
  ├─ VAE decode: streaming frame-by-frame → ffmpeg pipe
  ├─ ★ aggressive_cleanup()
  │
  ├─ Audio decode: latent → waveform via vocoder
  ├─ ★ aggressive_cleanup()
  │
  ├─ ffmpeg: mux video + audio → final MP4
  ├─ ★ aggressive_cleanup()
  │
  ├─ [every 5 jobs] full model unload → reload → re-compile → re-warmup
  │
  └─ job_end → report memory stats to UI
```

#### Memory Monitoring (expose to UI)

```python
import mlx.core as mx

def get_memory_stats() -> dict:
    """Called by the /api/v1/system/memory endpoint.
    Frontend displays these in real-time."""
    return {
        "active_memory_gb": mx.get_active_memory() / (1024**3),
        "cache_memory_gb": mx.get_cache_memory() / (1024**3),
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "system_available_gb": os_available_memory() / (1024**3),
        "generation_count_since_reload": generation_count,
        "next_reload_in": MAX_GENERATIONS_BEFORE_RELOAD - generation_count,
    }

# Also reset peak memory counter after each generation
# to track per-generation peaks:
mx.reset_peak_memory()
```

#### Warning Signs to Monitor

In the UI memory panel, flag these conditions:

| Condition | Severity | Action |
|-----------|----------|--------|
| `cache_memory > 2× active_memory` | Warning | Force `clear_cache()` |
| `peak_memory > 85% total RAM` | Critical | Reduce resolution or force model reload |
| `active_memory` grows between jobs | Warning | Trigger model reload |
| `system_available < 4GB` | Critical | Pause queue, unload model, alert user |

#### Testing Strategy

**Do not consider the app stable until it passes this test:**

```
Test: "Marathon Generation"
1. Configure: 97 frames, 768×512, distilled model
2. Queue 10 consecutive text-to-video generations
3. Monitor: active_memory, cache_memory, peak_memory after each
4. Pass criteria:
   - No OOM crash
   - Memory after generation 10 is within 20% of memory after generation 1
   - No generation takes >2× longer than generation 1
```

This test catches fragmentation issues that unit tests completely miss.

### Prompt Enhancement Pipeline (Qwen3.5-2B)

```python
from mlx_lm import load, generate

# Load lightweight model (~1.2GB in 4-bit)
model, tokenizer = load("mlx-community/Qwen3.5-2B-4bit")

# System prompt for LTX-2.3 prompt optimization
system = """You are a prompt enhancer for AI video generation.
Rewrite the user's short description into a detailed, cinematic prompt.
Include: subject appearance, specific movements, camera angles,
lighting, environment details, audio elements. Use one flowing paragraph.
Start directly with the action. Keep under 200 words."""

enhanced = generate(model, tokenizer, prompt=user_prompt, system=system, max_tokens=300)

# Unload to free memory before video generation
del model, tokenizer
mx.clear_cache()
```

### Weight Conversion

```bash
# From PyTorch (HuggingFace) to MLX
uv run mlx_video.convert \
    --hf-path Lightricks/LTX-2.3 \
    --mlx-path ~/models/ltx23-mlx

# Or use pre-converted community model
# notapalindrome/ltx2-mlx-av
```

---

## Performance Optimizations — Up to 3× Inference Speedup

Four MLX-specific optimizations that, combined, yield **up to 3× faster inference** on Apple Silicon. The biggest single contributor is TeaCache (Optimization 4), a proven technique already validated on LTX-Video.

### Optimization 1: Kernel Warm-Up Pass (~20-30% on first generation)

MLX compiles Metal kernels **on first use for each unique shape**. On a 19B parameter model like LTX-2.3, this compilation overhead can add 20-60 seconds to the first generation. Every subsequent generation with the same shapes skips compilation entirely.

**Solution**: run a micro-generation at app startup to force kernel compilation.

```python
def warmup_pipeline(pipe):
    """Run at app startup, after model load, before first user generation.
    Forces Metal kernel compilation for all operations in the pipeline.
    Uses minimum settings to complete fast while still hitting all code paths."""
    log.info("Warming up Metal kernels (one-time)...")
    pipe.generate(
        prompt="warmup",
        num_frames=9,          # Minimum (multiple of 8 + 1)
        steps=1,               # Single step — enough to compile all kernels
        height=256,
        width=256,
        guidance_scale=1.0,
    )
    aggressive_cleanup()       # Free the warmup output immediately
    log.info("Kernel warmup complete")
```

**Why this works**: MLX JIT-compiles Metal shaders per operation + shape. The warm-up pass triggers compilation for every op in the graph (attention, FFN, conv, normalization, etc.). Once compiled, kernels are cached by the Metal system and persist **across app restarts** (Metal shader cache is system-level).

**UX impact**: without warm-up, the first generation feels broken slow. With warm-up, every generation feels consistent. Display a "Preparing engine..." splash during warm-up.

**Important nuance**: if the user's first real generation uses a different resolution than the warm-up, some kernels may need recompilation for the new shapes. The warm-up covers the common case but doesn't eliminate all JIT overhead for novel shapes.

### Optimization 2: Compiled Denoising Steps via `mx.compile()` (~10-15%)

The diffusion denoising loop dispatches a separate Metal kernel for every operation at every timestep. With 8-40 steps and a 19B model, the CPU→GPU dispatch overhead becomes significant.

**Solution**: compile the model's forward pass with `mx.compile()` to fuse operations and reduce dispatch count.

```python
import mlx.core as mx

# Compile the DiT forward pass — fuses element-wise ops into single kernels
# This reduces Metal dispatch overhead significantly
compiled_forward = mx.compile(model.forward)

# Use in the denoising loop
for t in timesteps:
    # Each call now uses fused kernels — fewer dispatches, more GPU parallelism
    noise_pred = compiled_forward(latents, t, encoder_hidden_states)
    latents = scheduler.step(noise_pred, t, latents)
```

**Why this works**: `mx.compile()` traces the computation graph and fuses element-wise operations (GELU, LayerNorm components, residual adds) into single Metal kernels. Instead of N separate GPU dispatches for N ops, you get fewer, larger kernels with better GPU utilization.

**Important nuances**:
- Compile the **model forward pass**, not the entire denoising loop. The loop has dynamic control flow (different timesteps) that `mx.compile()` handles less well.
- `mx.compile()` works best when shapes are static across calls. Since all denoising steps use the same latent shape, this is ideal.
- First call after compilation incurs a one-time tracing cost. This is covered by Optimization 1 (warm-up pass).
- Some complex operations (custom attention with variable masks) may not fuse well. Profile to verify actual gains.

### Optimization 3: Latent Buffer Reuse (reduces fragmentation, ~5-10%)

Each generation typically allocates fresh latent tensors via `mx.random.normal(shape)`. For video, these are large (hundreds of MB). Repeated allocation and deallocation of large buffers fragments Metal memory (compounds the problem from the Memory Fragmentation section).

**Solution**: pre-allocate a noise buffer at a fixed shape and reuse it across generations.

```python
class LatentPool:
    """Pre-allocated latent buffer to reduce Metal memory fragmentation.
    Reuses the same GPU memory region across generations."""

    def __init__(self, max_shape):
        # Force-allocate by evaluating — ensures Metal actually reserves the memory
        self._buffer = mx.zeros(max_shape)
        mx.eval(self._buffer)
        self._shape = max_shape

    def get_noise(self, shape, key):
        """Generate random noise reusing the pre-allocated buffer region.
        If shape fits within max_shape, avoids new allocation."""
        if all(s <= m for s, m in zip(shape, self._shape)):
            # Generate noise in a view of the existing buffer
            noise = mx.random.normal(shape, key=key)
            mx.eval(noise)  # Force materialization in Metal memory
            return noise
        else:
            # Shape exceeds pool — fall back to fresh allocation
            # This should be rare if pool is sized for max resolution
            return mx.random.normal(shape, key=key)

# Initialize once at model load time, sized for max expected resolution
# Shape: (batch, channels, frames, height//8, width//8)
latent_pool = LatentPool(max_shape=(1, 128, 33, 96, 64))  # ~1280×768, 257 frames
```

**Why this works**: MLX is lazy — `mx.zeros()` doesn't allocate until evaluated. By force-evaluating at init time, we ensure Metal reserves a contiguous buffer. Subsequent noise generation into similar shapes is more likely to reuse the same memory region rather than fragmenting the pool.

**Important nuances**:
- MLX does NOT support true in-place mutation (`buffer[:] = ...` doesn't work like PyTorch). The benefit here is about reducing allocation frequency and guiding Metal's allocator, not zero-copy reuse.
- The `mx.eval()` call is critical — without it, lazy evaluation means no actual allocation occurs.
- Size the pool for your maximum expected resolution. Undersized pools fall back to fresh allocation (no harm, just no benefit).
- This optimization has **diminishing returns if aggressive_cleanup() is already working well**. Its main value is reducing the frequency of large allocations between cleanups.

### Optimization 4: TeaCache — Block-Level Output Caching (~1.5-2× speedup)

This is the single most impactful optimization for DiT-based video models. The community has demonstrated **2.3× inference speedup** on LTX-Video with negligible quality loss. It's training-free, compatible with LoRAs, and already proven on LTX specifically.

#### Why KV-caching (as used in LLMs) does NOT work for diffusion

A common misconception is to apply LLM-style KV-caching to diffusion transformers. This **does not work** because:

1. **In LLMs**: previous tokens don't change when generating a new token → their K/V remain valid → cache and reuse.
2. **In diffusion**: the **entire latent changes at every denoising step** (that's what denoising IS) → K and V computed from x_t are invalid for x_{t-1} → caching KV across steps produces degraded output.

The correct approach caches at a different level: **entire transformer block outputs**, not individual KV matrices.

#### How TeaCache actually works

TeaCache observes that between consecutive denoising timesteps, many transformer blocks produce **nearly identical outputs**. Instead of recomputing every block at every step, TeaCache:

1. Uses the **timestep embedding difference** as a cheap proxy to estimate how much the block output will change
2. Applies polynomial rescaling to refine this estimate
3. If the estimated change is below a threshold, **reuses the cached block output from the previous step**
4. If above threshold, **recomputes and updates the cache**

```python
# Conceptual TeaCache logic (simplified)
cached_block_outputs = {}

for t in timesteps:
    for layer_idx, block in enumerate(transformer.blocks):
        # Cheap estimate: how much will this block's output change?
        estimated_diff = estimate_output_change(t, t_prev, layer_idx)

        if estimated_diff < rel_l1_thresh and layer_idx in cached_block_outputs:
            # Skip this block entirely — reuse cached output
            x = x + cached_block_outputs[layer_idx]
        else:
            # Compute normally and update cache
            block_output = block(x, t)
            cached_block_outputs[layer_idx] = block_output
            x = x + block_output
```

The key insight: this is **not** caching KV inside attention. It's caching the **entire block output** (attention + MLP + residual), skipping the whole block computation when the output would be nearly identical.

#### Proven results on LTX-Video

| `rel_l1_thresh` | Speedup | Quality Impact |
|-----------------|---------|----------------|
| 0 (disabled) | 1.0× | baseline |
| 0.03 | **1.6×** | near-lossless |
| 0.05 | **2.1×** | minimal degradation |
| 0.08+ | 2.5×+ | visible quality loss |

*Source: ali-vilab/TeaCache, tested on LTX-Video specifically.*

**Recommended setting**: `rel_l1_thresh=0.03` for production quality, `rel_l1_thresh=0.05` for rapid iteration/preview.

#### Implementation for MLX

TeaCache needs to be ported to MLX (the reference implementation is PyTorch/CUDA). The core logic is simple:

```python
import mlx.core as mx

class TeaCacheMLX:
    """Training-free block-output caching for DiT models on MLX."""

    def __init__(self, rel_l1_thresh=0.03):
        self.thresh = rel_l1_thresh
        self.cache = {}           # layer_idx → cached output
        self.prev_t_emb = None    # previous timestep embedding

    def should_recompute(self, t_emb, layer_idx):
        """Estimate if block output changed enough to warrant recomputation."""
        if self.prev_t_emb is None or layer_idx not in self.cache:
            return True
        # Cheap L1 difference on timestep embeddings (rescaled)
        diff = mx.mean(mx.abs(t_emb - self.prev_t_emb)).item()
        return diff > self.thresh

    def get_or_compute(self, block, x, t_emb, layer_idx):
        if self.should_recompute(t_emb, layer_idx):
            output = block(x, t_emb)
            self.cache[layer_idx] = output
            return output
        else:
            return self.cache[layer_idx]

    def step_done(self, t_emb):
        """Call after each denoising step."""
        self.prev_t_emb = t_emb
```

**MLX advantage**: cached block outputs stay in unified memory with zero-copy. No CPU↔GPU transfer overhead like CUDA implementations.

#### References
- **TeaCache paper**: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model" (ali-vilab)
- **TeaCache for LTX-Video**: https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video
- **EasyCache** (improved version): https://github.com/H-EmbodVis/EasyCache — runtime-adaptive, up to 3.3× with SVG
- **ComfyUI integration**: https://github.com/welltop-cn/ComfyUI-TeaCache
- **License**: Apache 2.0

### Combined Pipeline Architecture

```
app_start
  │
  ├─ load models
  ├─ mx.compile(model.forward)        ← Optimization 2: compile forward pass
  ├─ create LatentPool(max_shape)      ← Optimization 3: pre-allocate buffers
  ├─ create TeaCacheMLX(thresh=0.03)   ← Optimization 4: block-output caching
  ├─ warmup_pipeline(pipe)             ← Optimization 1: force kernel compilation
  ├─ aggressive_cleanup()
  │
  ├─ [ready for user generations]
  │
  ├─ generation N:
  │   ├─ latents = latent_pool.get_noise(shape, key)
  │   ├─ denoise loop (compiled_forward + TeaCache skipping ~40-60% of blocks)
  │   ├─ ★ aggressive_cleanup() + tea_cache.clear()
  │   ├─ upscale
  │   ├─ ★ aggressive_cleanup()
  │   ├─ streaming VAE decode → ffmpeg pipe
  │   ├─ ★ aggressive_cleanup()
  │   └─ done
  │
  ├─ [every 5 generations: full model reload, re-compile, re-warmup]
  │
  └─ app_exit
```

### Benchmarking Expectations

| Hardware | Baseline | Opt 1-3 (compile+warmup+pool) | Opt 1-4 (+TeaCache) | Notes |
|----------|----------|-------------------------------|----------------------|-------|
| M2 Max 32GB | ~100s | ~65-75s | **~35-45s** | Distilled, 768×512, 97 frames |
| M3 Max 64GB | ~70s | ~45-55s | **~25-32s** | Distilled, 1280×704, 97 frames |
| M4 Ultra 128GB | ~40s | ~25-32s | **~15-20s** | Full model, 1080p, 97 frames |

*TeaCache at `rel_l1_thresh=0.03` (near-lossless). Times are approximate. Always profile on target hardware.*

### What Does NOT Help (avoid these)

- **KV-caching across diffusion steps** (LLM-style): the entire latent changes at every step, so K/V from the previous step are invalid. This produces visible quality degradation. Use TeaCache (block-level caching) instead.
- **Trying to batch multiple videos**: LTX-2.3 at 19B params leaves no room for batch>1 on consumer hardware.
- **Lower precision than int4**: quality degrades significantly below 4-bit for DiT models.
- **Reducing guidance scale to 1.0**: saves negligible compute (LTX distilled already doesn't use CFG).
- **`mx.set_memory_limit()`** (formerly `mx.metal.set_memory_limit()`): this caps MLX's allocation but doesn't prevent Metal's own caching. It causes premature OOM rather than solving fragmentation.
- **Temporal sliding window on the diffusion loop**: LTX-2.3's DiT uses global temporal attention. Windowing mid-diffusion breaks the attention pattern the model was trained with. For longer videos, use overlapping segment generation (ExtendPipeline) instead.

---

## LTX-2.3 Prompting Guide

Optimal prompts follow this structure (integrate into the UI as helper text):

1. **Main subject**: detailed description (appearance, clothing, expressions)
2. **Action**: specific, chronological, literal movement
3. **Environment**: location, lighting, atmosphere
4. **Camera**: angle, movement (pan, dolly, static, tracking)
5. **Style**: cinematic, realistic, animation, etc.
6. **Audio** (optional): dialogue in quotes, ambient sounds, music

Example:
> A middle-aged woman with curly red hair wearing a green wool coat walks through a misty forest path. She looks up and smiles as sunlight breaks through the canopy. Leaves fall gently around her. The camera slowly dollies forward, tracking her movement. Soft orchestral music plays. Birds chirp in the distance. Cinematic, warm color grading, shallow depth of field.

2.3 improvements:
- Handles complex prompts with multiple subjects much better
- Spatial relationships are more accurately respected
- Stylistic instructions are more faithfully followed
- Native portrait — just set the resolution, no hacks needed

---

## Development Workflow

### Initial Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd ltx-desktop-macos

# 2. Install system dependencies
brew install ffmpeg uv python@3.12

# 3. Setup Python backend
cd backend
uv sync
source .venv/bin/activate

# 4. Download and convert LTX-2.3 model (~43GB download, converts to MLX split + int8)
uv run python ../scripts/convert_ltx23.py --quantize --bits 8

# 5. Download prompt enhancer (~1.2GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3.5-2B-4bit')"

# 6. Start the backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 7. Open the Xcode project and run the SwiftUI app
open app/LTXDesktop.xcodeproj
```

### Current Status

All foundation sprints are complete. The app builds, launches, and performs real MLX inference (T2V, I2V, preview, prompt enhancement with quantized int8 model on 32GB machine).

LTX-2.3 (22B) migration is complete: vendored model architecture in `engine/ltx23_model/` (transformer, VAE encoder/decoder, audio VAE decoder, HiFi-GAN vocoder with BWE, text encoder, conversion script). End-to-end T2V and I2V verified working with int8 quantized model.

**Remaining items**:
- 97-frame marathon test at full resolution
- Progressive diffusion display (intermediate frames during generation)
- History view connection to real backend data
- Real TTS via MLX-Audio (currently sine-wave stub)
- Model download UI in Settings
- Batch generation queue

---

## Code Conventions

### Python (backend)
- Python 3.12+
- Package manager: `uv`
- Formatter/Linter: `ruff`
- Mandatory type hints
- Async for all FastAPI endpoints
- Google-style docstrings
- **Start simple**: single `main.py`, extract modules when needed

### Swift (frontend)
- SwiftUI, not UIKit
- MVVM architecture
- Swift 5.9+, macOS 14.0+ (Sonoma minimum for MLX)
- Combine for reactive data flow
- Async/await for backend communication

### General
- Conventional commits (feat:, fix:, docs:, refactor:)
- Branches: main, develop, feature/*
- PRs required for merging into main

---

## Resources and Links

### Core
- **LTX-2.3 Blog**: https://ltx.io/model/model-blog/ltx-2-3-release
- **LTX-2 GitHub**: https://github.com/Lightricks/LTX-2
- **LTX Desktop GitHub**: https://github.com/Lightricks/ltx-desktop
- **LTX-2.3 HuggingFace**: https://huggingface.co/Lightricks/LTX-2.3
- **LTX Prompting Guide**: https://ltx.video/blog/how-to-prompt-for-ltx-2
- **LTX Model License**: https://ltx.io/model/license

### MLX Ecosystem
- **MLX GitHub**: https://github.com/ml-explore/mlx
- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **mlx-lm**: https://github.com/ml-explore/mlx-lm
- **mlx-video-with-audio**: https://pypi.org/project/mlx-video-with-audio/
- **MLX-Video (Blaizzy)**: https://github.com/Blaizzy/mlx-video
- **mlx-audio**: https://github.com/Blaizzy/mlx-audio
- **WWDC25 MLX Session**: https://developer.apple.com/videos/play/wwdc2025/315/
- **Apple M5 MLX benchmarks**: https://machinelearning.apple.com/research/exploring-llms-mlx-m5

### Models
- **Pre-converted MLX model**: https://huggingface.co/notapalindrome/ltx2-mlx-av
- **Qwen3.5 GitHub**: https://github.com/QwenLM/Qwen3.5
- **Qwen3.5-2B HuggingFace**: https://huggingface.co/Qwen/Qwen3.5-2B
- **MLX Community (quantized models)**: https://huggingface.co/mlx-community

### Performance / Caching
- **TeaCache (LTX-Video specific)**: https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video
- **TeaCache paper**: https://liewfeng.github.io/TeaCache/
- **EasyCache (improved variant)**: https://github.com/H-EmbodVis/EasyCache
- **ComfyUI-TeaCache**: https://github.com/welltop-cn/ComfyUI-TeaCache

### Reference Apps
- **ltx-video-mac (SwiftUI PoC)**: https://github.com/james-see/ltx-video-mac

---

## Important Notes

1. **⚠️ #1 RISK: Metal memory fragmentation** — The app WILL crash after repeated generations if memory cleanup is not aggressive. See the "CRITICAL: Metal Memory Fragmentation" section. This is not optional. Implement `aggressive_cleanup()` at every pipeline stage boundary, streaming VAE decode, periodic model reload, and the Marathon Generation test before considering the app stable.
2. **The LTX-2.3 model weighs ~42GB** — expect a long initial download and significant storage requirements.
3. **Practical minimum RAM: 32GB** — below that, the experience will be very limited (low resolutions, few frames). Strongly recommend cloud text encoding as fallback for 16GB machines.
4. **LTX-2 LoRAs must be retrained for the 2.3 latent space** — 2.0 LoRAs are not compatible. Filter/tag LoRAs by version.
5. **mlx-video-with-audio currently only supports the distilled variant** — full dev model support is in progress.
6. **Use Qwen3.5-2B for prompt enhancement, NOT Gemma 12B** — 8× less memory, better instruction following, Apache 2.0 license, native MLX support.
7. **Always lazy-load the prompt enhancer** — never keep it in memory alongside the video model on machines < 64GB.
8. **VAE decode must stream frame-by-frame to ffmpeg** — never decode all frames into RAM simultaneously. This is the peak memory moment and the most common OOM trigger.
9. **ffmpeg must be installed separately** — `brew install ffmpeg` or bundle a static binary in the app.
10. **macOS 14.0 Sonoma minimum** for MLX — macOS 15 Sequoia recommended for recent Metal optimizations. macOS 16 (2026) will bring Metal 4 + Neural Accelerators on M5.
11. **Do NOT build a timeline editor for the MVP** — export FCPXML instead. Build the timeline only if user demand proves it.
12. **Rapid preview is a must-have for MVP** — generating a video takes minutes. A 5-second preview changes the UX fundamentally.
13. **Implement kernel warm-up + mx.compile() + TeaCache from day 1** — combined up to 3× speedup. TeaCache alone gives 1.6-2.1× on LTX-Video (proven). Show "Preparing engine..." during warm-up at app startup.
14. **LTX-2.3 is a DiT with global temporal attention** — you CANNOT window/chunk the diffusion loop mid-generation, and you CANNOT use LLM-style KV-caching across diffusion steps (the entire latent changes at each step). For longer videos, use ExtendPipeline. For faster inference, use TeaCache (block-output caching, not KV-caching). Model weights (~38GB) dominate memory, not latents (~300MB).
