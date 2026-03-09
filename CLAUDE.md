# CLAUDE.md — LTX Desktop macOS (Apple Silicon + MLX Local Inference)

## Project Vision

Native macOS app replicating **LTX Desktop** (Lightricks) with **100% local inference on Apple Silicon via MLX** instead of NVIDIA/CUDA or cloud API. No existing app combines AI video generation + editing locally on Apple Silicon.

---

## Technical Context

### LTX-2.3 — The Model
- **Architecture**: Diffusion Transformer (DiT) 19B params, synchronized audio+video in single pass
- **Capabilities**: T2V, I2V, V2V, A2V, keyframe interpolation, retake, extend
- **Resolutions**: up to 1920×1080 / 1080×1920 native, 16:9, 9:16
- **FPS**: 24-30 (up to 50 full model)
- **Audio**: built-in vocoder, synchronized generation
- **Variants**: `ltx-2.3-dev` (full bf16 ~42GB), `ltx-2.3-distilled` (8+4 steps), distilled-fp8
- **Text encoder**: Gemma 3 12B (for video generation, NOT prompt enhancement)
- **VAE**: rebuilt for 2.3, better texture preservation
- **HuggingFace**: `Lightricks/LTX-2.3`, pre-converted MLX: `notapalindrome/ltx2-mlx-av`

### MLX on Apple Silicon
- Unified CPU/GPU memory — no data copying
- Key packages: `mlx`, `mlx-video-with-audio`, `mlx-lm`, `mlx-audio`
- Weight conversion: PyTorch → MLX via `scripts/convert_ltx23.py`
- Quantization: int4, int8 support

---

## Application Architecture

```
SwiftUI → HTTP/WS :8000 → FastAPI → MLX Engine + ffmpeg + MLX-Audio → Apple Silicon Metal
Two-subprocess: (A) Gemma text encoder → exits → (B) Transformer+VAE generation
```

FastAPI in separate process for: crash isolation (OOM kills backend, not UI), GIL avoidance, WebSocket progress streaming, independent restart. See `engine/mlx_runner.py`.

### Prompt Enhancement — Qwen3.5-2B
- Use `mlx-community/Qwen3.5-2B-4bit` (~1.2GB) via mlx-lm. Apache 2.0 license.
- **Never coload with video model on <64GB RAM.** Lazy load → enhance → unload → `mx.clear_cache()` → load video model.
- See `engine/prompt_enhancer.py` for implementation.

---

## Features

### Phase 1 — AI Video Generator (MVP)

**DONE:**
- ✅ T2V with configurable resolution/frames/FPS/guidance/seed/negative prompt
- ✅ Synchronized audio generation in single pass
- ✅ Rapid Preview (384×256, 4 steps, seconds)
- ✅ I2V with reference image (drag & drop), image_strength param
- ✅ Prompt Enhancement (Qwen3.5-2B, lazy load/unload)
- ✅ Model management (auto download, distilled/full choice, cache in `~/.cache/huggingface/`)
- ✅ Audio decode pipeline (VAE → vocoder → WAV → mux MP4)
- ✅ Export MP4 via ffmpeg
- ✅ 2× pixel upscale via ffmpeg lanczos (latent-space upscale OOMs on 32GB at VAE decode)
- ✅ Marathon stability test PASS (10 gens, 97f@768×512, no OOM, stable timing)
- ✅ Audio generation (VAE → vocoder → WAV → mux MP4) — `--generate-audio` for v2.3
- ✅ Generation UX stage labels (Loading model / Generating / Decoding / Saving via WebSocket)

**REMAINING:**
- Audio quality — pipeline works (WAV → mux) but audio may sound noisy; investigate vocoder output quality
- Marathon test memory reporting — `mx.get_active_memory()` returns 0 in parent process; capture from subprocesses
- Generation performance (~8min for 97f@768×512) — TeaCache: 0% hit rate with 8-step distilled model (DISABLED); mx.compile: tracing overhead wasted in subprocess-per-gen arch (DISABLED). Next: persistent model server architecture to enable both, or kernel warm-up
- Progressive diffusion display (intermediate frames via WebSocket every N steps)
- Video retake (regenerate time segment) & extension (forward/backward)
- LoRA support (camera control, detail enhancement, custom .safetensors) — LoRAs must be 2.3-compatible
- Local TTS voiceover via MLX-Audio (Kokoro, Dia, CSM) — currently stub
- Background music generation (genre presets)
- History view connected to real backend data
- Model download UI in Settings
- Batch generation queue with priority management
- Parameter preset saving/loading
- FCPXML export for Final Cut Pro / Premiere Pro / DaVinci Resolve

### Phase 2 — Advanced Workflows (if product finds traction)
- Simple timeline (2 video + 5 audio tracks, trim/split/reorder)
- V2V, A2V, keyframe interpolation, multi-keyframe conditioning
- Latent cache system (save latents to disk for instant retake/extend)
- Depth & edge conditioning (control adapters)
- Intelligent model management (lazy load, auto-unload on idle, memory pressure detection)
- Chunked generation for long videos (overlapping segments via ExtendPipeline with alpha blending)

---

## Backend API Endpoints

```
POST /api/v1/generate/text-to-video     POST /api/v1/generate/image-to-video
POST /api/v1/generate/preview            POST /api/v1/generate/retake
POST /api/v1/generate/extend             GET  /api/v1/queue
POST /api/v1/queue/{job_id}/cancel       WS   /ws/progress/{job_id}
GET  /api/v1/models                      POST /api/v1/models/download
GET  /api/v1/loras                       POST /api/v1/loras/load
POST /api/v1/audio/tts                   POST /api/v1/audio/music
POST /api/v1/audio/mix                   POST /api/v1/export/video
POST /api/v1/export/fcpxml               POST /api/v1/export/premiere-xml
GET  /api/v1/system/info                 GET  /api/v1/system/health
GET  /api/v1/system/memory               POST /api/v1/prompt/enhance
```

---

## ⚠️ CRITICAL: Metal Memory Management

**#1 stability risk.** Without aggressive cleanup, Metal cache grows unbounded across generations → OOM after 2-4 runs.

### Rules (mandatory)
1. Call `aggressive_cleanup()` (gc.collect + mx.clear_cache + mx.eval barrier) between **every** pipeline stage
2. **Stream VAE decode** frame-by-frame to ffmpeg pipe — never decode all frames in RAM
3. **Periodic model reload** every 5 generations to reclaim fragmented Metal buffers
4. **Monitor** via `/api/v1/system/memory`: active, cache, peak memory + system available
5. See `engine/memory_manager.py` and `engine/streaming_vae.py` for implementation

### Warning thresholds (show in UI)
| Condition | Severity | Action |
|-----------|----------|--------|
| cache > 2× active | Warning | Force clear_cache() |
| peak > 85% total RAM | Critical | Reduce resolution or model reload |
| active grows between jobs | Warning | Trigger model reload |
| system available < 4GB | Critical | Pause queue, unload model, alert |

### Marathon Test (stability gate)
10 consecutive T2V at 97 frames 768×512. Pass: no OOM, memory gen10 within 20% of gen1, no gen >2× slower than gen1.

---

## Performance Optimizations (up to 3× combined)

### 1. Kernel Warm-Up (~20-30% first gen)
Run micro-generation at startup to force Metal kernel compilation. Show "Preparing engine..." splash. See `warmup_pipeline()`.

### 2. Compiled Denoising via `mx.compile()` (~10-15%)
Compile model forward pass to fuse element-wise ops into fewer Metal kernels. Compile the forward pass, not the loop.

### 3. Latent Buffer Reuse (~5-10%)
Pre-allocate noise buffer at max shape, reuse across generations. Reduces Metal fragmentation.

### 4. TeaCache — Block-Level Output Caching (~1.5-2×)
Cache entire transformer block outputs between consecutive timesteps when output change is below threshold. **Not** KV-caching (which does NOT work for diffusion — latent changes every step).
- `rel_l1_thresh=0.03` for production, `0.05` for preview
- See `engine/teacache.py` for MLX implementation

### Combined Pipeline Flow
```
app_start → load models → mx.compile() → LatentPool → TeaCache → warmup → cleanup → ready
job: enhance prompt → cleanup → encode text → cleanup → Stage 1 diffusion → cleanup
   → Stage 2 upscale → cleanup → streaming VAE decode → cleanup → audio decode → cleanup
   → ffmpeg mux → cleanup → [every 5 jobs: full model reload + re-compile + re-warmup]
```

### What Does NOT Help (avoid)
- KV-caching across diffusion steps (latent changes every step → invalid K/V)
- Batching (model too large for batch>1 on consumer hardware)
- Precision below int4 (quality degrades for DiT)
- `mx.set_memory_limit()` (causes premature OOM, doesn't fix fragmentation)
- Temporal sliding window on diffusion loop (breaks DiT global temporal attention)

---

## Hardware

- **Minimum**: 32GB RAM. Below that, very limited (low res, few frames, recommend cloud text encoding fallback)
- **Model weights**: ~21GB int8 (dominant cost, 85% of memory). Latents only ~300MB.
- **Budget (32GB)**: OS ~4GB + model ~21GB + buffers ~3GB + Metal cache ~2-4GB + enhancer ~1.2GB (when active)
- Auto-detect chip/RAM on launch, limit resolution/frames accordingly
- Show warning banner if RAM < 32GB

---

## MLX Pipeline Details

### Two-Stage Pipeline
Stage 1: low-res generation (768×512) with distilled model (8 steps) → Stage 2: 2× spatial upscale (4 steps) → VAE decode → audio decode → ffmpeg mux. See `engine/generate_v23.py`.

### Rapid Preview
384×256, 4 steps, single-stage. Seconds. User validates → launches full render.

### Progressive Diffusion Display (TODO)
Every Nth step, partial VAE decode → JPEG → WebSocket to frontend. ~200ms per decode.

### Two-Subprocess Architecture
Subprocess A: Gemma 3 12B 4-bit text encoding (~8.7GB peak) → exits, frees GPU. Subprocess B: transformer + VAE generation (~12.6GB peak). Embeddings passed via npz file. See `engine/mlx_runner.py`.

---

## LTX-2.3 Prompting Guide

Structure: (1) subject appearance, (2) specific action/movement, (3) environment/lighting, (4) camera angle/movement, (5) style, (6) audio elements. One flowing paragraph, ≤200 words.

Example:
> A middle-aged woman with curly red hair wearing a green wool coat walks through a misty forest path. She looks up and smiles as sunlight breaks through the canopy. Leaves fall gently around her. Camera slowly dollies forward. Soft orchestral music. Birds chirp. Cinematic, warm color grading, shallow depth of field.

---

## Code Conventions

### Python (backend)
- Python 3.12+, `uv` package manager, `ruff` formatter/linter
- Mandatory type hints, async FastAPI endpoints, Google-style docstrings
- Dependencies: see `backend/pyproject.toml`

### Swift (frontend)
- SwiftUI (not UIKit), MVVM, Swift 5.9+, macOS 14.0+ (Sonoma min)
- Combine for reactive flow, async/await for backend communication

### General
- Conventional commits (feat:, fix:, docs:, refactor:)
- Branches: main, develop, feature/*. PRs required for main.

---

## Key Resources

- **LTX-2.3**: [Blog](https://ltx.io/model/model-blog/ltx-2-3-release) · [GitHub](https://github.com/Lightricks/LTX-2) · [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3) · [Prompting](https://ltx.video/blog/how-to-prompt-for-ltx-2)
- **MLX**: [GitHub](https://github.com/ml-explore/mlx) · [Docs](https://ml-explore.github.io/mlx/) · [mlx-lm](https://github.com/ml-explore/mlx-lm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)
- **TeaCache**: [LTX-specific](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video) · [Paper](https://liewfeng.github.io/TeaCache/)
- **Models**: [MLX LTX-2](https://huggingface.co/notapalindrome/ltx2-mlx-av) · [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)
- **LTX Desktop (reference)**: [GitHub](https://github.com/Lightricks/ltx-desktop)

---

## Important Reminders

1. **Metal memory fragmentation is the #1 risk** — aggressive_cleanup() at every stage, streaming VAE, periodic model reload, marathon test before stable
2. **LTX-2.3 model is ~42GB** — long initial download, significant storage
3. **LTX-2.0 LoRAs incompatible with 2.3** — different latent space, must retrain
4. **mlx-video-with-audio only supports distilled variant** — full dev model support WIP
5. **Always lazy-load prompt enhancer** — never alongside video model on <64GB
6. **VAE decode must stream to ffmpeg** — never all frames in RAM (peak OOM trigger)
7. **ffmpeg required** — `brew install ffmpeg` or bundle static binary
8. **macOS 14 Sonoma minimum** — macOS 15+ recommended for Metal optimizations
9. **No timeline for MVP** — export FCPXML instead. Timeline only if user demand proves it
10. **Rapid preview is must-have** — minutes for full gen, seconds for preview changes UX fundamentally
11. **DiT has global temporal attention** — cannot window diffusion loop or use LLM-style KV-caching. Use ExtendPipeline for longer videos, TeaCache for speed.
