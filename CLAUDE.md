# CLAUDE.md — LTX Desktop macOS (Apple Silicon + MLX Local Inference)

## Project Vision

Native macOS app replicating **LTX Desktop** (Lightricks) with **100% local inference on Apple Silicon via MLX** instead of NVIDIA/CUDA or cloud API. No existing app combines AI video generation + editing locally on Apple Silicon.

---

## Technical Context

### LTX-2.3 — The Model
- **Architecture**: Diffusion Transformer (DiT) 19B params, synchronized audio+video in single pass
- **Model capabilities**: T2V, I2V, V2V, A2V, keyframe interpolation, retake, extend (app only implements T2V + I2V so far)
- **Tested resolutions**: up to 1280×704 on 32GB. 1920×1080 selectable in UI but untested on 32GB (likely OOMs)
- **FPS**: 24 (distilled model)
- **Audio**: built-in vocoder, synchronized generation (output noisy — quality issue)
- **Variants**: `ltx-2.3-dev` (full bf16 ~42GB), `ltx-2.3-distilled` (8+4 steps), distilled-fp8. Only distilled supported by mlx-video-with-audio.
- **Text encoder**: Gemma 3 12B (for video generation, NOT prompt enhancement)
- **VAE**: rebuilt for 2.3, better texture preservation
- **HuggingFace**: `Lightricks/LTX-2.3`, pre-converted MLX: `dgrauet/ltx-2.3-mlx-distilled-q8`

### MLX on Apple Silicon
- Unified CPU/GPU memory — no data copying
- Key packages: `mlx`, `mlx-video-with-audio`, `mlx-lm`, `mlx-audio`
- Weight conversion: PyTorch → MLX via [mlx-forge](https://github.com/dgrauet/mlx-forge) (`mlx-forge convert ltx-2.3`)
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
- ✅ Model management (auto download, variant selection, delete, cache in `~/.cache/huggingface/`)
- ✅ Model download UI in Settings (download/delete/progress tracking)
- ✅ Audio decode pipeline (VAE → vocoder → WAV → mux MP4) — `--generate-audio` for v2.3
- ✅ Export MP4 via ffmpeg (H.264/H.265/ProRes, CRF 18)
- ✅ FCPXML export for Final Cut Pro / DaVinci Resolve
- ✅ 2× pixel upscale via ffmpeg lanczos
- ✅ Marathon stability test PASS (10 gens, 97f@768×512, no OOM, stable timing)
- ✅ Generation UX stage labels (Loading model / Generating / Decoding / Saving via WebSocket)
- ✅ Progressive diffusion display (preview frame every 2 steps via WebSocket)
- ✅ History view with JSON persistence (GET/DELETE endpoints)
- ✅ Memory warning thresholds in Settings UI (cache, peak, available)

**REMAINING:**
- Audio quality — pipeline works but vocoder output noisy; needs investigation
- Generation performance (~8min for 97f@768×512) — TeaCache/mx.compile both disabled (see Performance section)
- Video retake & extend — endpoints exist but produce stub output (solid-color clips with sleep), need real inference
- LoRA support — endpoint infrastructure exists but untested end-to-end with real LoRA weights
- Local TTS voiceover via MLX-Audio (Kokoro, Dia, CSM) — currently sine-wave stub
- Background music generation — currently sine-wave stub
- Batch generation queue — queue logic exists, UI not connected
- Parameter preset saving/loading
- Hardware enforcement — limit resolution/frames based on detected RAM (currently all resolutions selectable)
- RAM < 32GB warning banner
- Automated memory actions (pause queue, unload model on pressure) — currently warnings only

### Phase 2 — Advanced Workflows (if product finds traction)
- Simple timeline (2 video + 5 audio tracks, trim/split/reorder)
- V2V, A2V, keyframe interpolation, multi-keyframe conditioning
- Latent cache system (save latents to disk for instant retake/extend)
- Depth & edge conditioning (control adapters)
- Intelligent model management (lazy load, auto-unload on idle, memory pressure detection)
- Chunked generation for long videos (overlapping segments via ExtendPipeline with alpha blending)

---

## Backend API Endpoints

### Working (real inference / real logic)
```
POST /api/v1/generate/text-to-video     POST /api/v1/generate/image-to-video
POST /api/v1/generate/preview            WS   /ws/progress/{job_id}
GET  /api/v1/queue                       GET  /api/v1/queue/{job_id}
POST /api/v1/queue/{job_id}/cancel       POST /api/v1/queue/{job_id}/priority
GET  /api/v1/models                      POST /api/v1/models/download
POST /api/v1/models/select               GET  /api/v1/models/{download_id}/status
DELETE /api/v1/models/{model_id}         POST /api/v1/prompt/enhance
POST /api/v1/export/video                POST /api/v1/export/fcpxml
POST /api/v1/audio/mix                   GET  /api/v1/system/health
GET  /api/v1/system/memory               GET  /api/v1/history
DELETE /api/v1/history/{job_id}
```

### Stubs (API exists, fake inference / placeholder output)
```
POST /api/v1/generate/retake             POST /api/v1/generate/extend
POST /api/v1/audio/tts                   POST /api/v1/audio/music
```

### Untested (code exists, no end-to-end verification with real LoRAs)
```
GET  /api/v1/loras                       POST /api/v1/loras/load
POST /api/v1/loras/unload/{lora_id}      PUT  /api/v1/loras/{lora_id}/strength
POST /api/v1/loras/import
```

---

## ⚠️ CRITICAL: Metal Memory Management

**#1 stability risk.** Without aggressive cleanup, Metal cache grows unbounded across generations → OOM after 2-4 runs.

### Rules (mandatory)
1. Call `aggressive_cleanup()` (gc.collect + mx.clear_cache + mx.eval barrier) between **every** pipeline stage
2. **Stream VAE decode** frame-by-frame to ffmpeg pipe — never decode all frames in RAM
3. **Periodic model reload** every 5 generations (auto-triggered via generation counter in memory_manager.py)
4. **Monitor** via `/api/v1/system/memory`: active, cache, peak memory + system available
5. See `engine/memory_manager.py` and `engine/ltx23_model/vae_decoder.py` (streaming decode) for implementation

### Warning thresholds
UI indicators in Settings (passive warnings only — no automated actions yet):
| Condition | Severity | UI indicator |
|-----------|----------|--------------|
| cache > 2× active | Warning | Yellow |
| peak > 85% total RAM | Critical | Red |
| system available < 4GB | Critical | Red |

**Not implemented:** automated pause queue, auto-unload model, proactive alerts. Warnings are display-only.

### Marathon Test (stability gate)
10 consecutive T2V at 97 frames 768×512. Pass: no OOM, memory gen10 within 20% of gen1, no gen >2× slower than gen1.

---

## Performance Optimizations

### Current Status — ALL DISABLED
With the subprocess-per-generation architecture, none of these optimizations are active:
- **TeaCache** — implemented (`engine/teacache.py`) but 0% cache hit rate with 8-step distilled model. Large sigma jumps between steps cause features to change too much. Designed for 20-50 step models.
- **mx.compile()** — tracing overhead (~2min) paid every subprocess invocation, compiled kernels lost on exit. Net negative.
- **Kernel warm-up** — not implemented. Would help first gen but requires persistent model server.
- **Latent buffer reuse** — not implemented.

To enable these: would need persistent model server (keep model loaded across gens). Dropped — incompatible with 32GB unload/reload pattern.

### What Does NOT Help (avoid)
- KV-caching across diffusion steps (latent changes every step → invalid K/V)
- Batching (model too large for batch>1 on consumer hardware)
- Precision below int4 (quality degrades for DiT)
- `mx.set_memory_limit()` (causes premature OOM, doesn't fix fragmentation)
- Temporal sliding window on diffusion loop (breaks DiT global temporal attention)

---

## Hardware

- **Minimum**: 32GB RAM. Below that, very limited (low res, few frames)
- **Model weights**: ~21GB int8 (dominant cost, 85% of memory). Latents only ~300MB.
- **Budget (32GB)**: OS ~4GB + model ~21GB + buffers ~3GB + Metal cache ~2-4GB + enhancer ~1.2GB (when active)
- Backend detects chip/RAM via `/api/v1/system/info` but **no enforcement** — all resolutions remain selectable regardless of RAM
- **Not implemented:** RAM < 32GB warning banner, auto-limit resolution/frames based on hardware

---

## MLX Pipeline Details

### Two-Stage Pipeline
Stage 1: low-res generation (768×512) with distilled model (8 steps) → Stage 2: 2× spatial upscale (4 steps) → VAE decode → audio decode → ffmpeg mux. See `engine/generate_v23.py`.

### Rapid Preview
384×256, 4 steps, single-stage. Seconds. User validates → launches full render.

### Progressive Diffusion Display
Every 2 steps, decode middle temporal frame → JPEG → temp file → base64 → WebSocket. ~800ms total overhead per 8-step gen. Enabled for T2V/I2V, disabled for rapid preview.

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
- **Models**: [MLX LTX-2.3](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q8) · [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)
- **LTX Desktop (reference)**: [GitHub](https://github.com/Lightricks/ltx-desktop)

---

## Important Reminders

1. **Metal memory fragmentation is the #1 risk** — aggressive_cleanup() at every stage, streaming VAE, periodic model reload
2. **Always lazy-load prompt enhancer** — never alongside video model on <64GB
3. **VAE decode must stream to ffmpeg** — never all frames in RAM (peak OOM trigger)
4. **ffmpeg required** — `brew install ffmpeg` or bundle static binary
5. **Only distilled variant supported** — mlx-video-with-audio doesn't support full dev model
6. **LTX-2.0 LoRAs incompatible with 2.3** — different latent space, must retrain
7. **DiT has global temporal attention** — cannot window diffusion loop or use LLM-style KV-caching
8. **1920×1080 untested on 32GB** — max verified resolution is 1280×704
