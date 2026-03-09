# Agent Team — LTX Desktop macOS

## How It Works

One terminal. One lead agent. Sub-agents are spawned via `Task()` and run in the background. The lead coordinates, delegates, and integrates.

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
claude
```

Then give the lead its instruction.

---

## Agent Definitions

### 🎯 Lead (this agent)

**Role**: Project coordination, architecture decisions, cross-agent integration
**Reads**: root `CLAUDE.md`, `AGENTS.md`

Responsibilities:
- Break down work into agent-specific tasks
- Resolve cross-agent conflicts (especially the API contract between backend and frontend)
- Review and integrate work from all agents
- Follow the sprint plan below
- Run final integration checks

Does NOT write application code directly — delegates everything.

---

### ⚙️ backend-engine

**Role**: MLX inference pipelines, model management, memory management, performance
**Workspace**: `backend/engine/` and `backend/audio/`
**Reads**: `backend/CLAUDE.md`

Spawn pattern:
```
Task(agent="backend-engine", prompt="Implement memory_manager.py with aggressive_cleanup(), get_memory_stats(), and periodic_reload(). See backend/CLAUDE.md for the exact cleanup pattern.")
```

Responsibilities:
- All MLX inference code (T2V, I2V, retake, extend, upscaler, preview)
- `memory_manager.py` — aggressive_cleanup(), periodic model reload, streaming VAE decode
- `model_manager.py` — lazy loading, unloading, HuggingFace download
- `prompt_enhancer.py` — Qwen3.5-2B via mlx-lm with lazy load/unload
- `lora_manager.py` — LoRA loading and application
- `teacache.py` — TeaCache MLX port
- `mx.compile()` on model forward pass + kernel warm-up pass
- Latent pool pre-allocation
- Audio pipelines (TTS via MLX-Audio, mixing)

Key constraints:
- Every pipeline function MUST call `aggressive_cleanup()` between stages
- Streaming VAE decode to ffmpeg pipe — never all frames in memory
- Periodic model reload every 5 generations
- Qwen3.5-2B and LTX-2.3 must NEVER coexist in memory on < 64GB

---

### 🌐 backend-api

**Role**: FastAPI server, endpoints, WebSocket, queue, export
**Workspace**: `backend/main.py`, `backend/api/`, `backend/export/`, `backend/utils/`
**Reads**: `backend/CLAUDE.md`

Spawn pattern:
```
Task(agent="backend-api", prompt="Create main.py with FastAPI. Implement /api/v1/system/health, /system/info, /system/memory endpoints. See backend/CLAUDE.md.")
```

Responsibilities:
- `main.py` — FastAPI app, startup/shutdown events (warm-up, cleanup)
- All route handlers: generation, queue, models, audio, export, system
- WebSocket for real-time progress + intermediate frame streaming
- Generation queue with priority, cancellation
- Export: ffmpeg encoding, FCPXML, Premiere XML
- Config management

Key constraints:
- All generation endpoints async, delegate to engine pipelines
- WebSocket streams: progress %, step count, memory stats, preview frames
- Health endpoint reports: model loaded, memory stats, generation count

API contract (shared with frontend agent):
```
POST /api/v1/generate/text-to-video    → { job_id }
POST /api/v1/generate/image-to-video   → { job_id }
POST /api/v1/generate/preview          → { job_id }
POST /api/v1/generate/retake           → { job_id }
POST /api/v1/generate/extend           → { job_id }
GET  /api/v1/queue                     → [{ job_id, status, progress }]
POST /api/v1/queue/{id}/cancel         → { success }
WS   /ws/progress/{job_id}            → stream { step, total_steps, pct, memory, preview_frame? }
GET  /api/v1/models                    → [{ id, name, size, loaded }]
POST /api/v1/models/download           → { download_id }
GET  /api/v1/loras                     → [{ id, name, type, compatible }]
POST /api/v1/loras/load                → { success }
POST /api/v1/audio/tts                 → { audio_path }
POST /api/v1/audio/music               → { audio_path }
POST /api/v1/audio/mix                 → { output_path }
POST /api/v1/export/video              → { output_path }
POST /api/v1/export/fcpxml             → { output_path }
GET  /api/v1/system/info               → { chip, ram_total, ram_available, macos_version }
GET  /api/v1/system/health             → { status, model_loaded, generation_count }
GET  /api/v1/system/memory             → { active_gb, cache_gb, peak_gb, available_gb }
POST /api/v1/prompt/enhance            → { original, enhanced }
```

---

### 🖥️ frontend

**Role**: SwiftUI macOS app, UI/UX, backend communication
**Workspace**: `app/`
**Reads**: `app/CLAUDE.md`

Spawn pattern:
```
Task(agent="frontend", prompt="Create the Xcode project. Implement ProcessManager.swift to start/stop the Python backend, BackendService.swift for HTTP+WebSocket to localhost:8000, and a minimal GenerationView with prompt field and generate button. See app/CLAUDE.md.")
```

Responsibilities:
- All SwiftUI views (Generation, Preview, History, Settings, LoRA)
- Backend communication (HTTP + WebSocket via BackendService)
- Python backend process lifecycle (ProcessManager)
- Memory monitoring panel
- "Preparing engine..." splash during warm-up

Key constraints:
- SwiftUI only, MVVM, macOS 14.0+
- Async/await for all backend calls
- Handle backend crashes gracefully (error dialog + restart button)
- Dark mode default

---

### 🧪 qa

**Role**: Testing, scripts, CI, stability validation
**Workspace**: `tests/`, `scripts/`
**Reads**: `tests/CLAUDE.md`, `scripts/CLAUDE.md`

Spawn pattern:
```
Task(agent="qa", prompt="Create setup.sh that checks/installs python 3.12, uv, ffmpeg on macOS. Create download_models.sh. See scripts/CLAUDE.md.")
```

Responsibilities:
- `test_marathon.py` — 10 consecutive generations stability test (BLOCKS RELEASE)
- `test_memory.py`, `test_inference.py`, `test_api.py`
- `setup.sh`, `download_models.sh`, `dev.sh`

Key constraints:
- Marathon test MUST pass before any sprint advances
- All scripts idempotent
- All tests runnable headless (no SwiftUI)

---

## Sprint Status

All 5 sprints have been completed. Code is written and app builds/launches. Real MLX inference is working (T2V, I2V, preview, prompt enhancement with quantized int8 model).

LTX-2.3 (22B) migration is complete: vendored model architecture (`engine/ltx23_model/`), conversion script, VAE encoder/decoder, audio VAE decoder, HiFi-GAN vocoder with BWE. End-to-end T2V and I2V verified working.

**Remaining**: 97-frame marathon test, progressive diffusion display, history view integration, real TTS via MLX-Audio.

---

## File Ownership

```
backend/engine/     → backend-engine owns
backend/audio/      → backend-engine owns
backend/api/        → backend-api owns
backend/export/     → backend-api owns
backend/utils/      → backend-api owns
backend/main.py     → backend-api owns
app/                → frontend owns
tests/              → qa owns
scripts/            → qa owns
CLAUDE.md           → lead owns
AGENTS.md           → lead owns
pyproject.toml      → backend-api owns (backend-engine can request dep additions)
```

## Conflict Resolution

- **API contract changes**: backend-api proposes → lead approves → frontend adapts
- **Memory issues found by QA**: blocks all sprints → backend-engine fixes → QA re-tests
- **Cross-agent file edits**: never. If you need something from another agent's territory, ask the lead to coordinate.
