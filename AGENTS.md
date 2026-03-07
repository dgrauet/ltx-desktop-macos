# Agent Team — LTX Desktop macOS

## How It Works

One terminal. One lead agent. Sub-agents are spawned via `Task()` and run in the background. The lead coordinates, delegates, and integrates.

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
claude
```

Then give the lead its first instruction (see README.md).

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
Task(agent="qa", prompt="Create setup.sh that checks/installs python 3.12, uv, ffmpeg on macOS. Create download_models.sh for notapalindrome/ltx2-mlx-av and Qwen/Qwen3.5-2B. See scripts/CLAUDE.md.")
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

## Sprint Plan

### Sprint 1 (Week 1-2): Foundation

```
1. Task(agent="backend-engine", prompt="
   Implement backend/engine/memory_manager.py with:
   - aggressive_cleanup() function (gc.collect + mx.metal.clear_cache + barrier)
   - get_memory_stats() returning active/cache/peak/available
   - periodic_reload_check() that tracks generation count
   Then implement a basic T2V pipeline wrapper in backend/engine/pipelines/text_to_video.py 
   that uses mlx-video-with-audio, calls aggressive_cleanup() between stages,
   integrates mx.compile() on the model forward pass, and runs a kernel warm-up.
   Read backend/CLAUDE.md first.")

2. Task(agent="qa", prompt="
   Create scripts/setup.sh and scripts/download_models.sh.
   setup.sh: check macOS >= 14, check/install python 3.12, uv, ffmpeg via brew.
   download_models.sh: download notapalindrome/ltx2-mlx-av and Qwen/Qwen3.5-2B via huggingface-cli.
   Both must be idempotent. Read scripts/CLAUDE.md first.")

3. [After Agent 1 delivers] Task(agent="backend-api", prompt="
   Create backend/main.py with FastAPI. Implement:
   - GET /api/v1/system/health
   - GET /api/v1/system/info (detect Apple Silicon chip, RAM)
   - GET /api/v1/system/memory (call engine's get_memory_stats)
   - POST /api/v1/generate/text-to-video (call engine's T2V pipeline, return job_id)
   - WS /ws/progress/{job_id} (stream progress from the pipeline)
   Start as a single main.py file. Read backend/CLAUDE.md first.")

4. [After Agent 2 delivers] Task(agent="frontend", prompt="
   Create the Xcode project in app/LTXDesktop.xcodeproj.
   Implement:
   - ProcessManager.swift: start backend subprocess (uvicorn), poll /health until ready
   - BackendService.swift: HTTP client + WebSocket for progress
   - A minimal GenerationView: prompt TextField, generate Button, progress ProgressView
   - Show 'Preparing engine...' until health check passes
   Read app/CLAUDE.md first.")

5. [After all above] Task(agent="qa", prompt="
   Write tests/test_marathon.py: 10 consecutive T2V generations (9 frames, 256×256, 1 step 
   for speed). Verify no OOM, memory within 20% of gen 1, no gen takes >2× gen 1.
   Write tests/conftest.py with the backend fixture.
   RUN the marathon test and report results. Read tests/CLAUDE.md first.")
```

**GATE**: Marathon test must pass. Fix memory issues before Sprint 2.

### Sprint 2 (Week 3-4): Core Features

```
Parallel:
  Task(agent="backend-engine", prompt="Implement rapid preview pipeline (384×256, 4 steps, single-stage). Implement progressive diffusion display (extract intermediate frames every 4th step). Implement I2V pipeline.")
  Task(agent="backend-api", prompt="Add POST /generate/preview, POST /generate/image-to-video, and intermediate frame support on the WebSocket.")
  Task(agent="frontend", prompt="Add rapid preview button, image drag-and-drop for I2V, progressive diffusion display in PreviewView, and the MemoryMonitor panel in Settings.")

Then:
  Task(agent="backend-engine", prompt="Implement streaming VAE decode to ffmpeg pipe. Port TeaCache to MLX (see CLAUDE.md Optimization 4).")
  Task(agent="qa", prompt="Re-run marathon test with TeaCache enabled. Verify speedup > 1.3× and memory stability.")
```

### Sprint 3 (Week 5-6): Enhancement

```
Parallel:
  Task(agent="backend-engine", prompt="Implement prompt_enhancer.py with Qwen3.5-2B lazy load/unload. Implement retake and extend pipelines. Implement periodic model reload.")
  Task(agent="backend-api", prompt="Add POST /prompt/enhance, POST /generate/retake, POST /generate/extend endpoints.")
  Task(agent="frontend", prompt="Build HistoryView with video grid and thumbnails. Build SettingsView with model management. Add prompt enhance button with preview.")
```

### Sprint 4 (Week 7-8): Polish

```
Parallel:
  Task(agent="backend-engine", prompt="Implement lora_manager.py. Implement audio pipelines (TTS via MLX-Audio, music mixing).")
  Task(agent="backend-api", prompt="Add LoRA endpoints, audio endpoints, export endpoints (ffmpeg + FCPXML).")
  Task(agent="frontend", prompt="Build LoRAView. Build export UI. Full polish pass.")
```

### Sprint 5 (Week 9-10): Stability & Ship

```
  Task(agent="qa", prompt="Run full test suite. Run marathon test on target hardware configs. Report all failures.")
  [Fix all failures]
  Lead writes README.md and prepares release.
```

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
