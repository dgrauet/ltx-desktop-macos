# Frontend (SwiftUI) — CLAUDE.md

This is the native macOS SwiftUI application for LTX Desktop. It communicates with the Python backend via HTTP and WebSocket on `localhost:8000`.

**Agent 3** owns this entire directory.

---

## Tech Stack

- SwiftUI (not UIKit)
- Swift 5.9+
- macOS 14.0+ (Sonoma) deployment target
- MVVM architecture
- Combine for reactive data flow
- Async/await for backend communication
- No external Swift packages (keep it lean for v1)

## Architecture

```
┌─────────────────────────────────────────┐
│              SwiftUI Views              │
│  (GenerationView, PreviewView, etc.)    │
└──────────────┬──────────────────────────┘
               │ @Published properties
               ▼
┌─────────────────────────────────────────┐
│            ViewModels (MVVM)            │
│  (GenerationVM, HistoryVM, SettingsVM)  │
└──────────────┬──────────────────────────┘
               │ async calls
               ▼
┌─────────────────────────────────────────┐
│             Services Layer              │
│  (BackendService, ProcessManager)       │
└──────────────┬──────────────────────────┘
               │ HTTP + WebSocket
               ▼
         localhost:8000
      (Python FastAPI backend)
```

## API Contract

The backend exposes these endpoints. Use them via `BackendService.swift`:

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
POST /api/v1/export/video              → { output_path }
POST /api/v1/export/fcpxml             → { output_path }
GET  /api/v1/system/info               → { chip, ram_total, ram_available, macos_version }
GET  /api/v1/system/health             → { status, model_loaded, generation_count }
GET  /api/v1/system/memory             → { active_gb, cache_gb, peak_gb, available_gb }
POST /api/v1/prompt/enhance            → { original, enhanced }
```

## Key Behaviors

### Backend Process Lifecycle

`ProcessManager.swift` must:
1. On app launch: start the Python backend as a subprocess (`uvicorn main:app --port 8000`)
2. Poll `/api/v1/system/health` until the backend responds
3. Show "Preparing engine..." splash during kernel warm-up (poll until `model_loaded: true`)
4. On app quit: send SIGTERM to the subprocess, wait 5s, SIGKILL if needed
5. On backend crash: show error dialog with "Restart Backend" button
6. Store the backend PID for cleanup

### Memory Monitoring

`MemoryMonitor.swift` must:
- Poll `/api/v1/system/memory` every 2 seconds during generation, every 10s idle
- Display in the settings panel or a floating indicator:
  - Active memory (GB)
  - Cache memory (GB)
  - Peak memory (GB)
  - System available (GB)
  - Generations since last model reload

**Warning thresholds** (show colored indicators):
| Condition | Color | Message |
|-----------|-------|---------|
| cache > 2× active | Yellow | "High cache, cleanup recommended" |
| peak > 85% total RAM | Red | "Memory critical" |
| available < 4GB | Red | "Low memory — reduce resolution" |

### Generation Flow (UX)

```
User writes prompt
  → [optional] tap "Enhance" → show enhanced prompt (editable)
  → tap "Generate"
  → show progress bar + memory stats
  → [if rapid preview enabled] show 384×256 preview after ~5s
  → [during generation] show progressive diffusion frames updating
  → generation complete → show video in PreviewView
  → video saved to history automatically
```

### Error Handling

- Backend unreachable → "Backend not running. Starting..." → auto-restart
- OOM error from backend → "Not enough memory. Try lower resolution or fewer frames."
- Generation failed → show error message, keep prompt and params, allow retry
- Model not downloaded → redirect to Settings → Model Management

## Directory Structure

```
app/
├── LTXDesktop.xcodeproj
├── LTXDesktop/
│   ├── LTXDesktopApp.swift            # @main entry point
│   ├── ContentView.swift              # Sidebar + NavigationSplitView
│   │
│   ├── Views/
│   │   ├── GenerationView.swift       # Prompt + params + generate + preview
│   │   ├── PreviewView.swift          # AVPlayer + progressive display
│   │   ├── HistoryView.swift          # LazyVGrid of video thumbnails
│   │   ├── SettingsView.swift         # TabView: General, Models, Memory, LoRA
│   │   └── LoRAView.swift             # LoRA list with toggle switches
│   │
│   ├── ViewModels/
│   │   ├── GenerationViewModel.swift  # Prompt state, generation params, job tracking
│   │   ├── HistoryViewModel.swift     # Video archive data source
│   │   ├── SettingsViewModel.swift    # Model list, download state, preferences
│   │   └── MemoryViewModel.swift      # Polling memory stats, warning state
│   │
│   ├── Models/
│   │   ├── GenerationJob.swift        # Codable: job_id, status, progress, params
│   │   ├── VideoItem.swift            # Codable: path, prompt, seed, params, date
│   │   ├── ModelInfo.swift            # Codable: id, name, size, loaded
│   │   ├── MemoryStats.swift          # Codable: active_gb, cache_gb, peak_gb
│   │   └── AppSettings.swift          # @AppStorage wrapper
│   │
│   ├── Services/
│   │   ├── BackendService.swift       # URLSession HTTP + URLSessionWebSocketTask
│   │   ├── ProcessManager.swift       # Process() for backend subprocess
│   │   ├── MemoryMonitor.swift        # Timer-based polling of /system/memory
│   │   └── FFmpegWrapper.swift        # Process() shell-out to ffmpeg
│   │
│   └── Utils/
│       ├── Constants.swift            # API base URL, file paths, defaults
│       └── Extensions.swift           # Date formatting, file size formatting
│
└── Resources/
    └── Assets.xcassets                # App icon, accent color
```

## Design Guidelines

- **No third-party UI libraries** for v1 — use native SwiftUI components
- **Dark mode default** — video generation tools are typically dark-themed
- **Sidebar navigation**: Generation, History, Settings
- **Minimal chrome** — let the video preview dominate the screen
- **Parameter controls**: use `Slider`, `Stepper`, `Picker` — avoid text fields for numeric values
- **Drag & drop** for image upload (I2V) on the GenerationView
- **Keyboard shortcuts**: Cmd+G = Generate, Cmd+E = Enhance prompt, Esc = Cancel generation
