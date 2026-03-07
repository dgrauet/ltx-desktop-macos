# Scripts — CLAUDE.md

Setup and development scripts for LTX Desktop macOS. **Agent 4 (QA & DevOps)** owns this directory.

---

## Scripts

### setup.sh — Full Environment Setup

Must be idempotent (safe to re-run). Checks and installs:

```bash
#!/bin/bash
set -e

# 1. Check macOS version (>= 14.0 Sonoma)
# 2. Check/install Homebrew
# 3. Check/install Python 3.12+ via brew
# 4. Check/install uv (Python package manager)
# 5. Check/install ffmpeg via brew
# 6. cd backend && uv sync (create venv + install deps)
# 7. Print system info (chip, RAM, macOS version)
# 8. Print next steps (download models, start dev)
```

**Rules**:
- Never use `sudo` without asking
- Check if each tool exists before installing
- Print clear progress messages
- Exit with helpful error if Apple Silicon not detected (MLX won't work on Intel)

### download_models.sh — Model Download

```bash
#!/bin/bash
set -e

# 1. Check disk space (need ~50GB free)
# 2. Download LTX-2.3 distilled MLX model
#    huggingface-cli download notapalindrome/ltx2-mlx-av
# 3. Download Qwen3.5-2B for prompt enhancement
#    huggingface-cli download Qwen/Qwen3.5-2B
# 4. Verify downloads (check file sizes / checksums)
# 5. Print model locations and sizes
```

**Rules**:
- Show download progress
- Support resume (huggingface-cli handles this)
- Check if already downloaded before re-downloading
- Print total disk usage at the end

### dev.sh — Development Launch

```bash
#!/bin/bash
set -e

# 1. Check prerequisites (python, uv, ffmpeg, models)
# 2. cd backend && source .venv/bin/activate
# 3. Start backend: uvicorn main:app --host 127.0.0.1 --port 8000 --reload &
# 4. Wait for backend health check
# 5. Open Xcode project: open app/LTXDesktop.xcodeproj
# 6. Print status and instructions
# 7. Trap SIGINT to cleanly stop backend on Ctrl+C
```

**Rules**:
- Don't start a second backend if one is already running (check port 8000)
- Clean shutdown on Ctrl+C
- Print the backend URL and health check status

## All Scripts Must

- Start with `#!/bin/bash` and `set -e`
- Be executable (`chmod +x`)
- Work from the project root directory
- Print colored output (green=success, red=error, yellow=warning)
- Be idempotent
