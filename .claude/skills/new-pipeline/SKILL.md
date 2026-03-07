---
name: new-pipeline
description: Scaffold a new generation pipeline end-to-end (engine + API route + WebSocket + frontend). Use when adding T2V, I2V, retake, extend, or other generation modes.
disable-model-invocation: true
args:
  - name: pipeline_name
    description: "Name of the pipeline (e.g., text_to_video, image_to_video, retake, extend)"
    required: true
---

# New Pipeline Scaffold

Create a complete generation pipeline for `{pipeline_name}` across all layers.

## Checklist

### 1. Engine Pipeline (`backend/engine/pipelines/{pipeline_name}.py`)

Create the pipeline function following these mandatory patterns:

```python
import mlx.core as mx
from engine.memory_manager import aggressive_cleanup, get_memory_stats

async def run_{pipeline_name}(
    params: dict,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> tuple[str, dict]:
    """Run {pipeline_name} pipeline.

    Args:
        params: Generation parameters.
        progress_callback: Called with (step, total_steps, pct).

    Returns:
        Tuple of (output_path, memory_stats).
    """
    # Stage 1: Text/prompt encoding
    # ... encoding logic ...
    aggressive_cleanup()  # MANDATORY

    # Stage 2: Diffusion (denoising loop)
    # ... diffusion with progress_callback ...
    aggressive_cleanup()  # MANDATORY

    # Stage 3: Upscale (if two-stage)
    # ... upscaler logic ...
    aggressive_cleanup()  # MANDATORY

    # Stage 4: VAE decode — MUST stream frame-by-frame to ffmpeg
    # NEVER decode all frames into memory
    # ... streaming decode ...
    aggressive_cleanup()  # MANDATORY

    # Stage 5: Audio decode (if applicable)
    # ... audio decode ...
    aggressive_cleanup()  # MANDATORY

    return output_path, get_memory_stats()
```

Key rules:
- `aggressive_cleanup()` between EVERY stage
- Streaming VAE decode to ffmpeg pipe (never all frames in memory)
- Return memory stats alongside output
- Accept a progress callback for WebSocket updates
- Use `mx.compile()` on model forward pass if not already compiled

### 2. API Route (`backend/main.py` or `backend/api/routes/generation.py`)

Add the endpoint:

```python
@app.post("/api/v1/generate/{pipeline_name}")
async def generate_{pipeline_name}(request: {PipelineName}Request) -> dict:
    job_id = str(uuid.uuid4())
    # Queue the job for background execution
    # ... queue logic ...
    return {"job_id": job_id}
```

- Async handler
- Creates a job_id, adds to generation queue
- Returns immediately with job_id
- Background task runs the engine pipeline
- Streams progress via WebSocket at `/ws/progress/{job_id}`

### 3. WebSocket Progress

Ensure the pipeline's progress callback sends updates through the existing WebSocket handler:

```json
{
  "step": 5,
  "total_steps": 8,
  "pct": 0.625,
  "memory": {"active_gb": 25.3, "cache_gb": 2.1},
  "preview_frame": "<base64 jpeg or null>"
}
```

### 4. Frontend (`app/LTXDesktop/`)

Add to the SwiftUI frontend:
- Request model in `Models/` matching the API request body
- Method in `Services/BackendService.swift` calling the new endpoint
- UI trigger in `Views/GenerationView.swift` (button, form fields for parameters)

### 5. Update API Contract

Add the new endpoint to the API contract in `AGENTS.md` (lines 89-111) if not already present.

### 6. Test

Create or update `tests/test_api.py` with a test for the new endpoint.

## After Scaffolding

Run the memory-reviewer agent on the new pipeline code to verify all memory safety rules are followed.
