"""LTX Desktop macOS — FastAPI Backend.

Sprint 1: system endpoints, T2V generation, WebSocket progress.
Sprint 2: preview, I2V, intermediate frame streaming.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from engine.memory_manager import aggressive_cleanup, get_memory_stats
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import TextToVideoPipeline
from engine.pipelines.preview import PreviewPipeline
from engine.pipelines.image_to_video import ImageToVideoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Global state ---

model_manager = ModelManager()
t2v_pipeline = TextToVideoPipeline(model_manager)
preview_pipeline = PreviewPipeline(model_manager)
i2v_pipeline = ImageToVideoPipeline(model_manager)

# Job tracking: job_id -> {status, progress, result, error}
jobs: dict[str, dict[str, Any]] = {}

# WebSocket connections per job
ws_connections: dict[str, list[WebSocket]] = {}


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model stub, warm up. Shutdown: unload."""
    log.info("Starting LTX Desktop backend...")
    model_manager.load_model("notapalindrome/ltx2-mlx-av")
    aggressive_cleanup()
    log.info("Backend ready")
    yield
    log.info("Shutting down...")
    model_manager.unload_all()


app = FastAPI(
    title="LTX Desktop Backend",
    version="0.2.0",
    lifespan=lifespan,
)


# --- Request/Response models ---

class T2VRequest(BaseModel):
    """Text-to-Video generation request."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    width: int = Field(default=768, ge=256, le=1920)
    height: int = Field(default=512, ge=256, le=1920)
    num_frames: int = Field(default=97, ge=9)
    steps: int = Field(default=8, ge=1, le=50)
    seed: int = Field(default=42)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    fps: int = Field(default=24, ge=1, le=60)


class PreviewRequest(BaseModel):
    """Rapid preview generation request. Resolution and steps are fixed."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int = Field(default=42)
    fps: int = Field(default=24, ge=1, le=60)


class I2VRequest(BaseModel):
    """Image-to-Video generation request."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    source_image_path: str = Field(..., min_length=1)
    width: int = Field(default=768, ge=256, le=1920)
    height: int = Field(default=512, ge=256, le=1920)
    num_frames: int = Field(default=97, ge=9)
    steps: int = Field(default=8, ge=1, le=50)
    seed: int = Field(default=42)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    fps: int = Field(default=24, ge=1, le=60)


class JobResponse(BaseModel):
    """Response with a job ID."""
    job_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    generation_count: int


class SystemInfoResponse(BaseModel):
    """System information response."""
    chip: str
    ram_total_gb: int
    ram_available_gb: float
    macos_version: str


class MemoryResponse(BaseModel):
    """Memory statistics response."""
    active_memory_gb: float
    cache_memory_gb: float
    peak_memory_gb: float
    system_available_gb: float
    generation_count_since_reload: int
    next_reload_in: int


# --- System endpoints ---

@app.get("/api/v1/system/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    stats = get_memory_stats()
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded(),
        generation_count=stats["generation_count_since_reload"],
    )


@app.get("/api/v1/system/info", response_model=SystemInfoResponse)
async def system_info():
    """Detect Apple Silicon chip and RAM."""
    chip = "Unknown"
    ram_total_gb = 0
    macos_version = platform.mac_ver()[0] or "Unknown"

    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        pass

    try:
        ram_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        ram_total_gb = ram_bytes // (1024**3)
    except Exception:
        pass

    stats = get_memory_stats()
    return SystemInfoResponse(
        chip=chip,
        ram_total_gb=ram_total_gb,
        ram_available_gb=stats["system_available_gb"],
        macos_version=macos_version,
    )


@app.get("/api/v1/system/memory", response_model=MemoryResponse)
async def memory_stats():
    """Real-time Metal memory statistics."""
    stats = get_memory_stats()
    return MemoryResponse(**stats)


# --- Generation endpoints ---

@app.post("/api/v1/generate/text-to-video", response_model=JobResponse)
async def generate_t2v(req: T2VRequest):
    """Start a text-to-video generation job."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}

    asyncio.create_task(_run_t2v(job_id, req))

    return JobResponse(job_id=job_id)


@app.post("/api/v1/generate/preview", response_model=JobResponse)
async def generate_preview(req: PreviewRequest):
    """Start a rapid preview generation job (384x256, 4 steps)."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}

    asyncio.create_task(_run_preview(job_id, req))

    return JobResponse(job_id=job_id)


@app.post("/api/v1/generate/image-to-video", response_model=JobResponse)
async def generate_i2v(req: I2VRequest):
    """Start an image-to-video generation job."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}

    asyncio.create_task(_run_i2v(job_id, req))

    return JobResponse(job_id=job_id)


async def _run_t2v(job_id: str, req: T2VRequest) -> None:
    """Execute T2V generation in background."""
    jobs[job_id]["status"] = "running"

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame)

    try:
        result = await t2v_pipeline.generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            progress_callback=progress_cb,
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "job_id": result.job_id,
            "output_path": result.output_path,
            "duration_seconds": result.duration_seconds,
            "memory_after": result.memory_after,
            "stages": result.stages,
        }
        await _broadcast_progress(job_id, 0, 0, 1.0, done=True)
        log.info("Job %s completed in %.2fs", job_id, result.duration_seconds)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log.error("Job %s failed: %s", job_id, e)
        await _broadcast_progress(job_id, 0, 0, 0.0, error=str(e))


async def _run_preview(job_id: str, req: PreviewRequest) -> None:
    """Execute rapid preview generation in background."""
    jobs[job_id]["status"] = "running"

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame)

    try:
        result = await preview_pipeline.generate(
            prompt=req.prompt,
            seed=req.seed,
            fps=req.fps,
            progress_callback=progress_cb,
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "job_id": result.job_id,
            "output_path": result.output_path,
            "duration_seconds": result.duration_seconds,
            "memory_after": result.memory_after,
            "stages": result.stages,
        }
        await _broadcast_progress(job_id, 0, 0, 1.0, done=True)
        log.info("Preview %s completed in %.2fs", job_id, result.duration_seconds)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log.error("Preview %s failed: %s", job_id, e)
        await _broadcast_progress(job_id, 0, 0, 0.0, error=str(e))


async def _run_i2v(job_id: str, req: I2VRequest) -> None:
    """Execute I2V generation in background."""
    jobs[job_id]["status"] = "running"

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame)

    try:
        result = await i2v_pipeline.generate(
            prompt=req.prompt,
            source_image_path=req.source_image_path,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            progress_callback=progress_cb,
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {
            "job_id": result.job_id,
            "output_path": result.output_path,
            "duration_seconds": result.duration_seconds,
            "memory_after": result.memory_after,
            "stages": result.stages,
        }
        await _broadcast_progress(job_id, 0, 0, 1.0, done=True)
        log.info("I2V %s completed in %.2fs", job_id, result.duration_seconds)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log.error("I2V %s failed: %s", job_id, e)
        await _broadcast_progress(job_id, 0, 0, 0.0, error=str(e))


# --- Queue endpoints ---

@app.get("/api/v1/queue")
async def list_queue():
    """List all jobs and their status."""
    return [
        {"job_id": jid, "status": info["status"], "progress": info["progress"]}
        for jid, info in jobs.items()
    ]


@app.post("/api/v1/queue/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    if job_id not in jobs:
        return {"success": False, "error": "Job not found"}
    jobs[job_id]["status"] = "cancelled"
    return {"success": True}


@app.get("/api/v1/queue/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific job."""
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]


# --- WebSocket progress ---

@app.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """Stream generation progress via WebSocket."""
    await websocket.accept()

    if job_id not in ws_connections:
        ws_connections[job_id] = []
    ws_connections[job_id].append(websocket)

    try:
        # Send current state immediately
        if job_id in jobs:
            await websocket.send_json({
                "job_id": job_id,
                "status": jobs[job_id]["status"],
                "progress": jobs[job_id]["progress"],
            })

        # Keep connection alive until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in ws_connections:
            ws_connections[job_id] = [
                ws for ws in ws_connections[job_id] if ws != websocket
            ]


async def _broadcast_progress(
    job_id: str,
    step: int,
    total_steps: int,
    pct: float,
    done: bool = False,
    error: str | None = None,
    preview_frame: str | None = None,
) -> None:
    """Broadcast progress update to all WebSocket connections for a job."""
    if job_id not in ws_connections:
        return

    memory = get_memory_stats()
    msg: dict[str, Any] = {
        "job_id": job_id,
        "step": step,
        "total_steps": total_steps,
        "pct": round(pct, 4),
        "memory": memory,
        "done": done,
        "error": error,
    }

    if preview_frame is not None:
        msg["preview_frame"] = preview_frame

    dead: list[WebSocket] = []
    for ws in ws_connections[job_id]:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        ws_connections[job_id].remove(ws)


# --- Entry point ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
