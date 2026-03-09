"""LTX Desktop macOS — FastAPI Backend.

Sprint 1: system endpoints, T2V generation, WebSocket progress.
Sprint 2: preview, I2V, intermediate frame streaming.
Sprint 4: LoRA management, audio (TTS/music/mix), video export, FCPXML export.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import random
import subprocess
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from engine.memory_manager import aggressive_cleanup, get_memory_stats
from engine.model_manager import ModelManager
from engine.pipelines.text_to_video import TextToVideoPipeline
from engine.pipelines.preview import PreviewPipeline
from engine.pipelines.image_to_video import ImageToVideoPipeline
from engine.pipelines.retake import RetakePipeline
from engine.pipelines.extend import ExtendPipeline
from engine.prompt_enhancer import PromptEnhancer
from engine.lora_manager import LoRAManager
from audio.tts_engine import TTSEngine
from audio.audio_mixer import AudioMixer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Global state ---

model_manager = ModelManager()
t2v_pipeline = TextToVideoPipeline(model_manager)
preview_pipeline = PreviewPipeline(model_manager)
i2v_pipeline = ImageToVideoPipeline(model_manager)
retake_pipeline = RetakePipeline(model_manager)
extend_pipeline = ExtendPipeline(model_manager)
prompt_enhancer = PromptEnhancer()
lora_manager = LoRAManager(model_manager)
tts_engine = TTSEngine()
audio_mixer = AudioMixer()

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
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    fps: int = Field(default=24, ge=1, le=60)
    upscale: bool = Field(default=False, description="2x spatial upscale via latent upsampler")


class PreviewRequest(BaseModel):
    """Rapid preview generation request. Resolution and steps are fixed."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    fps: int = Field(default=24, ge=1, le=60)
    source_image_path: str | None = Field(default=None)
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    upscale: bool = Field(default=False, description="2x spatial upscale (384x256 -> 768x512)")


class I2VRequest(BaseModel):
    """Image-to-Video generation request."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    source_image_path: str = Field(..., min_length=1)
    width: int = Field(default=768, ge=256, le=1920)
    height: int = Field(default=512, ge=256, le=1920)
    num_frames: int = Field(default=97, ge=9)
    steps: int = Field(default=8, ge=1, le=50)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    fps: int = Field(default=24, ge=1, le=60)
    image_strength: float = Field(default=0.85, ge=0.0, le=1.0)
    upscale: bool = Field(default=False, description="2x spatial upscale via latent upsampler")


class EnhanceRequest(BaseModel):
    """Prompt enhancement request."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    is_i2v: bool = Field(default=False)


class EnhanceResponse(BaseModel):
    """Prompt enhancement response."""
    original: str
    enhanced: str


class RetakeRequest(BaseModel):
    """Retake (segment regeneration) request."""
    source_video_path: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=2000)
    start_time_s: float = Field(..., ge=0.0)
    end_time_s: float = Field(..., gt=0.0)
    steps: int = Field(default=8, ge=1, le=50)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    fps: int = Field(default=24, ge=1, le=60)


class ExtendRequest(BaseModel):
    """Video extension request."""
    source_video_path: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=2000)
    direction: str = Field(default="forward", pattern="^(forward|backward)$")
    extension_frames: int = Field(default=49, ge=9)
    steps: int = Field(default=8, ge=1, le=50)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    fps: int = Field(default=24, ge=1, le=60)


class JobResponse(BaseModel):
    """Response with a job ID."""
    job_id: str


def _resolve_seed(seed: int) -> int:
    """Return a concrete seed — generates a random one if seed is -1."""
    if seed < 0:
        return random.randint(0, 2**31 - 1)
    return seed


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


class LoRAInfo(BaseModel):
    """LoRA metadata."""
    id: str
    name: str
    path: str
    lora_type: str
    compatible: bool
    loaded: bool
    size_mb: float


class LoadLoRARequest(BaseModel):
    """Request to load a LoRA by ID."""
    lora_id: str = Field(..., min_length=1)


class TTSRequest(BaseModel):
    """Text-to-speech synthesis request."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class MusicRequest(BaseModel):
    """Background music generation request."""
    genre: str = Field(default="ambient")
    duration: float = Field(default=10.0, ge=1.0, le=300.0)


class MixRequest(BaseModel):
    """Audio mix request — combine TTS and/or music into a video."""
    video_path: str = Field(..., min_length=1)
    tts_path: str | None = None
    music_path: str | None = None
    music_volume: float = Field(default=0.3, ge=0.0, le=1.0)
    tts_volume: float = Field(default=1.0, ge=0.0, le=1.0)


class ExportVideoRequest(BaseModel):
    """Video re-encode/export request."""
    video_path: str = Field(..., min_length=1)
    codec: str = Field(default="h264", pattern="^(h264|h265|hevc|prores)$")
    output_format: str = Field(default="mp4", pattern="^(mp4|mov)$")
    bitrate: str = Field(default="8M")


class ExportFCPXMLRequest(BaseModel):
    """FCPXML export request for Final Cut Pro."""
    video_path: str = Field(..., min_length=1)
    clip_name: str = Field(default="LTX Generated Clip")
    frame_rate: str = Field(default="24/1")


class PathResponse(BaseModel):
    """Response carrying a single output file path."""
    output_path: str


class SuccessResponse(BaseModel):
    """Generic success/failure response."""
    success: bool
    message: str = ""


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

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None, *, status: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame, status=status)

    try:
        result = await t2v_pipeline.generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.steps,
            seed=_resolve_seed(req.seed),
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            upscale=req.upscale,
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

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None, *, status: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame, status=status)

    try:
        result = await preview_pipeline.generate(
            prompt=req.prompt,
            seed=_resolve_seed(req.seed),
            fps=req.fps,
            image=req.source_image_path,
            image_strength=req.image_strength,
            upscale=req.upscale,
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

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None, *, status: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame, status=status)

    try:
        result = await i2v_pipeline.generate(
            prompt=req.prompt,
            source_image_path=req.source_image_path,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            steps=req.steps,
            seed=_resolve_seed(req.seed),
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            image_strength=req.image_strength,
            upscale=req.upscale,
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


@app.post("/api/v1/generate/retake", response_model=JobResponse)
async def generate_retake(req: RetakeRequest):
    """Start a retake (segment regeneration) job."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}

    asyncio.create_task(_run_retake(job_id, req))

    return JobResponse(job_id=job_id)


@app.post("/api/v1/generate/extend", response_model=JobResponse)
async def generate_extend(req: ExtendRequest):
    """Start a video extension job."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}

    asyncio.create_task(_run_extend(job_id, req))

    return JobResponse(job_id=job_id)


async def _run_retake(job_id: str, req: RetakeRequest) -> None:
    """Execute retake generation in background."""
    jobs[job_id]["status"] = "running"

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None, *, status: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame, status=status)

    try:
        result = await retake_pipeline.generate(
            source_video_path=req.source_video_path,
            prompt=req.prompt,
            start_time_s=req.start_time_s,
            end_time_s=req.end_time_s,
            steps=req.steps,
            seed=_resolve_seed(req.seed),
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
        log.info("Retake %s completed in %.2fs", job_id, result.duration_seconds)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log.error("Retake %s failed: %s", job_id, e)
        await _broadcast_progress(job_id, 0, 0, 0.0, error=str(e))


async def _run_extend(job_id: str, req: ExtendRequest) -> None:
    """Execute video extension in background."""
    jobs[job_id]["status"] = "running"

    async def progress_cb(step: int, total: int, pct: float, preview_frame: str | None = None, *, status: str | None = None) -> None:
        jobs[job_id]["progress"] = pct
        await _broadcast_progress(job_id, step, total, pct, preview_frame=preview_frame, status=status)

    try:
        result = await extend_pipeline.generate(
            source_video_path=req.source_video_path,
            prompt=req.prompt,
            direction=req.direction,
            extension_frames=req.extension_frames,
            steps=req.steps,
            seed=_resolve_seed(req.seed),
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
        log.info("Extend %s completed in %.2fs", job_id, result.duration_seconds)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        log.error("Extend %s failed: %s", job_id, e)
        await _broadcast_progress(job_id, 0, 0, 0.0, error=str(e))


# --- Prompt enhancement endpoint ---

@app.post("/api/v1/prompt/enhance", response_model=EnhanceResponse)
async def enhance_prompt(req: EnhanceRequest):
    """Enhance a prompt via Qwen3.5-2B (lazy load/unload)."""
    if not prompt_enhancer.is_available():
        raise HTTPException(
            status_code=503,
            detail="Prompt enhancer not available: install mlx-lm",
        )

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, prompt_enhancer.enhance, req.prompt, req.is_i2v
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prompt enhancement failed: {exc}",
        )
    return EnhanceResponse(original=req.prompt, enhanced=result)


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
    status: str | None = None,
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

    if status is not None:
        msg["status"] = status

    dead: list[WebSocket] = []
    for ws in ws_connections[job_id]:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        ws_connections[job_id].remove(ws)


# --- LoRA endpoints ---

@app.get("/api/v1/loras", response_model=list[LoRAInfo])
async def list_loras():
    """List all available LoRAs."""
    loras = lora_manager.list_loras()
    return [LoRAInfo(**vars(lora)) for lora in loras]


@app.post("/api/v1/loras/load", response_model=SuccessResponse)
async def load_lora(req: LoadLoRARequest):
    """Load (activate) a LoRA by ID."""
    try:
        lora_manager.load_lora(req.lora_id)
        return SuccessResponse(success=True, message=f"LoRA '{req.lora_id}' loaded")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"LoRA '{req.lora_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/loras/unload/{lora_id}", response_model=SuccessResponse)
async def unload_lora(lora_id: str):
    """Unload (deactivate) a LoRA by ID."""
    lora_manager.unload_lora(lora_id)
    return SuccessResponse(success=True, message=f"LoRA '{lora_id}' unloaded")


# --- Audio endpoints ---

@app.post("/api/v1/audio/tts", response_model=PathResponse)
async def audio_tts(req: TTSRequest):
    """Generate TTS audio from text (local, on-device)."""
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: tts_engine.synthesize(req.text, req.voice, req.speed)
    )
    return PathResponse(output_path=result)


@app.post("/api/v1/audio/music", response_model=PathResponse)
async def audio_music(req: MusicRequest):
    """Generate background music (stub: sine wave placeholder)."""
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: audio_mixer.generate_music_stub(req.genre, req.duration)
    )
    return PathResponse(output_path=result)


@app.post("/api/v1/audio/mix", response_model=PathResponse)
async def audio_mix(req: MixRequest):
    """Mix TTS and/or music into a video."""
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: audio_mixer.mix(
            video_path=req.video_path,
            tts_path=req.tts_path,
            music_path=req.music_path,
            music_volume=req.music_volume,
            tts_volume=req.tts_volume,
        ),
    )
    return PathResponse(output_path=result)


# --- Export endpoints ---

@app.post("/api/v1/export/video", response_model=PathResponse)
async def export_video(req: ExportVideoRequest):
    """Re-encode a video with the specified codec and format."""
    import shutil
    import subprocess
    from pathlib import Path

    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg_bin).exists():
        raise HTTPException(status_code=503, detail="ffmpeg not found")

    output_dir = Path.home() / ".ltx-desktop" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    codec_map = {"h264": "libx264", "h265": "libx265", "hevc": "libx265", "prores": "prores_ks"}
    vcodec = codec_map.get(req.codec, "libx264")
    ext = "mov" if req.output_format == "mov" or req.codec == "prores" else "mp4"

    output_path = output_dir / f"export_{str(uuid.uuid4())[:8]}.{ext}"

    cmd = [
        ffmpeg_bin, "-y", "-i", req.video_path,
        "-c:v", vcodec, "-b:v", req.bitrate,
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Export failed: {result.stderr[:200]}")
    return PathResponse(output_path=str(output_path))


@app.post("/api/v1/export/fcpxml", response_model=PathResponse)
async def export_fcpxml(req: ExportFCPXMLRequest):
    """Export a video clip as FCPXML for Final Cut Pro."""
    import json as _json
    import shutil
    import subprocess
    from pathlib import Path

    output_dir = Path.home() / ".ltx-desktop" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"export_{str(uuid.uuid4())[:8]}.fcpxml"

    # Probe duration with ffprobe
    ffprobe_bin = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    duration_s = 4.0
    if Path(ffprobe_bin).exists():
        probe = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_format", req.video_path],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            try:
                duration_s = float(_json.loads(probe.stdout)["format"]["duration"])
            except Exception:
                pass

    # Build minimal FCPXML 1.11
    fps_num, fps_den = req.frame_rate.split("/")
    frame_duration = f"1/{fps_num}s"
    total_frames = int(duration_s * int(fps_num) / int(fps_den))
    duration_str = f"{total_frames}/{fps_num}s"
    clip_uid = str(uuid.uuid4())

    fcpxml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.11">
    <resources>
        <format id="r1" name="FFVideoFormat{fps_num}p" frameDuration="{frame_duration}" width="1920" height="1080"/>
        <asset id="r2" name="{req.clip_name}" uid="{clip_uid}" start="0s" duration="{duration_str}" hasVideo="1" hasAudio="1">
            <media-rep kind="original-media" src="file://{req.video_path}"/>
        </asset>
    </resources>
    <library>
        <event name="LTX Desktop Exports">
            <project name="{req.clip_name}">
                <sequence format="r1" duration="{duration_str}" tcStart="0s" tcFormat="NDF">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="{req.clip_name}" duration="{duration_str}" start="0s"/>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>'''

    output_path.write_text(fcpxml, encoding="utf-8")
    log.info("FCPXML exported: %s", output_path)
    return PathResponse(output_path=str(output_path))


# --- Entry point ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
