"""Training subprocess stderr protocol (extends the mlx_runner stderr convention)."""
from __future__ import annotations


def format_step(*, step: int, loss: float, lr: float, peak_gb: float) -> str:
    return f"STEP:{step}:{loss!r}:{lr!r}:{peak_gb!r}"


def format_sample(path: str) -> str:
    return f"SAMPLE:{path}"


def format_done(path: str) -> str:
    return f"DONE:{path}"


def format_error(msg: str) -> str:
    return f"ERROR:{msg}"


def parse_line(line: str) -> dict | None:
    line = line.rstrip("\n")
    if line.startswith("STEP:"):
        try:
            _, step, loss, lr, peak = line.split(":", 4)
            return {"type": "step", "step": int(step), "loss": float(loss),
                    "lr": float(lr), "peak_mem_gb": float(peak)}
        except ValueError:
            return None
    if line.startswith("SAMPLE:"):
        return {"type": "sample", "path": line[len("SAMPLE:"):]}
    if line.startswith("DONE:"):
        return {"type": "done", "lora_path": line[len("DONE:"):]}
    if line.startswith("STATUS:"):
        return {"type": "status", "status": line[len("STATUS:"):]}
    if line.startswith("ERROR:"):
        return {"type": "error", "message": line[len("ERROR:"):]}
    return None


def preflight_verdict(peak_gb: float, available_gb: float) -> str:
    if peak_gb < 0.60 * available_gb:
        return "ok"
    if peak_gb < available_gb:
        return "risky"
    return "oom"
