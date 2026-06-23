"""P0: training stderr protocol round-trips and preflight verdict thresholds."""
from engine.training.protocol import (
    parse_line, format_step, format_done, format_error, preflight_verdict,
)


def test_step_round_trip():
    line = format_step(step=10, loss=0.0423, lr=5e-4, peak_gb=24.5)
    evt = parse_line(line)
    assert evt == {"type": "step", "step": 10, "loss": 0.0423,
                   "lr": 5e-4, "peak_mem_gb": 24.5}


def test_done_and_error():
    assert parse_line(format_done("/x/lora.safetensors")) == {
        "type": "done", "lora_path": "/x/lora.safetensors"}
    assert parse_line(format_error("OOM"))["type"] == "error"


def test_unmatched_line_is_none():
    assert parse_line("INFO: loading weights") is None


def test_preflight_verdict_thresholds():
    assert preflight_verdict(peak_gb=10.0, available_gb=26.0) == "ok"
    assert preflight_verdict(peak_gb=20.0, available_gb=26.0) == "risky"
    assert preflight_verdict(peak_gb=27.0, available_gb=26.0) == "oom"


def test_step_malformed_returns_none():
    """parse_line("STEP:abc") must return None, not raise."""
    assert parse_line("STEP:abc") is None
