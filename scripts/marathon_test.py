#!/usr/bin/env python3
"""Marathon stability test — 10 consecutive T2V generations.

Pass criteria (from CLAUDE.md):
  - No OOM crash
  - Memory at gen10 within 20% of gen1
  - No generation >2× slower than gen1

Usage:
    cd backend && ../.venv/bin/python ../scripts/marathon_test.py
    # or from project root:
    backend/.venv/bin/python scripts/marathon_test.py
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import time

# Ensure backend is importable
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

import mlx.core as mx

NUM_GENERATIONS = 10
HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 97
FPS = 24
SEED_BASE = 42

PROMPTS = [
    "A golden retriever running through a wheat field at sunset, slow motion, cinematic lighting",
    "Ocean waves crashing against rocky cliffs under a stormy sky, dramatic clouds, aerial view",
    "A neon-lit street in Tokyo at night, rain reflections, cyberpunk atmosphere, tracking shot",
    "A butterfly emerging from its chrysalis in macro detail, soft bokeh background, time lapse",
    "An astronaut floating above Earth, stars visible, helmet reflection, slow rotation",
    "A steam locomotive crossing a stone viaduct in misty mountains, golden hour, wide angle",
    "Colorful paint drops falling into water in slow motion, abstract patterns, macro lens",
    "A fox walking through a snowy forest, breath visible, moonlight filtering through trees",
    "Hot air balloons rising over Cappadocia at dawn, fairy chimneys below, warm colors",
    "A dancer performing in an empty warehouse, dust particles in light beams, contemporary style",
]


def get_memory_mb() -> dict:
    """Get current Metal memory stats in MB."""
    return {
        "active_mb": mx.get_active_memory() / 1024 / 1024,
        "peak_mb": mx.get_peak_memory() / 1024 / 1024,
        "cache_mb": mx.get_cache_memory() / 1024 / 1024,
    }


async def run_marathon() -> None:
    from engine.mlx_runner import run_mlx_generation

    output_dir = os.path.expanduser("~/.ltx-desktop/outputs/marathon_test")
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    print(f"\n{'='*70}")
    print(f"MARATHON TEST: {NUM_GENERATIONS} generations @ {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames")
    print(f"{'='*70}\n")

    for i in range(NUM_GENERATIONS):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = SEED_BASE + i
        output_path = os.path.join(output_dir, f"marathon_{i+1:02d}.mp4")

        mem_before = get_memory_mb()
        mx.reset_peak_memory()

        print(f"[Gen {i+1}/{NUM_GENERATIONS}] Starting...")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  Memory before: active={mem_before['active_mb']:.0f}MB, cache={mem_before['cache_mb']:.0f}MB")

        t0 = time.monotonic()

        try:
            await run_mlx_generation(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                seed=seed,
                fps=FPS,
                output_path=output_path,
                tiling="auto",
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "gen": i + 1,
                "duration": None,
                "error": str(e),
            })
            continue

        duration = time.monotonic() - t0
        mem_after = get_memory_mb()
        file_size = os.path.getsize(output_path) / 1024 / 1024

        result = {
            "gen": i + 1,
            "duration": duration,
            "active_mb": mem_after["active_mb"],
            "peak_mb": mem_after["peak_mb"],
            "cache_mb": mem_after["cache_mb"],
            "file_mb": file_size,
        }
        results.append(result)

        print(f"  Done in {duration:.1f}s | file={file_size:.1f}MB")
        print(f"  Memory after: active={mem_after['active_mb']:.0f}MB, peak={mem_after['peak_mb']:.0f}MB, cache={mem_after['cache_mb']:.0f}MB")

        # Cleanup between generations
        gc.collect()
        mx.clear_cache()

    # Summary
    print(f"\n{'='*70}")
    print("MARATHON TEST RESULTS")
    print(f"{'='*70}\n")

    successful = [r for r in results if r.get("duration") is not None]
    failed = [r for r in results if r.get("duration") is None]

    if not successful:
        print("ALL GENERATIONS FAILED")
        return

    print(f"{'Gen':>4} {'Duration':>10} {'Active MB':>10} {'Peak MB':>10} {'Cache MB':>10} {'File MB':>8}")
    print("-" * 60)
    for r in results:
        if r.get("duration"):
            print(f"{r['gen']:>4} {r['duration']:>9.1f}s {r['active_mb']:>10.0f} {r['peak_mb']:>10.0f} {r['cache_mb']:>10.0f} {r['file_mb']:>7.1f}")
        else:
            print(f"{r['gen']:>4}    FAILED — {r.get('error', 'unknown')[:40]}")

    # Pass/fail criteria
    print(f"\n{'='*70}")
    print("PASS/FAIL CRITERIA")
    print(f"{'='*70}\n")

    gen1 = successful[0]
    gen_last = successful[-1]

    # 1. No OOM
    oom_pass = len(failed) == 0
    print(f"1. No OOM crashes:       {'PASS' if oom_pass else 'FAIL'} ({len(failed)} failures)")

    # 2. Memory within 20%
    if len(successful) >= 2:
        mem_ratio = gen_last["active_mb"] / gen1["active_mb"] if gen1["active_mb"] > 0 else 1.0
        mem_pass = mem_ratio <= 1.20
        print(f"2. Memory gen{gen_last['gen']} ≤ 120% gen1: {'PASS' if mem_pass else 'FAIL'} "
              f"(gen1={gen1['active_mb']:.0f}MB, gen{gen_last['gen']}={gen_last['active_mb']:.0f}MB, ratio={mem_ratio:.2f})")
    else:
        mem_pass = False
        print("2. Memory stability:     SKIP (not enough successful gens)")

    # 3. No gen >2× slower than gen1
    durations = [r["duration"] for r in successful]
    slowest = max(durations)
    speed_pass = slowest <= gen1["duration"] * 2.0
    slowest_gen = [r["gen"] for r in successful if r["duration"] == slowest][0]
    print(f"3. No gen >2× gen1 time: {'PASS' if speed_pass else 'FAIL'} "
          f"(gen1={gen1['duration']:.1f}s, slowest=gen{slowest_gen} {slowest:.1f}s, ratio={slowest/gen1['duration']:.2f})")

    all_pass = oom_pass and mem_pass and speed_pass
    print(f"\nOVERALL: {'✅ PASS' if all_pass else '❌ FAIL'}")

    # Stats
    avg_duration = sum(durations) / len(durations)
    print(f"\nAverage duration: {avg_duration:.1f}s")
    print(f"Total time: {sum(durations):.1f}s ({sum(durations)/60:.1f} min)")


if __name__ == "__main__":
    asyncio.run(run_marathon())
