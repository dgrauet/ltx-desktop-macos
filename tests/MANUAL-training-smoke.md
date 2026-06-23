# Manual Smoke — LoRA Training (J4b P1)

End-to-end checklist: dataset → preprocess → preflight → train → LoRA in picker → generate.
Run on a **clean GPU** (no Xcode-debugged apps, no screen recording, no games).
Timings are unmeasured; a real training run can take a while depending on step count and hardware.

---

## Prerequisites

- App backend running (`uv run --project backend uvicorn app:app --reload` or via the macOS app bundle).
- At least 6 short video clips, ideally 704×480 resolution. Each clip must satisfy `frame_count % 8 == 1` (e.g., 25, 33, 41 frames). Use ffmpeg to trim/pad if needed.
- A downloaded LTX-2.3 model variant (q4 recommended for 32GB; q8 for 64GB+).
- **Close all other GPU-heavy apps** (Xcode debugger attached to a target, games, OBS, etc.) before starting.

---

## Step 1 — Verify clean GPU state

- [ ] Quit any app with an active GPU/display client (Xcode with a device attached, Final Cut, games, screen-recording tools).
- [ ] Open Activity Monitor → GPU History — confirm no other process is holding sustained GPU usage.
- [ ] Confirm the app backend is running and `/api/v1/system/health` returns `{"status": "ok"}`.

---

## Step 2 — Create a dataset

- [ ] Open the app and navigate to the **Training** tab.
- [ ] Click **New Dataset** (or the + button in the Dataset Builder panel).
- [ ] Give the dataset a name (e.g., `smoke-test-01`).
- [ ] Drag and drop at least 6 video clips into the clip list.
  - Expected: each clip shows its filename, duration, and a frame-count indicator.
  - If a clip has an invalid frame count (not `% 8 == 1`), the UI should flag it in red.
- [ ] For each clip, enter a short caption in the caption field (e.g., `A person walking on a sunny street.`). Captions must be non-empty.
- [ ] Click **Save Manifest**.
  - Expected: success toast; manifest saved to `~/.ltx-desktop/datasets/<dataset-id>/manifest.json`.

---

## Step 3 — Preprocess the dataset

- [ ] Preprocessing runs automatically on the first preflight or training run — there is no separate "Preprocess" button.
  - The backend calls `preprocess_dataset()` against the clips directory during preflight/training startup.
  - Preprocess encodes each clip through the Gemma text encoder and VAE, saving results to `.precomputed/` alongside the videos dir.
- [ ] After running preflight (Step 4), verify:
  - `~/.ltx-desktop/datasets/<dataset-id>/.precomputed/` is populated (2 files per clip: video latents + text embeddings = 12 files for 6 clips).

---

## Step 4 — Run preflight

- [ ] Navigate to the **Training Config** panel (second panel in the Training tab).
- [ ] Review the default settings:
  - Rank: 32 (default)
  - Steps: 20 (smoke-appropriate; bump to 200+ for a real LoRA)
  - **Low-RAM mode**: toggle ON if on a 32GB machine.
- [ ] Click **Run Preflight**.
  - Expected: the backend validates frame counts, resolution divisibility, caption completeness, and disk space; returns a structured verdict.
  - If verdict is PASS: proceed.
  - If verdict is FAIL: fix the flagged issues (re-check frame counts, missing captions, disk space) and re-run preflight.
- [ ] Note the verdict message for your records.

---

## Step 5 — Start training

- [ ] Confirm **Low-RAM mode** is toggled ON (32GB machines) or OFF (64GB+).
  - Low-RAM mode uses: q4 quantized base + batch size 1 + gradient checkpointing + reduced validation.
  - Default (Low-RAM OFF): full bf16 base, standard batch/grad settings — appropriate for larger machines.
- [ ] Click **Start Training**.
  - Expected: the Progress panel activates; a WebSocket connection opens to `/ws/progress/{job_id}`; step counter advances.
- [ ] While training runs, attempt to start a generation from the Generate tab.
  - Expected: the backend returns **409 Conflict** and the UI shows a "Training in progress" error — the generation ↔ training exclusion lock is working.
- [ ] If a macOS GPU watchdog kill occurs (supervisor logs "Impacting Interactivity"):
  - Expected: the supervisor retries once automatically; training resumes.
  - If the second attempt also fails: close all other GPU apps and restart training.
- [ ] Wait for training to complete (status changes to `completed`).
  - Expected: the Progress panel shows final step count; training_store persists the run result.

---

## Step 6 — Verify LoRA output

- [ ] Check the run output directory for the LoRA file:
  ```
  ~/.ltx-desktop/training/<run_id>/checkpoints/lora_weights_step_00020.safetensors
  ```
  Expected: file exists, size ~600 MB for rank-32 (varies by rank).
- [ ] Confirm the LoRA was auto-imported: open the **Generate** tab → expand the LoRA picker.
  - Expected: a new entry matching the training run name appears in the LoRA list without requiring a manual import step.

---

## Step 7 — Generate with the new LoRA

- [ ] In the Generate tab, select the newly imported LoRA in the LoRA picker.
- [ ] Set LoRA strength to 1.0.
- [ ] Enter a short T2V prompt and click **Generate**.
  - Expected: generation proceeds normally (distilled pipeline: 8+3 steps → decode → mp4); the LoRA is fused into the transformer at load time ("Fusing LoRA … strength=1.00" in backend logs).
- [ ] Inspect the output video.
  - Note: with only 6 clips and 20 training steps, subject fidelity will be minimal — the goal here is a clean pipeline run, not quality assessment.
- [ ] Check backend logs confirm no key-mismatch error during LoRA fusion.

---

## Step 8 — Post-run checks

- [ ] Navigate to **Settings → Memory** — confirm memory metrics look stable (no runaway cache after training).
- [ ] Run a second generation without the LoRA selected — confirm it completes normally (exclusion lock released after training).
- [ ] Optionally: delete the dataset from the Dataset Builder and confirm the manifest is removed.

---

## Known limitations (not defects)

- **Per-step loss is always 0.0** — `StepCallback` provides step index and sample paths but no loss value. A real loss curve requires an upstream `ltx-2-mlx` step-loss callback (tracked as P2).
- **AV and V2V training are not implemented in P1** — the Training tab only supports T2V LoRA training.
- **`mx.get_peak_memory()` reports inflated values during training** (~6–7× RSS on 32GB) — this is an MLX accounting artifact and is not surfaced in the UI.
- **Timings are not benchmarked** — expect preprocessing to take several minutes for 6 clips; training at 20 steps is fast (a few minutes on 32GB with low-RAM mode), but 200+ steps will be substantially longer.
