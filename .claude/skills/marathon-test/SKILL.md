---
name: marathon-test
description: Run the marathon stability test (10 consecutive generations) and analyze memory trends. Use to validate app stability before releases.
disable-model-invocation: true
---

# Marathon Stability Test

Run the marathon generation stability test — the release gate for LTX Desktop macOS.

## Steps

1. Check that the Python backend environment is set up:
   ```bash
   cd backend && uv sync --quiet
   ```

2. Run the marathon test:
   ```bash
   cd backend && uv run pytest ../tests/test_marathon.py -v --tb=long -s
   ```

3. Analyze the output for these pass/fail criteria:
   - **No OOM crashes** during any of the 10 generations
   - **Memory stability**: memory after generation 10 must be within 20% of memory after generation 1
   - **Performance stability**: no generation takes more than 2x longer than generation 1
   - **No kernel panics or Metal errors**

4. If the test file doesn't exist yet, report that `tests/test_marathon.py` needs to be created first (see tests/CLAUDE.md for the spec).

5. Report results in this format:
   ```
   ## Marathon Test Results

   | Gen # | Duration | Active Memory | Cache Memory | Peak Memory |
   |-------|----------|---------------|--------------|-------------|
   | 1     | Xs       | X.XGB         | X.XGB        | X.XGB       |
   | ...   | ...      | ...           | ...          | ...         |
   | 10    | Xs       | X.XGB         | X.XGB        | X.XGB       |

   **Memory drift**: X% (gen 10 vs gen 1) — PASS/FAIL (threshold: 20%)
   **Slowest gen**: X (Xs, ratio vs gen 1: X.Xx) — PASS/FAIL (threshold: 2.0x)
   **OOM crashes**: 0 — PASS/FAIL

   **Overall**: PASS / FAIL
   ```

6. If FAIL, identify the likely cause (memory leak, missing cleanup, fragmentation) and suggest specific fixes.
