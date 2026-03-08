"""Entry point for quantized model inference.

Applies the quantized weight loading patch, then delegates to
``mlx_video.generate_av.main()`` for the actual generation.

Usage::

    python -m engine.generate_av_quantized --prompt "..." --model-repo /path/to/int4 ...

All arguments are passed through to ``mlx_video.generate_av.main()``.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Apply quantized weight loading patch BEFORE importing generate_av
from engine.quantized_patch import apply_patch

apply_patch()

# Now delegate to the original CLI
from mlx_video.generate_av import main

if __name__ == "__main__":
    main()
