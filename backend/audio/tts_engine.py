"""Text-to-Speech engine using MLX-Audio (local, on-device).

Sprint 4 stub: generates a sine-wave placeholder audio file via ffmpeg.
Real implementation uses mlx_audio with Kokoro or CSM model.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Locate the ffmpeg binary on the system.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        RuntimeError: If ffmpeg cannot be found.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(candidate).exists():
                ffmpeg_bin = candidate
                break
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    return ffmpeg_bin


class TTSEngine:
    """Local text-to-speech synthesis via MLX-Audio.

    Sprint 4 stub: produces a sine-wave WAV whose duration approximates the
    reading time of the supplied text. Real implementation will call
    mlx_audio with the Kokoro or CSM model.
    """

    OUTPUT_DIR: Path = Path.home() / ".ltx-desktop" / "audio" / "tts"

    @classmethod
    def is_available(cls) -> bool:
        """Return True if mlx_audio is importable and TTS is possible.

        Returns:
            True when mlx_audio can be imported, False otherwise.
        """
        try:
            import mlx_audio  # noqa: F401

            return True
        except ImportError:
            return False

    def synthesize(self, text: str, voice: str = "default", speed: float = 1.0) -> str:
        """Synthesize speech from text and write it to a WAV file.

        Uses mlx_audio when available; falls back to a sine-wave stub via
        ffmpeg so the rest of the pipeline can be tested without the full
        TTS model installed.

        Args:
            text: The text to synthesise into speech.
            voice: Voice identifier (see list_voices()). Ignored by the stub.
            speed: Playback speed multiplier (1.0 = normal). Affects duration
                estimate in the stub.

        Returns:
            Absolute path to the generated WAV file as a string.

        Raises:
            RuntimeError: If ffmpeg is not found (stub path) or mlx_audio
                fails (real path).
        """
        log.info("TTSEngine.synthesize: text=%r, voice=%s, speed=%.2f", text[:60], voice, speed)

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_id = str(uuid.uuid4())[:8]
        output_path = self.OUTPUT_DIR / f"tts_{output_id}.wav"

        if self.is_available():
            # Real path — delegate to mlx_audio (implemented in a future sprint)
            log.info("TTSEngine: mlx_audio available — stub still used in Sprint 4")

        # Stub path: generate a sine-wave WAV via ffmpeg.
        # Duration estimate: words / words_per_second / speed_factor
        word_count = len(text.split())
        words_per_second = 2.5
        duration = max(1.0, word_count / words_per_second / max(speed, 0.1))

        ffmpeg_bin = _find_ffmpeg()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=220:duration={duration:.3f}",
            "-ar", "22050",
            "-ac", "1",
            str(output_path),
        ]
        log.debug("TTSEngine stub command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg TTS stub failed: {result.stderr[:500]}")

        log.info("TTSEngine: stub WAV written to %s (duration=%.2fs)", output_path, duration)
        return str(output_path)

    def list_voices(self) -> list[str]:
        """Return the list of supported voice identifiers.

        Returns:
            List of voice name strings. The first entry is the default.
        """
        return ["default", "kokoro_af", "kokoro_am", "csm_dialog"]
