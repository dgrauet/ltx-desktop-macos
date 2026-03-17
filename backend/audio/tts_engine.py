"""Text-to-Speech engine using MLX-Audio Kokoro model (local, on-device).

Lazy-loads the Kokoro TTS model on first use, generates speech, then
immediately unloads the model and frees Metal memory. The TTS model must
NEVER coexist with the video model in memory on machines < 64GB.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)

# Kokoro model ID on HuggingFace (MLX-optimized).
_KOKORO_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# Map user-facing voice names to Kokoro voice IDs.
# Kokoro has 54 voice presets across multiple languages.
_VOICE_MAP: dict[str, str] = {
    # Default → pleasant American female
    "default": "af_heart",
    # American female voices
    "kokoro_af_heart": "af_heart",
    "kokoro_af_bella": "af_bella",
    "kokoro_af_nova": "af_nova",
    "kokoro_af_sarah": "af_sarah",
    "kokoro_af_nicole": "af_nicole",
    "kokoro_af_sky": "af_sky",
    # American male voices
    "kokoro_am_adam": "am_adam",
    "kokoro_am_echo": "am_echo",
    "kokoro_am_michael": "am_michael",
    "kokoro_am_liam": "am_liam",
    # British female voices
    "kokoro_bf_alice": "bf_alice",
    "kokoro_bf_emma": "bf_emma",
    "kokoro_bf_lily": "bf_lily",
    # British male voices
    "kokoro_bm_daniel": "bm_daniel",
    "kokoro_bm_george": "bm_george",
    # Legacy names (backward compat with stub)
    "kokoro_af": "af_heart",
    "kokoro_am": "am_adam",
    # csm_dialog from old stub — map to a conversational-sounding voice
    "csm_dialog": "af_bella",
}

# Language code mapping based on voice prefix.
_LANG_CODE_MAP: dict[str, str] = {
    "a": "a",   # American English
    "b": "b",   # British English
    "j": "j",   # Japanese
    "z": "z",   # Chinese
}


def _voice_to_lang_code(voice_id: str) -> str:
    """Infer language code from Kokoro voice ID prefix.

    Args:
        voice_id: Kokoro voice identifier (e.g. "af_heart", "bm_daniel").

    Returns:
        Language code string for Kokoro generate().
    """
    if voice_id and len(voice_id) >= 1:
        prefix = voice_id[0]
        return _LANG_CODE_MAP.get(prefix, "a")
    return "a"


class TTSEngine:
    """Local text-to-speech synthesis via MLX-Audio Kokoro model.

    The model is loaded on first synthesize() call and unloaded immediately
    after generation completes. It is never kept resident between calls.
    This follows the same lazy-load pattern as PromptEnhancer.
    """

    OUTPUT_DIR: Path = Path.home() / ".ltx-desktop" / "audio" / "tts"

    @classmethod
    def is_available(cls) -> bool:
        """Return True if mlx_audio is importable and TTS is possible.

        Returns:
            True when mlx_audio can be imported, False otherwise.
        """
        try:
            from mlx_audio.tts.utils import load_model  # noqa: F401

            return True
        except ImportError:
            return False

    def synthesize(self, text: str, voice: str = "default", speed: float = 1.0) -> str:
        """Synthesize speech from text and write it to a WAV file.

        Lazy-loads the Kokoro TTS model, generates audio, saves to WAV,
        then immediately unloads the model and frees Metal memory.

        Args:
            text: The text to synthesise into speech.
            voice: Voice identifier (see list_voices()). Mapped to Kokoro
                voice preset. Unknown voices fall back to "af_heart".
            speed: Playback speed multiplier (1.0 = normal, 0.5-2.0 range).

        Returns:
            Absolute path to the generated WAV file as a string.

        Raises:
            RuntimeError: If mlx-audio is not installed or model fails to load.
        """
        log.info("TTSEngine.synthesize: text=%r, voice=%s, speed=%.2f", text[:60], voice, speed)

        if not self.is_available():
            raise RuntimeError(
                "mlx-audio not installed. Run: uv add mlx-audio"
            )

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_id = str(uuid.uuid4())[:8]
        output_path = self.OUTPUT_DIR / f"tts_{output_id}.wav"

        # Resolve voice name to Kokoro voice ID.
        kokoro_voice = _VOICE_MAP.get(voice, voice)
        # If the voice isn't in our map and doesn't look like a raw Kokoro ID,
        # fall back to default.
        if voice not in _VOICE_MAP and "_" not in voice:
            kokoro_voice = "af_heart"
            log.warning("TTSEngine: unknown voice %r, falling back to af_heart", voice)

        lang_code = _voice_to_lang_code(kokoro_voice)

        # Import inside method — only when mlx-audio is confirmed available.
        from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]
        import mlx.core as mx  # noqa: F811
        import soundfile as sf

        model = None
        try:
            log.info("TTSEngine: loading Kokoro model %s", _KOKORO_MODEL_ID)
            model = load_model(_KOKORO_MODEL_ID)
            log.info("TTSEngine: model loaded — generating speech (voice=%s, lang=%s, speed=%.2f)",
                      kokoro_voice, lang_code, speed)

            # Kokoro generate() yields results (supports streaming).
            # Collect all audio chunks and concatenate.
            audio_chunks: list[np.ndarray] = []
            sample_rate: int = 24000  # Kokoro default

            for result in model.generate(
                text=text,
                voice=kokoro_voice,
                speed=speed,
                lang_code=lang_code,
            ):
                # result.audio is an mx.array, convert to numpy for saving.
                audio_np = np.array(result.audio, copy=False)
                audio_chunks.append(audio_np)
                if hasattr(result, "sample_rate") and result.sample_rate:
                    sample_rate = int(result.sample_rate)

            if not audio_chunks:
                raise RuntimeError("TTSEngine: Kokoro produced no audio output")

            # Concatenate all chunks into a single waveform.
            full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

            # Ensure 1D float32 for soundfile.
            if full_audio.ndim > 1:
                full_audio = full_audio.squeeze()
            full_audio = full_audio.astype(np.float32)

            # Write WAV file.
            sf.write(str(output_path), full_audio, sample_rate)

            duration = len(full_audio) / sample_rate
            log.info("TTSEngine: WAV written to %s (duration=%.2fs, sr=%d, samples=%d)",
                      output_path, duration, sample_rate, len(full_audio))

        except Exception:
            log.exception("TTSEngine: failed to generate speech")
            raise
        finally:
            # Always unload model and free memory, even on error.
            if model is not None:
                del model
            aggressive_cleanup()
            log.info("TTSEngine: model unloaded and memory freed")

        return str(output_path)

    def list_voices(self) -> list[str]:
        """Return the list of supported voice identifiers.

        Returns:
            List of voice name strings. The first entry is the default.
            Names are user-facing identifiers that map to Kokoro presets.
        """
        return list(_VOICE_MAP.keys())
