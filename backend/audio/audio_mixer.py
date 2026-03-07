"""Multi-track audio mixing for video output.

Mixes voiceover TTS, background music, and generated video audio
using ffmpeg filter graphs. Music is ducked to 30% volume (20% with voiceover).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

# Frequency (Hz) used by the music stub per genre.
_GENRE_FREQUENCIES: dict[str, int] = {
    "ambient": 110,
    "electronic": 220,
    "orchestral": 330,
    "jazz": 165,
    "cinematic": 196,
    "upbeat": 262,
    "dark": 98,
    "nature": 146,
}


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


class AudioMixer:
    """Multi-track audio mixer that muxes TTS, music, and video audio via ffmpeg.

    All mixing is performed with ffmpeg filter graphs — no MLX is required.
    The mixed result is written back into a new MP4 container.
    """

    OUTPUT_DIR: Path = Path.home() / ".ltx-desktop" / "audio" / "mixed"
    OUTPUT_DIR_MUSIC: Path = Path.home() / ".ltx-desktop" / "audio" / "music"

    MUSIC_GENRES: list[str] = [
        "ambient",
        "electronic",
        "orchestral",
        "jazz",
        "cinematic",
        "upbeat",
        "dark",
        "nature",
    ]

    def mix(
        self,
        video_path: str,
        tts_path: str | None = None,
        music_path: str | None = None,
        music_volume: float = 0.3,
        tts_volume: float = 1.0,
        output_path: str | None = None,
    ) -> str:
        """Mix audio tracks onto a video and write the result to a new MP4.

        The original video audio is always included. TTS and music are
        optional additional inputs. When both TTS and music are supplied,
        music is ducked to 0.2 (rather than the default music_volume) so
        that speech remains intelligible.

        Args:
            video_path: Absolute path to the source MP4.
            tts_path: Optional absolute path to a TTS WAV file.
            music_path: Optional absolute path to a background music file.
            music_volume: Base music volume (0.0–1.0). Defaults to 0.3.
            tts_volume: TTS volume (0.0–1.0). Defaults to 1.0.
            output_path: Where to write the mixed MP4. Defaults to a
                timestamped file in OUTPUT_DIR.

        Returns:
            Absolute path to the mixed MP4 as a string.

        Raises:
            RuntimeError: If ffmpeg is not found or the mix command fails.
        """
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_id = str(uuid.uuid4())[:8]
            output_path = str(self.OUTPUT_DIR / f"mixed_{output_id}.mp4")

        ffmpeg_bin = _find_ffmpeg()

        # Build input list and filter graph incrementally.
        # Input 0 is always the video (carries the original audio on stream [0:a]).
        inputs: list[str] = ["-i", video_path]
        audio_labels: list[str] = ["[0:a]"]
        filter_parts: list[str] = []

        input_index = 1

        if tts_path:
            inputs += ["-i", tts_path]
            label = f"[{input_index}:a]"
            scaled_label = f"[tts_scaled]"
            filter_parts.append(f"{label}volume={tts_volume:.4f}{scaled_label}")
            audio_labels.append(scaled_label)
            input_index += 1

        if music_path:
            # Duck music when TTS is also present
            effective_music_vol = 0.2 if tts_path else music_volume
            inputs += ["-i", music_path]
            label = f"[{input_index}:a]"
            scaled_label = f"[music_scaled]"
            filter_parts.append(f"{label}volume={effective_music_vol:.4f}{scaled_label}")
            audio_labels.append(scaled_label)
            input_index += 1

        # Combine all audio streams with amix
        num_inputs = len(audio_labels)
        mix_inputs = "".join(audio_labels)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={num_inputs}:duration=first:dropout_transition=2[aout]"
        )

        filter_graph = "; ".join(filter_parts)
        log.debug("AudioMixer filter graph: %s", filter_graph)

        cmd = [
            ffmpeg_bin,
            "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        log.info(
            "AudioMixer.mix: video=%s tts=%s music=%s → %s",
            video_path,
            tts_path,
            music_path,
            output_path,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mix failed: {result.stderr[:500]}")

        log.info("AudioMixer: mixed output written to %s", output_path)
        return output_path

    def generate_music_stub(self, genre: str = "ambient", duration: float = 10.0) -> str:
        """Generate a placeholder music WAV using a sine wave via ffmpeg.

        Args:
            genre: Music genre name (see MUSIC_GENRES). Controls the sine
                frequency used by the stub.
            duration: Duration of the generated audio in seconds.

        Returns:
            Absolute path to the generated WAV file as a string.

        Raises:
            RuntimeError: If ffmpeg is not found or the command fails.
        """
        self.OUTPUT_DIR_MUSIC.mkdir(parents=True, exist_ok=True)

        frequency = _GENRE_FREQUENCIES.get(genre.lower(), 180)
        output_id = str(uuid.uuid4())[:8]
        output_path = self.OUTPUT_DIR_MUSIC / f"music_{genre}_{output_id}.wav"

        ffmpeg_bin = _find_ffmpeg()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:duration={duration:.3f}",
            "-ar", "44100",
            "-ac", "2",
            str(output_path),
        ]

        log.info(
            "AudioMixer.generate_music_stub: genre=%s freq=%dHz duration=%.2fs → %s",
            genre,
            frequency,
            duration,
            output_path,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg music stub failed: {result.stderr[:500]}")

        log.info("AudioMixer: music stub written to %s", output_path)
        return str(output_path)
