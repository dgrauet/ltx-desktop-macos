"""Audio VAE decoder for LTX-2.3.

Decodes audio latent (B, 8, T, 16) → stereo mel spectrogram (B, 2, T', 64).
Architecture: conv_in(8→512) → 2 mid ResBlocks → 3 upsample levels → conv_out(128→2).
Uses causal Conv2d along HEIGHT axis (frequency) with PixelNorm.

Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

_materialize = mx.eval

LATENT_DOWNSAMPLE_FACTOR = 4


def _pixel_norm(x: mx.array, dim: int = 1, eps: float = 1e-6) -> mx.array:
    """Per-pixel RMS normalization along channel dimension."""
    rms = mx.sqrt(mx.mean(x * x, axis=dim, keepdims=True) + eps)
    return x / rms


class CausalConv2d(nn.Module):
    """2D convolution with causal padding along HEIGHT (frequency axis).

    Weight format: (out_C, kH, kW, in_C) — MLX format.
    Causal along height: pad top only. Symmetric along width.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((out_channels, kernel_size, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

        # Pre-compute padding: causal along height, symmetric along width
        pad_h = kernel_size - 1
        pad_w = kernel_size - 1
        # (batch, channels, height, width)
        self._padding = [
            (0, 0), (0, 0),
            (pad_h, 0),  # causal: all padding on top
            (pad_w // 2, pad_w - pad_w // 2),  # symmetric
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, self._padding)
        # (B, C, H, W) → (B, H, W, C) for MLX conv2d
        x = x.transpose(0, 2, 3, 1)
        out = mx.conv2d(x, self.weight)
        # (B, H, W, C) → (B, C, H, W)
        out = out.transpose(0, 3, 1, 2)
        return out + self.bias[None, :, None, None]


class SimpleResBlock2d(nn.Module):
    """Residual block with PixelNorm normalization."""

    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = CausalConv2d(in_channels, out_channels)
        self.conv2 = CausalConv2d(out_channels, out_channels)
        if in_channels != out_channels:
            self.skip = CausalConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        h = _pixel_norm(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = _pixel_norm(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class Upsample2d(nn.Module):
    """2D nearest-neighbor upsample + conv, drops first row (causal axis)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv2d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        b, c, h, w = x.shape
        # Nearest neighbor 2×
        x = x[:, :, :, None, :, None]
        x = mx.broadcast_to(x, (b, c, h, 2, w, 2))
        x = x.reshape(b, c, h * 2, w * 2)
        x = self.conv(x)
        # Drop first row to undo encoder's causal padding
        x = x[:, :, 1:, :]
        return x


class AudioDecoder(nn.Module):
    """Audio VAE decoder: latent (B, 8, T, 16) → mel (B, 2, T', 64).

    Architecture (3 upsample levels, ch_mult=[1, 2, 4]):
    - conv_in: 8 → 512
    - mid: 2× ResBlock(512)
    - level 2: 3× ResBlock(512→512), upsample
    - level 1: 3× ResBlock(512→256), upsample
    - level 0: 3× ResBlock(256→128)
    - norm_out + silu + conv_out: 128 → 2
    """

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 3,
        z_channels: int = 8,
        mel_bins: int = 16,
    ):
        super().__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.z_channels = z_channels
        self.mel_bins = mel_bins

        # Per-channel statistics (128 = z_channels × mel_bins)
        self.mean_of_means = mx.zeros((ch,))
        self.std_of_means = mx.ones((ch,))

        base_ch = ch * ch_mult[-1]  # 512

        self.conv_in = CausalConv2d(z_channels, base_ch)

        # Mid block
        self.mid_block_1 = SimpleResBlock2d(base_ch)
        self.mid_block_2 = SimpleResBlock2d(base_ch)

        # Up blocks (reverse order of ch_mult)
        self.up_res_blocks: list[list[SimpleResBlock2d]] = []
        self.up_upsamplers: list[Upsample2d | None] = []
        block_in = base_ch

        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append(SimpleResBlock2d(block_in, block_out))
                block_in = block_out
            self.up_res_blocks.append(res_blocks)
            self.up_upsamplers.append(
                Upsample2d(block_out) if i_level != 0 else None
            )

        self.conv_out = CausalConv2d(ch, out_ch)

    def _denormalize_latents(self, x: mx.array) -> mx.array:
        """Denormalize: patchify → denorm → unpatchify.

        x: (B, z_ch, T, mel_bins) → flatten to (B, T, z_ch*mel_bins) → denorm → unflatten.
        """
        b, c, t, f = x.shape
        # Patchify: (B, C, T, F) → (B, T, C*F)
        x = x.transpose(0, 2, 1, 3).reshape(b, t, c * f)
        # Denormalize
        mean = self.mean_of_means[None, None, :]
        std = self.std_of_means[None, None, :]
        x = x * std + mean
        # Unpatchify: (B, T, C*F) → (B, C, T, F)
        x = x.reshape(b, t, c, f).transpose(0, 2, 1, 3)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Decode audio latent to mel spectrogram.

        Args:
            x: (B, 8, T, 16) audio latent.

        Returns:
            (B, 2, T', 64) stereo mel spectrogram.
        """
        x = x.astype(mx.float32)
        x = self._denormalize_latents(x)

        _b, _c, t, f = x.shape
        target_frames = t * LATENT_DOWNSAMPLE_FACTOR
        # Causal: subtract (factor - 1) for padding compensation
        target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        h = self.conv_in(x)
        _materialize(h)

        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        _materialize(h)

        for res_blocks, upsample in zip(self.up_res_blocks, self.up_upsamplers):
            for block in res_blocks:
                h = block(h)
            if upsample is not None:
                h = upsample(h)
            _materialize(h)

        h = _pixel_norm(h)
        h = nn.silu(h)
        h = self.conv_out(h)

        target_mel = f * LATENT_DOWNSAMPLE_FACTOR  # 16 * 4 = 64
        h = h[:, :self.out_ch, :target_frames, :target_mel]

        return h


def load_audio_decoder(model_dir: Path) -> AudioDecoder:
    """Load audio VAE decoder from split safetensors.

    Args:
        model_dir: Directory containing audio_vae.safetensors.

    Returns:
        Loaded AudioDecoder.
    """
    weights_path = model_dir / "audio_vae.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"audio_vae.safetensors not found in {model_dir}")

    raw = mx.load(str(weights_path))
    prefix = "audio_vae."

    decoder = AudioDecoder()

    # Per-channel statistics
    mean_key = f"{prefix}per_channel_statistics._mean_of_means"
    std_key = f"{prefix}per_channel_statistics._std_of_means"
    if mean_key in raw:
        decoder.mean_of_means = raw[mean_key]
    if std_key in raw:
        decoder.std_of_means = raw[std_key]

    # conv_in (weights already in MLX format from conversion)
    _load_conv2d(raw, f"{prefix}conv_in.conv", decoder.conv_in)

    # Mid blocks
    _load_resblock(raw, f"{prefix}mid.block_1", decoder.mid_block_1)
    _load_resblock(raw, f"{prefix}mid.block_2", decoder.mid_block_2)

    # Up blocks — map our sequential index to PyTorch level index (reversed)
    for i, (res_blocks, upsample) in enumerate(
        zip(decoder.up_res_blocks, decoder.up_upsamplers)
    ):
        pt_level = decoder.num_resolutions - 1 - i
        for j, block in enumerate(res_blocks):
            _load_resblock(raw, f"{prefix}up.{pt_level}.block.{j}", block)
        if upsample is not None:
            _load_conv2d(
                raw, f"{prefix}up.{pt_level}.upsample.conv.conv", upsample.conv
            )

    # conv_out
    _load_conv2d(raw, f"{prefix}conv_out.conv", decoder.conv_out)

    loaded = sum(1 for k in raw if k.startswith(prefix))
    log.info(
        "Audio VAE decoder loaded: %d keys, %.2f GB",
        loaded, mx.get_active_memory() / (1024**3),
    )

    return decoder


def _load_conv2d(raw: dict, prefix: str, conv: CausalConv2d) -> None:
    """Load CausalConv2d weights (already in MLX O,kH,kW,I format)."""
    for suffix in ("weight", "bias"):
        key = f"{prefix}.{suffix}"
        if key in raw:
            setattr(conv, suffix, raw[key])


def _load_resblock(raw: dict, prefix: str, block: SimpleResBlock2d) -> None:
    """Load SimpleResBlock2d weights."""
    _load_conv2d(raw, f"{prefix}.conv1.conv", block.conv1)
    _load_conv2d(raw, f"{prefix}.conv2.conv", block.conv2)
    if block.skip is not None:
        _load_conv2d(raw, f"{prefix}.nin_shortcut.conv", block.skip)
