"""BigVGAN v2 vocoder for LTX-2.3 — MLX port of Lightricks ltx-core reference.

Architecture: BigVGAN v2 with SnakeBeta + anti-aliased Activation1d.
Internal convention: channels-first (B, C, T), transposed to (B, T, C) for MLX conv ops.
"""
import logging
import math
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

log = logging.getLogger(__name__)

def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """Compute a kaiser-windowed sinc lowpass filter."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0

    # Use numpy for kaiser window (MLX doesn't have it)
    window = np.kaiser(kernel_size, beta).astype(np.float32)
    if even:
        time = np.arange(-half_size, half_size, dtype=np.float32) + 0.5
    else:
        time = np.arange(kernel_size, dtype=np.float32) - half_size

    if cutoff == 0:
        filt = np.zeros_like(time)
    else:
        filt = 2 * cutoff * window * np.sinc(2 * cutoff * time)
        filt /= filt.sum()

    return mx.array(filt.reshape(1, 1, kernel_size))


# --- Anti-aliased resampling ---

class LowPassFilter1d(nn.Module):
    """Kaiser-sinc lowpass filter via depthwise conv."""

    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6,
                 stride: int = 1, kernel_size: int = 12) -> None:
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        # (1, 1, K) — expanded to (C, K, 1) on first call
        self.filter = _kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self._filter_expanded: mx.array | None = None

    def _get_filter(self, channels: int) -> mx.array:
        if self._filter_expanded is None or self._filter_expanded.shape[0] != channels:
            self._filter_expanded = mx.broadcast_to(
                self.filter, (channels, 1, self.filter.shape[-1])
            ).transpose(0, 2, 1)  # (C, K, 1)
        return self._filter_expanded

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) channels-first
        c = x.shape[1]
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad_left, self.pad_right)], mode="edge")
        x = x.transpose(0, 2, 1)  # (B, T, C) for MLX conv1d
        out = mx.conv1d(x, self._get_filter(c), stride=self.stride, groups=c)
        return out.transpose(0, 2, 1)  # (B, C, T_out)


class UpSample1d(nn.Module):
    """Upsample by ratio using kaiser-sinc interpolation."""

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None) -> None:
        super().__init__()
        self.ratio = ratio
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.pad = ks // ratio - 1
        self.pad_left = self.pad * ratio + (ks - ratio) // 2
        self.pad_right = self.pad * ratio + (ks - ratio + 1) // 2
        self.filter = _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, ks)
        self._filter_expanded: mx.array | None = None

    def _get_filter(self, channels: int) -> mx.array:
        if self._filter_expanded is None or self._filter_expanded.shape[0] != channels:
            self._filter_expanded = mx.broadcast_to(
                self.filter, (channels, 1, self.filter.shape[-1])
            ).transpose(0, 2, 1)  # (C, K, 1)
        return self._filter_expanded

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) channels-first
        c = x.shape[1]
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad, self.pad)], mode="edge")
        x = x.transpose(0, 2, 1)  # (B, T, C) for MLX conv_transpose1d
        out = mx.conv_transpose1d(x, self._get_filter(c), stride=self.ratio, groups=c)
        out = out * self.ratio
        out = out.transpose(0, 2, 1)  # (B, C, T_out)
        end = out.shape[2] - self.pad_right if self.pad_right else out.shape[2]
        return out[:, :, self.pad_left:end]


class DownSample1d(nn.Module):
    """Downsample by ratio using kaiser-sinc lowpass filter."""

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None) -> None:
        super().__init__()
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio,
            stride=ratio, kernel_size=ks,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


# --- SnakeBeta activation ---

class SnakeBeta(nn.Module):
    """SnakeBeta: x + (1/beta) * sin(alpha * x)^2. Alpha/beta in log-space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) channels-first
        alpha = mx.exp(self.alpha)[None, :, None]  # (1, C, 1)
        beta = mx.exp(self.beta)[None, :, None]
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample → SnakeBeta → downsample."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


# --- AMPBlock1 (Adaptive Multi-Period) ---

class AMPBlock1(nn.Module):
    """Residual block with dilated convolutions and anti-aliased SnakeBeta."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1,
                      dilation=d, padding=_get_padding(kernel_size, d))
            for d in dilation
        ]
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, stride=1,
                      dilation=1, padding=_get_padding(kernel_size, 1))
            for _ in dilation
        ]
        self.acts1 = [Activation1d(channels) for _ in dilation]
        self.acts2 = [Activation1d(channels) for _ in dilation]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) channels-first
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            # Transpose for MLX conv: (B, C, T) -> (B, T, C)
            xt = a1(x)
            xt_t = xt.transpose(0, 2, 1)
            xt_t = c1(xt_t)
            xt = xt_t.transpose(0, 2, 1)  # back to (B, C, T)

            xt = a2(xt)
            xt_t = xt.transpose(0, 2, 1)
            xt_t = c2(xt_t)
            xt = xt_t.transpose(0, 2, 1)

            x = x + xt
        return x


# --- Vocoder ---

class Vocoder(nn.Module):
    """BigVGAN v2 vocoder: mel spectrogram → waveform.

    Architecture matches embedded_config.json vocoder section.
    Stereo output, no tanh at final, clamp [-1, 1].
    """

    def __init__(
        self,
        upsample_rates: List[int] = None,
        upsample_kernel_sizes: List[int] = None,
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
    ) -> None:
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [5, 2, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Input: 128 channels (2 stereo × 64 mel bins)
        self.conv_pre = nn.Conv1d(128, upsample_initial_channel, kernel_size=7,
                                  stride=1, padding=3)

        # Upsample layers (ConvTranspose1d)
        self.ups = []
        ch = upsample_initial_channel
        for i, (rate, ksize) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = ch // 2
            self.ups.append(
                nn.ConvTranspose1d(ch, out_ch, kernel_size=ksize,
                                   stride=rate, padding=(ksize - rate) // 2)
            )
            ch = out_ch

        # Residual blocks
        self.resblocks = []
        for i in range(self.num_upsamples):
            rch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(rch, k, tuple(d)))

        # Final activation + output conv
        final_ch = upsample_initial_channel // (2 ** self.num_upsamples)
        self.act_post = Activation1d(final_ch)
        self.conv_post = nn.Conv1d(final_ch, 2, kernel_size=7, stride=1,
                                   padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Mel spectrogram (B, 2, T, 64) — stereo mel from audio VAE.

        Returns:
            Waveform (B, 2, T_audio) clipped to [-1, 1].
        """
        # Reshape: (B, 2, T, 64) → (B, 2, 64, T) → (B, 128, T)
        x = x.transpose(0, 1, 3, 2)
        b, s, m, t = x.shape
        x = x.reshape(b, s * m, t)

        # conv_pre: transpose for MLX (B, 128, T) → (B, T, 128) → conv → (B, T, 1536) → (B, 1536, T)
        x = self.conv_pre(x.transpose(0, 2, 1)).transpose(0, 2, 1)

        for i in range(self.num_upsamples):
            # NO leaky_relu for AMP blocks (matches ltx-core reference)
            # ConvTranspose1d: (B, C, T) → (B, T, C) → conv_t → (B, T_up, C/2) → (B, C/2, T_up)
            x_t = x.transpose(0, 2, 1)
            x_t = self.ups[i](x_t)
            x = x_t.transpose(0, 2, 1)

            start = i * self.num_kernels
            xs = self.resblocks[start](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[start + j](x)
            x = xs / self.num_kernels
            # Materialize to bound computation graph (mx.eval = MLX tensor eval)
            mx.eval(x)  # noqa: S307

        x = self.act_post(x)

        # conv_post
        x = self.conv_post(x.transpose(0, 2, 1)).transpose(0, 2, 1)

        return mx.clip(x, -1, 1)


def load_vocoder(model_dir: Path) -> Vocoder:
    """Load the BigVGAN v2 vocoder from split safetensors.

    Weights are stored with 'vocoder.' prefix. Conv1d weights are in MLX format
    (out, K, in). ConvTranspose1d weights are in PyTorch format (in, out, K) and
    need transposing.
    """
    path = model_dir / "vocoder.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"vocoder.safetensors not found in {model_dir}")

    vocoder = Vocoder()
    raw = mx.load(str(path))

    weights = []
    for key, value in raw.items():
        if not key.startswith("vocoder.") or key.startswith("vocoder.bwe") or key.startswith("vocoder.mel"):
            continue
        clean = key[len("vocoder."):]

        # ConvTranspose1d weights stored as PyTorch (in, out, K) — transpose to MLX (out, K, in)
        if "ups." in clean and "weight" in clean and value.ndim == 3:
            value = mx.transpose(value, (1, 2, 0))

        weights.append((clean, value))

    vocoder.load_weights(weights, strict=False)
    log.info("Vocoder: loaded %d weight tensors", len(weights))

    return vocoder
