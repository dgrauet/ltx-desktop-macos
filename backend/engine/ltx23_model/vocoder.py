"""HiFi-GAN / BigVGAN v2 vocoder for LTX-2.3.

Converts mel spectrograms to audio waveforms. Includes:
- Main vocoder: mel (B, 2, T, 64) → waveform (B, 2, T_audio) at 16 kHz
- BWE (bandwidth extension): 16 kHz → 48 kHz via residual + resampler
- MelSTFT: waveform → log-mel spectrogram (for BWE input)

The main vocoder uses BigVGAN v2 architecture with anti-aliased
SnakeBeta activations and kaiser-sinc resampling filters.

Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

_materialize = mx.eval

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# Core convolution modules
# ---------------------------------------------------------------------------


class Conv1d(nn.Module):
    """1D convolution with dilation support.

    Weight format: (out_C, K, in_C) — MLX format.
    Input/output: (B, C, T).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, self.padding)])
        # (B, C, T) → (B, T, C) for MLX conv1d
        x = x.transpose(0, 2, 1)
        out = mx.conv1d(x, self.weight, stride=self.stride, dilation=self.dilation)
        out = out.transpose(0, 2, 1)
        return out + self.bias[None, :, None]


class ConvTranspose1d(nn.Module):
    """1D transposed convolution for upsampling.

    Weight format: (out_C, K, in_C) — MLX format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)  # (B, C, T) → (B, T, C)
        out = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)
        out = out.transpose(0, 2, 1)  # (B, T, C) → (B, C, T)
        return out + self.bias[None, :, None]


# ---------------------------------------------------------------------------
# Activations and anti-aliased resampling
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """Snake activation: x + (1/exp(beta)) * sin(x * exp(alpha))^2."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha[None, :, None])
        beta = mx.exp(self.beta[None, :, None])
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(x * alpha), 2)


def _kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """Compute kaiser-windowed sinc filter. Returns (1, 1, K)."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    amp = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amp > 50.0:
        beta = 0.1102 * (amp - 8.7)
    elif amp >= 21.0:
        beta = 0.5842 * (amp - 21) ** 0.4 + 0.07886 * (amp - 21.0)
    else:
        beta = 0.0

    window = np.kaiser(kernel_size, beta)
    time = (np.arange(-half_size, half_size) + 0.5) if even else (np.arange(kernel_size) - half_size)

    if cutoff == 0:
        filt = np.zeros_like(time)
    else:
        x = 2 * cutoff * time
        sinc = np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))
        filt = 2 * cutoff * window * sinc
        filt /= filt.sum()

    return mx.array(filt.reshape(1, 1, kernel_size).astype(np.float32))


def _replicate_pad_1d(x: mx.array, left: int, right: int) -> mx.array:
    """Edge-replicate padding for (B, C, T)."""
    parts = []
    if left > 0:
        parts.append(mx.repeat(x[:, :, :1], left, axis=2))
    parts.append(x)
    if right > 0:
        parts.append(mx.repeat(x[:, :, -1:], right, axis=2))
    return mx.concatenate(parts, axis=2)


def _depthwise_conv1d(x: mx.array, filt: mx.array, stride: int = 1) -> mx.array:
    """Depthwise 1D conv: reshape to (B*C, T, 1), convolve, reshape back."""
    b, c, t = x.shape
    x_flat = x.reshape(b * c, t, 1)
    k = filt.shape[2]
    w = filt.reshape(1, k, 1)
    out = mx.conv1d(x_flat, w, stride=stride)
    return out.reshape(b, c, -1)


def _depthwise_conv_transpose1d(x: mx.array, filt: mx.array, stride: int = 1) -> mx.array:
    """Depthwise transposed 1D conv."""
    b, c, t = x.shape
    x_flat = x.reshape(b * c, t, 1)
    k = filt.shape[2]
    w = filt.reshape(1, k, 1)
    out = mx.conv_transpose1d(x_flat, w, stride=stride)
    return out.reshape(b, c, -1)


class LowPassFilter1d(nn.Module):
    """Low-pass filter using depthwise conv with kaiser sinc kernel."""

    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6,
                 stride: int = 1, kernel_size: int = 12):
        super().__init__()
        self.kernel_size = kernel_size
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = _kaiser_sinc_filter(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = _replicate_pad_1d(x, self.pad_left, self.pad_right)
        return _depthwise_conv1d(x, self.filter, stride=self.stride)


class UpSample1d(nn.Module):
    """Anti-aliased upsampling via transposed depthwise conv + sinc filter."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None,
                 window_type: str = "kaiser"):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            rolloff = 0.99
            width = math.ceil(6 / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = np.arange(self.kernel_size) / ratio - width
            t_r = time_axis * rolloff
            t_c = np.clip(t_r, -6, 6)
            window = np.cos(t_c * math.pi / 6 / 2) ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                sinc_vals = np.where(t_r == 0, 1.0, np.sin(np.pi * t_r) / (np.pi * t_r))
            sinc_filter = (sinc_vals * window * rolloff / ratio).reshape(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            sinc_filter = _kaiser_sinc_filter(
                0.5 / ratio, 0.6 / ratio, self.kernel_size
            )
            sinc_filter = np.array(sinc_filter).reshape(1, 1, -1)

        self.filter = mx.array(sinc_filter.astype(np.float32))

    def __call__(self, x: mx.array) -> mx.array:
        x = _replicate_pad_1d(x, self.pad, self.pad)
        x = self.ratio * _depthwise_conv_transpose1d(x, self.filter, stride=self.stride)
        return x[:, :, self.pad_left: x.shape[2] - self.pad_right]


class DownSample1d(nn.Module):
    """Anti-aliased downsampling via low-pass filter with stride."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=ks
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample → activate → downsample."""

    def __init__(self, activation: nn.Module, up_ratio: int = 2, down_ratio: int = 2,
                 up_kernel_size: int = 12, down_kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class AMPBlock1(nn.Module):
    """BigVGAN v2 residual block with anti-aliased SnakeBeta activations."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.acts1 = []
        self.acts2 = []

        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs1.append(Conv1d(channels, channels, kernel_size, padding=pad, dilation=d))
            self.convs2.append(Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
            self.acts1.append(Activation1d(SnakeBeta(channels)))
            self.acts2.append(Activation1d(SnakeBeta(channels)))

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x


# ---------------------------------------------------------------------------
# STFT for BWE
# ---------------------------------------------------------------------------


class _STFTFn(nn.Module):
    """STFT via conv1d with precomputed DFT basis (loaded from checkpoint)."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        # Buffers loaded from checkpoint — MLX format: (n_freqs*2, K, 1)
        self.forward_basis = mx.zeros((n_freqs * 2, filter_length, 1))
        self.inverse_basis = mx.zeros((n_freqs * 2, filter_length, 1))

    def __call__(self, y: mx.array) -> tuple[mx.array, mx.array]:
        """Compute magnitude and phase from waveform (B, T)."""
        if y.ndim == 2:
            y = y[:, None, :]  # (B, 1, T)

        # Causal left-only padding
        left_pad = max(0, self.win_length - self.hop_length)
        if left_pad > 0:
            y = mx.pad(y, [(0, 0), (0, 0), (left_pad, 0)])

        # (B, 1, T) → (B, T, 1) for MLX conv1d
        y_t = y.transpose(0, 2, 1)
        spec = mx.conv1d(y_t, self.forward_basis, stride=self.hop_length)
        spec = spec.transpose(0, 2, 1)  # (B, n_freqs*2, T_frames)

        n_freqs = spec.shape[1] // 2
        real = spec[:, :n_freqs]
        imag = spec[:, n_freqs:]
        magnitude = mx.sqrt(real ** 2 + imag ** 2)
        phase = mx.arctan2(imag, real)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Log-mel spectrogram with precomputed mel basis (from checkpoint)."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int, n_mel: int):
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel, n_freqs))

    def mel_spectrogram(self, y: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Compute log-mel. Returns (log_mel, magnitude, phase, energy)."""
        magnitude, phase = self.stft_fn(y)
        energy = mx.sqrt((magnitude ** 2).sum(axis=1))
        mel = mx.einsum("mf,bft->bmt", self.mel_basis, magnitude)
        log_mel = mx.log(mx.clip(mel, a_min=1e-5, a_max=None))
        return log_mel, magnitude, phase, energy


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------


class Vocoder(nn.Module):
    """HiFi-GAN / BigVGAN v2 vocoder: mel → waveform.

    Input: (B, 2, T, mel_bins) stereo mel spectrogram.
    Output: (B, 2, T_audio) stereo waveform.
    """

    def __init__(
        self,
        upsample_initial_channel: int = 1536,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        stereo: bool = True,
        use_tanh_at_final: bool = False,
        use_bias_at_final: bool = False,
        apply_final_activation: bool = True,
    ):
        super().__init__()

        if upsample_rates is None:
            upsample_rates = [5, 2, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.upsample_factor = math.prod(upsample_rates)

        in_channels = 128 if stereo else 64  # 2 × 64 mel bins

        self.conv_pre = Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # Upsample stages
        self.ups = []
        for i, (rate, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            padding = (k - rate) // 2
            self.ups.append(ConvTranspose1d(in_ch, out_ch, k, rate, padding))

        # AMP residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, tuple(dilations)))

        # Post-activation
        final_ch = upsample_initial_channel // (2 ** self.num_upsamples)
        self.act_post = Activation1d(SnakeBeta(final_ch))

        # Output conv
        out_ch = 2 if stereo else 1
        self.conv_post = Conv1d(final_ch, out_ch, 7, padding=3)
        if not use_bias_at_final:
            self.conv_post.bias = mx.zeros((out_ch,))

    def __call__(self, x: mx.array) -> mx.array:
        """Convert mel spectrogram to waveform.

        Args:
            x: (B, 2, T, mel_bins) stereo mel.

        Returns:
            (B, 2, T_audio) waveform.
        """
        # (B, 2, T, 64) → (B, 2, 64, T) → (B, 128, T)
        x = x.transpose(0, 1, 3, 2)
        b, s, m, t = x.shape
        x = x.reshape(b, s * m, t)

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)

            # Multi-receptive field fusion
            start = i * self.num_kernels
            block_outs = [self.resblocks[start + j](x) for j in range(self.num_kernels)]
            x = mx.stack(block_outs, axis=0).mean(axis=0)
            _materialize(x)

        x = self.act_post(x)
        x = self.conv_post(x)
        x = mx.tanh(x)

        return x


# ---------------------------------------------------------------------------
# VocoderWithBWE — bandwidth extension wrapper
# ---------------------------------------------------------------------------


class VocoderWithBWE(nn.Module):
    """Vocoder + bandwidth extension: mel → 16 kHz → 48 kHz waveform."""

    def __init__(
        self,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
        input_sr: int = 16000,
        output_sr: int = 48000,
        hop_length: int = 80,
    ):
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.output_sample_rate = output_sr
        self.hop_length = hop_length
        self.resampler = UpSample1d(ratio=output_sr // input_sr, window_type="hann")

    def _compute_mel(self, audio: mx.array) -> mx.array:
        """Compute log-mel from waveform (B, C, T) → (B, C, n_mels, T_frames)."""
        b, n_ch, t = audio.shape
        flat = audio.reshape(b * n_ch, t)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(b, n_ch, mel.shape[1], mel.shape[2])

    def __call__(self, mel_spec: mx.array) -> mx.array:
        """Run vocoder + BWE.

        Args:
            mel_spec: (B, 2, T, mel_bins) stereo mel.

        Returns:
            (B, 2, T_out) waveform clipped to [-1, 1] at output_sr.
        """
        x = self.vocoder(mel_spec)
        _materialize(x)

        _, _, length_16k = x.shape
        output_length = length_16k * self.output_sr // self.input_sr

        # Pad to multiple of hop_length
        remainder = length_16k % self.hop_length
        if remainder != 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, self.hop_length - remainder)])

        mel = self._compute_mel(x)  # (B, C, n_mels, T_frames)
        mel_for_bwe = mel.transpose(0, 1, 3, 2)  # (B, C, T_frames, n_mels)

        residual = self.bwe_generator(mel_for_bwe)
        skip = self.resampler(x)

        return mx.clip(residual + skip, -1, 1)[:, :, :output_length]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_vocoder(model_dir: Path) -> VocoderWithBWE:
    """Load vocoder + BWE from split safetensors and config.

    Args:
        model_dir: Directory containing vocoder.safetensors and embedded_config.json.

    Returns:
        VocoderWithBWE ready for inference.
    """
    weights_path = model_dir / "vocoder.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"vocoder.safetensors not found in {model_dir}")

    # Read config — vocoder config is nested: cfg["vocoder"]["vocoder"] and cfg["vocoder"]["bwe"]
    config_path = model_dir / "embedded_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        vocoder_section = cfg.get("vocoder", {})
        voc_cfg = vocoder_section.get("vocoder", {})
        bwe_cfg = vocoder_section.get("bwe", {})
    else:
        voc_cfg = {}
        bwe_cfg = {}

    # Build main vocoder
    vocoder = Vocoder(
        upsample_initial_channel=voc_cfg.get("upsample_initial_channel", 1536),
        upsample_rates=voc_cfg.get("upsample_rates", [5, 2, 2, 2, 2, 2]),
        upsample_kernel_sizes=voc_cfg.get("upsample_kernel_sizes", [11, 4, 4, 4, 4, 4]),
        resblock_kernel_sizes=voc_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilation_sizes=voc_cfg.get("resblock_dilation_sizes", [[1, 3, 5]] * 3),
        stereo=voc_cfg.get("stereo", True),
        use_tanh_at_final=voc_cfg.get("use_tanh_at_final", False),
        use_bias_at_final=voc_cfg.get("use_bias_at_final", False),
    )

    # Build BWE generator
    bwe = Vocoder(
        upsample_initial_channel=bwe_cfg.get("upsample_initial_channel", 512),
        upsample_rates=bwe_cfg.get("upsample_rates", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=bwe_cfg.get("upsample_kernel_sizes", [12, 11, 4, 4, 4]),
        resblock_kernel_sizes=bwe_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilation_sizes=bwe_cfg.get("resblock_dilation_sizes", [[1, 3, 5]] * 3),
        stereo=bwe_cfg.get("stereo", True),
        use_tanh_at_final=bwe_cfg.get("use_tanh_at_final", False),
        use_bias_at_final=bwe_cfg.get("use_bias_at_final", False),
        apply_final_activation=bwe_cfg.get("apply_final_activation", False),
    )

    # Build MelSTFT
    mel_stft = MelSTFT(
        filter_length=bwe_cfg.get("n_fft", 512),
        hop_length=bwe_cfg.get("hop_length", 80),
        win_length=bwe_cfg.get("win_size", 512),
        n_mel=bwe_cfg.get("num_mels", 64),
    )

    # Build wrapper
    wrapper = VocoderWithBWE(
        vocoder=vocoder,
        bwe_generator=bwe,
        mel_stft=mel_stft,
        input_sr=bwe_cfg.get("input_sampling_rate", 16000),
        output_sr=bwe_cfg.get("output_sampling_rate", 48000),
        hop_length=bwe_cfg.get("hop_length", 80),
    )

    # Load weights
    # Key structure in file: vocoder.conv_pre.weight (main),
    # vocoder.bwe_generator.conv_pre.weight (BWE),
    # vocoder.mel_stft.mel_basis (STFT)
    raw = mx.load(str(weights_path))

    _load_vocoder_weights(raw, vocoder, "vocoder")
    _load_vocoder_weights(raw, bwe, "vocoder.bwe_generator")
    _load_mel_stft_weights(raw, mel_stft, "vocoder.mel_stft")

    loaded = len(raw)
    log.info(
        "Vocoder+BWE loaded: %d keys, %.2f GB",
        loaded, mx.get_active_memory() / (1024**3),
    )

    return wrapper


def _load_vocoder_weights(raw: dict, vocoder: Vocoder, prefix: str) -> None:
    """Load weights for a single Vocoder instance."""
    _load_conv1d(raw, f"{prefix}.conv_pre", vocoder.conv_pre)

    for i, up in enumerate(vocoder.ups):
        _load_conv_transpose1d(raw, f"{prefix}.ups.{i}", up)

    for i, block in enumerate(vocoder.resblocks):
        bp = f"{prefix}.resblocks.{i}"
        for j, conv in enumerate(block.convs1):
            _load_conv1d(raw, f"{bp}.convs1.{j}", conv)
        for j, conv in enumerate(block.convs2):
            _load_conv1d(raw, f"{bp}.convs2.{j}", conv)
        for j, act in enumerate(block.acts1):
            _load_activation1d(raw, f"{bp}.acts1.{j}", act)
        for j, act in enumerate(block.acts2):
            _load_activation1d(raw, f"{bp}.acts2.{j}", act)

    if vocoder.act_post is not None:
        _load_activation1d(raw, f"{prefix}.act_post", vocoder.act_post)

    _load_conv1d(raw, f"{prefix}.conv_post", vocoder.conv_post)


def _load_mel_stft_weights(raw: dict, mel_stft: MelSTFT, prefix: str) -> None:
    """Load MelSTFT buffers (forward_basis, inverse_basis, mel_basis)."""
    for buf in ("forward_basis", "inverse_basis"):
        key = f"{prefix}.stft_fn.{buf}"
        if key in raw:
            # Stored as PyTorch (n_freqs*2, 1, K) — transpose to MLX (n_freqs*2, K, 1)
            w = raw[key]
            if w.ndim == 3 and w.shape[1] == 1:
                # Already in (out, 1, K) PyTorch format → transpose to (out, K, 1)
                w = w.transpose(0, 2, 1)
            setattr(mel_stft.stft_fn, buf, w)

    key = f"{prefix}.mel_basis"
    if key in raw:
        mel_stft.mel_basis = raw[key]


def _load_conv1d(raw: dict, prefix: str, conv: Conv1d) -> None:
    """Load Conv1d weights (already in MLX O,K,I format from conversion)."""
    for suffix in ("weight", "bias"):
        key = f"{prefix}.{suffix}"
        if key in raw:
            setattr(conv, suffix, raw[key])


def _load_conv_transpose1d(raw: dict, prefix: str, conv: ConvTranspose1d) -> None:
    """Load ConvTranspose1d weights, handling potential format differences.

    The conversion script should transpose (I,O,K) → (O,K,I), but some
    files may still have PyTorch format. We detect and handle both.
    """
    key_w = f"{prefix}.weight"
    key_b = f"{prefix}.bias"

    if key_w in raw:
        w = raw[key_w]
        # Check if weight needs transpose: in PyTorch format (I,O,K),
        # dim 0 = in_channels > out_channels = dim 1. In MLX format (O,K,I),
        # dim 0 = out_channels. Use bias shape to determine out_channels.
        if key_b in raw:
            expected_out = raw[key_b].shape[0]
            if w.shape[0] != expected_out and w.shape[1] == expected_out:
                # PyTorch format (I, O, K) → MLX (O, K, I)
                w = w.transpose(1, 2, 0)
        conv.weight = w

    if key_b in raw:
        conv.bias = raw[key_b]


def _load_activation1d(raw: dict, prefix: str, act: Activation1d) -> None:
    """Load Activation1d (SnakeBeta + upsample/downsample filters)."""
    for param in ("alpha", "beta"):
        key = f"{prefix}.act.{param}"
        if key in raw:
            setattr(act.act, param, raw[key])

    # Upsample filter
    key = f"{prefix}.upsample.filter"
    if key in raw:
        act.upsample.filter = raw[key]

    # Downsample lowpass filter
    key = f"{prefix}.downsample.lowpass.filter"
    if key in raw:
        act.downsample.lowpass.filter = raw[key]
