"""VAE video encoder for LTX-2.3.

Encodes pixel video [B, 3, F, H, W] → latent [B, 128, F', H', W'].
Architecture from embedded_config.json: 9 blocks, causal padding,
pixel_norm, patch_size=4, zero spatial padding.

Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

# mx.eval is MLX tensor materialization (NOT Python eval)
_materialize = mx.eval

# LTX-2.3 encoder block config (from embedded_config.json)
_ENCODER_BLOCKS = [
    ["res_x", {"num_layers": 4}],
    ["compress_space_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 6}],
    ["compress_time_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 4}],
    ["compress_all_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 2}],
    ["compress_all_res", {"multiplier": 1}],
    ["res_x", {"num_layers": 2}],
]

# Stride mapping for downsample block types
_STRIDE_MAP = {
    "compress_all_res": (2, 2, 2),
    "compress_time_res": (2, 1, 1),
    "compress_space_res": (1, 2, 2),
}


def _pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Pixel normalization across channels (dim=1)."""
    variance = mx.mean(x * x, axis=1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


def _patchify(x: mx.array, patch_size: int) -> mx.array:
    """Space-to-depth: (B, C, F, H, W) -> (B, C*p*p, F, H/p, W/p)."""
    if patch_size == 1:
        return x
    b, c, f, h, w = x.shape
    p = patch_size
    x = x.reshape(b, c, f, h // p, p, w // p, p)
    x = x.transpose(0, 1, 4, 6, 2, 3, 5)
    x = x.reshape(b, c * p * p, f, h // p, w // p)
    return x


def _space_to_depth(x: mx.array, stride: tuple[int, int, int]) -> mx.array:
    """Rearrange spatial/temporal dims into channels.

    (B, C, D, H, W) -> (B, C*p1*p2*p3, D/p1, H/p2, W/p3)
    """
    b, c, d, h, w = x.shape
    p1, p2, p3 = stride
    x = x.reshape(b, c, d // p1, p1, h // p2, p2, w // p3, p3)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshape(b, c * p1 * p2 * p3, d // p1, h // p2, w // p3)
    return x


class Conv3d(nn.Module):
    """3D convolution via temporal iteration over 2D spatial convolutions.

    Weight format: (out_C, kT, kH, kW, in_C) — MLX-converted format.
    Uses causal temporal padding (replicate first frame only).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        k = kernel_size
        self.kernel_size = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = mx.zeros((out_channels, k, k, k, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 3D conv with causal temporal padding.

        Args:
            x: Input (B, C, T, H, W).

        Returns:
            Output (B, C_out, T, H, W).
        """
        b, c, t, h, w = x.shape
        k = self.kernel_size
        p = k // 2

        # Spatial zero-padding
        if p > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (p, p), (p, p)])

        # Causal temporal padding: replicate first frame on the left only
        if k > 1:
            first = mx.repeat(x[:, :, :1], k - 1, axis=2)
            x = mx.concatenate([first, x], axis=2)

        _, _, t_pad, h_pad, w_pad = x.shape
        h_out = h_pad - k + 1
        w_out = w_pad - k + 1
        t_out = t_pad - k + 1

        output = None
        for kt in range(k):
            w_slice = self.weight[:, kt, :, :, :]
            x_slice = x[:, :, kt:kt + t_out, :, :]

            x_2d = x_slice.transpose(0, 2, 1, 3, 4)
            x_2d = x_2d.reshape(b * t_out, c, h_pad, w_pad)
            x_2d = x_2d.transpose(0, 2, 3, 1)

            conv_out = mx.conv2d(x_2d, w_slice, padding=0)

            _, _, _, c_out = conv_out.shape
            conv_out = conv_out.reshape(b, t_out, h_out, w_out, c_out)
            conv_out = conv_out.transpose(0, 4, 1, 2, 3)

            if output is None:
                output = conv_out
            else:
                output = output + conv_out

        return output + self.bias[None, :, None, None, None]


class ResBlock(nn.Module):
    """Residual block with pixel norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv3d(channels, channels)
        self.conv2 = Conv3d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = _pixel_norm(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = _pixel_norm(x)
        x = nn.silu(x)
        x = self.conv2(x)
        return x + residual


class ResBlockGroup(nn.Module):
    """Group of residual blocks."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.res_blocks = [ResBlock(channels) for _ in range(num_blocks)]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
        return x


class SpaceToDepthDownsample(nn.Module):
    """Downsample via conv then space-to-depth, with group-averaged residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ):
        super().__init__()
        self.stride = stride
        stride_prod = math.prod(stride)
        # Group size for averaging the skip connection
        self.group_size = in_channels * stride_prod // out_channels
        # Conv reduces channels before space-to-depth packs them back up
        self.conv = Conv3d(in_channels, out_channels // stride_prod)

    def __call__(self, x: mx.array) -> mx.array:
        # Causal temporal padding if temporal stride
        if self.stride[0] == 2:
            first = x[:, :, :1, :, :]
            x = mx.concatenate([first, x], axis=2)

        # Skip connection: direct space-to-depth with channel group averaging
        x_in = _space_to_depth(x, self.stride)
        b, c_packed, d, h, w = x_in.shape
        g = self.group_size
        c_out = c_packed // g
        x_in = x_in.reshape(b, c_out, g, d, h, w)
        x_in = mx.mean(x_in, axis=2)

        # Conv path
        x = self.conv(x)
        x = _space_to_depth(x, self.stride)

        return x + x_in


class VideoEncoder(nn.Module):
    """LTX-2.3 VAE video encoder.

    Encodes (B, 3, F, H, W) → normalized latent (B, 128, F', H', W').
    F must be 1 + 8*k (e.g. 1, 9, 17, 25...).
    """

    def __init__(
        self,
        encoder_blocks: list | None = None,
        base_channels: int = 128,
        latent_channels: int = 128,
        patch_size: int = 4,
    ):
        super().__init__()

        if encoder_blocks is None:
            encoder_blocks = _ENCODER_BLOCKS

        self.patch_size = patch_size
        self.latent_channels = latent_channels

        # After patchify: 3 * patch_size^2 = 48 channels
        in_channels = 3 * patch_size * patch_size
        feature_channels = base_channels

        self.conv_in = Conv3d(in_channels, feature_channels)

        self.down_blocks = []
        self.block_types = []

        for block_name, block_params in encoder_blocks:
            config = block_params if isinstance(block_params, dict) else {"num_layers": block_params}

            if block_name == "res_x":
                block = ResBlockGroup(feature_channels, config["num_layers"])
                self.block_types.append("res")
            elif block_name in _STRIDE_MAP:
                stride = _STRIDE_MAP[block_name]
                multiplier = config.get("multiplier", 2)
                out_channels = feature_channels * multiplier
                block = SpaceToDepthDownsample(feature_channels, out_channels, stride)
                feature_channels = out_channels
                self.block_types.append("downsample")
            else:
                raise ValueError(f"Unknown encoder block: {block_name}")

            self.down_blocks.append(block)

        # Output: pixel_norm + silu + conv_out
        # UNIFORM log_var mode: output channels = latent_channels + 1
        self.conv_out = Conv3d(feature_channels, latent_channels + 1)

        # Per-channel statistics for normalization
        self.mean_of_means = mx.zeros((latent_channels,))
        self.std_of_means = mx.ones((latent_channels,))

        # Scale/shift table for final norm (like decoder's last_scale_shift_table)
        self.last_scale_shift_table = mx.zeros((2, feature_channels))

    def __call__(self, x: mx.array) -> mx.array:
        """Encode video to normalized latent.

        Args:
            x: (B, 3, F, H, W) video in [-1, 1]. F must be 1+8k.

        Returns:
            (B, 128, F', H', W') normalized latent.
        """
        # Patchify: (B, 3, F, H, W) → (B, 48, F, H/4, W/4)
        x = _patchify(x, self.patch_size)

        # conv_in
        x = self.conv_in(x)
        _materialize(x)

        # Down blocks
        for i, (block, btype) in enumerate(zip(self.down_blocks, self.block_types)):
            x = block(x)
            _materialize(x)

        # Final norm + scale/shift + activation
        x = _pixel_norm(x)
        shift = self.last_scale_shift_table[0][None, :, None, None, None]
        scale = 1 + self.last_scale_shift_table[1][None, :, None, None, None]
        x = x * scale + shift
        x = nn.silu(x)

        # conv_out: (B, feature_ch, F', H', W') → (B, 129, F', H', W')
        x = self.conv_out(x)
        _materialize(x)

        # UNIFORM log_var: first 128 channels are means, last 1 is shared logvar
        means = x[:, :self.latent_channels]

        # Normalize using per-channel statistics
        mean_stat = self.mean_of_means.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std_stat = self.std_of_means.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        means = means.astype(mx.float32)
        normalized = (means - mean_stat) / std_stat

        return normalized


def load_vae_encoder(model_dir: Path) -> VideoEncoder:
    """Load VAE encoder from split safetensors file.

    Args:
        model_dir: Path to model directory containing vae_encoder.safetensors
                   and embedded_config.json.

    Returns:
        Loaded VideoEncoder ready for inference.
    """
    import json

    # Read encoder config
    config_path = model_dir / "embedded_config.json"
    encoder_blocks = _ENCODER_BLOCKS
    if config_path.exists():
        with open(config_path) as f:
            embedded = json.load(f)
        vae_config = embedded.get("vae", {})
        if "encoder_blocks" in vae_config:
            encoder_blocks = vae_config["encoder_blocks"]
            log.info("Using encoder config from embedded_config.json")

    encoder = VideoEncoder(encoder_blocks=encoder_blocks)

    # Load weights
    weights_path = model_dir / "vae_encoder.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"vae_encoder.safetensors not found in {model_dir}")

    raw = mx.load(str(weights_path))
    prefix = "vae_encoder."

    # Load per-channel statistics
    for suffix, attr in [
        ("per_channel_statistics._mean_of_means", "mean_of_means"),
        ("per_channel_statistics._std_of_means", "std_of_means"),
    ]:
        key = f"{prefix}{suffix}"
        if key in raw:
            setattr(encoder, attr, raw[key])

    # Load conv_in
    for suffix in ("weight", "bias"):
        key = f"{prefix}conv_in.conv.{suffix}"
        if key in raw:
            setattr(encoder.conv_in, suffix, raw[key])

    # Load conv_out
    for suffix in ("weight", "bias"):
        key = f"{prefix}conv_out.conv.{suffix}"
        if key in raw:
            setattr(encoder.conv_out, suffix, raw[key])

    # Load conv_norm_out / last_scale_shift_table
    key = f"{prefix}conv_norm_out.weight"
    if key in raw:
        # pixel_norm has no learnable params — this shouldn't exist for pixel_norm
        pass

    # Load down_blocks
    for block_idx, (block, btype) in enumerate(zip(encoder.down_blocks, encoder.block_types)):
        if btype == "res":
            for res_idx, res_block in enumerate(block.res_blocks):
                for conv_name in ("conv1", "conv2"):
                    conv = getattr(res_block, conv_name)
                    for suffix in ("weight", "bias"):
                        key = f"{prefix}down_blocks.{block_idx}.res_blocks.{res_idx}.{conv_name}.conv.{suffix}"
                        if key in raw:
                            setattr(conv, suffix, raw[key])
                        else:
                            # Try norm1/norm2 naming (some models use this)
                            alt = conv_name.replace("conv", "norm")
                            alt_key = f"{prefix}down_blocks.{block_idx}.res_blocks.{res_idx}.{alt}.conv.{suffix}"
                            if alt_key in raw:
                                setattr(conv, suffix, raw[alt_key])
        elif btype == "downsample":
            for suffix in ("weight", "bias"):
                key = f"{prefix}down_blocks.{block_idx}.conv.conv.{suffix}"
                if key in raw:
                    setattr(block.conv, suffix, raw[key])

    # Count loaded weights
    loaded = sum(1 for k in raw if k.startswith(prefix))
    log.info(
        "VAE encoder loaded: %d keys, %.2f GB active",
        loaded, mx.get_active_memory() / (1024**3),
    )

    return encoder


def encode_image(
    image: mx.array,
    encoder: VideoEncoder,
) -> mx.array:
    """Encode a single image to latent space.

    Args:
        image: (B, 3, 1, H, W) image normalized to [-1, 1].
               H and W must be divisible by 32.

    Returns:
        (B, 128, 1, H/32, W/32) normalized latent.
    """
    return encoder(image)
