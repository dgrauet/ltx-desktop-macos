"""VAE video decoder for LTX-2.3.

Decodes latent [B, 128, F', H', W'] → pixel video [B, 3, F, H, W].
Architecture from embedded_config.json: 9 blocks, no timestep conditioning,
non-causal, pixel_norm, patch_size=4, zero spatial padding.

Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger(__name__)

# mx.eval is MLX tensor materialization (NOT Python eval)
_materialize = mx.eval

# LTX-2.3 decoder block config (from embedded_config.json)
_DECODER_BLOCKS = [
    ["res_x", {"num_layers": 4}],
    ["compress_space", {"multiplier": 2}],
    ["res_x", {"num_layers": 6}],
    ["compress_time", {"multiplier": 2}],
    ["res_x", {"num_layers": 4}],
    ["compress_all", {"multiplier": 1}],
    ["res_x", {"num_layers": 2}],
    ["compress_all", {"multiplier": 2}],
    ["res_x", {"num_layers": 2}],
]

# Stride mapping for upsample block types
_STRIDE_MAP = {
    "compress_all": (2, 2, 2),
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
}


def _pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Pixel normalization across channels (dim=1)."""
    variance = mx.mean(x * x, axis=1, keepdims=True)
    return x * mx.rsqrt(variance + eps)


class Conv3d(nn.Module):
    """3D convolution via temporal iteration over 2D spatial convolutions.

    Weight format: (out_C, kT, kH, kW, in_C) — MLX-converted format.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        k = kernel_size
        self.kernel_size = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Weights stored in MLX format: (out_C, kT, kH, kW, in_C)
        self.weight = mx.zeros((out_channels, k, k, k, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 3D conv with symmetric temporal padding (non-causal).

        Args:
            x: Input (B, C, T, H, W).

        Returns:
            Output (B, C_out, T, H, W) — same T, H, W with padding.
        """
        b, c, t, h, w = x.shape
        k = self.kernel_size
        p = k // 2  # spatial padding

        # Spatial zero-padding
        if p > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (p, p), (p, p)])

        # Non-causal temporal padding: replicate edge frames
        if k > 1:
            first = mx.repeat(x[:, :, :1], p, axis=2)
            last = mx.repeat(x[:, :, -1:], p, axis=2)
            x = mx.concatenate([first, x, last], axis=2)

        _, _, t_pad, h_pad, w_pad = x.shape
        h_out = h_pad - k + 1
        w_out = w_pad - k + 1
        t_out = t_pad - k + 1

        # Iterate over temporal kernel positions, accumulating 2D conv results
        output = None
        for kt in range(k):
            # Weight slice: (out_C, kH, kW, in_C) from (out_C, kT, kH, kW, in_C)
            w_slice = self.weight[:, kt, :, :, :]  # (out_C, kH, kW, in_C)

            # Temporal slice of input
            x_slice = x[:, :, kt:kt + t_out, :, :]  # (B, C, T_out, H_pad, W_pad)

            # Reshape for 2D conv: (B*T_out, H_pad, W_pad, C)
            x_2d = x_slice.transpose(0, 2, 1, 3, 4)  # (B, T_out, C, H, W)
            x_2d = x_2d.reshape(b * t_out, c, h_pad, w_pad)
            x_2d = x_2d.transpose(0, 2, 3, 1)  # (B*T_out, H, W, C)

            conv_out = mx.conv2d(x_2d, w_slice, padding=0)  # (B*T_out, H_out, W_out, C_out)

            # Reshape back to 5D
            _, _, _, c_out = conv_out.shape
            conv_out = conv_out.reshape(b, t_out, h_out, w_out, c_out)
            conv_out = conv_out.transpose(0, 4, 1, 2, 3)  # (B, C_out, T_out, H_out, W_out)

            if output is None:
                output = conv_out
            else:
                output = output + conv_out

        return output + self.bias[None, :, None, None, None]


class ResBlock(nn.Module):
    """Residual block with pixel norm (no timestep conditioning for V2.3)."""

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


class DepthToSpaceUpsample(nn.Module):
    """Upsample via conv then depth-to-space rearrangement."""

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels_reduction_factor = out_channels_reduction_factor
        stride_prod = math.prod(stride)
        conv_out = stride_prod * in_channels // out_channels_reduction_factor
        self.final_out_channels = in_channels // out_channels_reduction_factor
        self.conv = Conv3d(in_channels, conv_out)

    def _depth_to_space(self, x: mx.array, c_out: int) -> mx.array:
        """Rearrange channels into spatial/temporal dims."""
        b, c, t, h, w = x.shape
        ft, fh, fw = self.stride
        x = x.reshape(b, c_out, ft, fh, fw, t, h, w)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, c_out, t * ft, h * fh, w * fw)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        ft = self.stride[0]
        x = self.conv(x)
        x = self._depth_to_space(x, self.final_out_channels)
        # Remove first temporal frame when temporal stride > 1
        if ft > 1:
            x = x[:, :, 1:]
        return x


@mx.compile
def unpatchify(x: mx.array, patch_size: int) -> mx.array:
    """Depth-to-space: (B, C*p*p, F, H, W) -> (B, C, F, H*p, W*p)."""
    if patch_size == 1:
        return x
    b, c_packed, f, h, w = x.shape
    p = patch_size
    c = c_packed // (p * p)
    x = x.reshape(b, c, p, p, f, h, w)
    x = x.transpose(0, 1, 4, 5, 2, 6, 3)
    x = x.reshape(b, c, f, h * p, w * p)
    return x


class VideoDecoder(nn.Module):
    """LTX-2.3 VAE video decoder.

    Non-causal, no timestep conditioning, pixel_norm, patch_size=4.
    """

    def __init__(self, decoder_blocks: list | None = None, base_channels: int = 128):
        super().__init__()

        if decoder_blocks is None:
            decoder_blocks = _DECODER_BLOCKS

        self.patch_size = 4

        # Per-channel statistics for latent denormalization
        self.mean = mx.zeros((128,))
        self.std = mx.zeros((128,))

        # Starting feature channels = base_channels * 8 (matches PyTorch/reference)
        feature_channels = base_channels * 8
        reversed_blocks = list(reversed(decoder_blocks))

        # conv_in: latent_channels -> feature_channels
        self.conv_in = Conv3d(128, feature_channels)

        # Build up_blocks (reversed order)
        self.up_blocks: list[ResBlockGroup | DepthToSpaceUpsample] = []
        self.block_types: list[str] = []

        for block_name, block_params in reversed_blocks:
            if block_name == "res_x":
                block = ResBlockGroup(feature_channels, block_params["num_layers"])
                self.up_blocks.append(block)
                self.block_types.append("res")
            elif block_name in _STRIDE_MAP:
                stride = _STRIDE_MAP[block_name]
                multiplier = block_params.get("multiplier", 1)
                block = DepthToSpaceUpsample(
                    in_channels=feature_channels,
                    stride=stride,
                    out_channels_reduction_factor=multiplier,
                )
                feature_channels = feature_channels // multiplier
                self.up_blocks.append(block)
                self.block_types.append("upsample")

        self.final_channels = feature_channels

        # conv_out: final_channels -> 48 (3 * patch_size**2)
        self.conv_out = Conv3d(feature_channels, 3 * self.patch_size ** 2)

        # Final scale/shift table (no timestep conditioning)
        self.last_scale_shift_table = mx.zeros((2, feature_channels))

    def __call__(self, latent: mx.array) -> mx.array:
        """Decode latent to video.

        Args:
            latent: (B, 128, F', H', W') normalized latent.

        Returns:
            (B, 3, F, H, W) video in [-1, 1].
        """
        # Denormalize
        x = latent * self.std[None, :, None, None, None]
        x = x + self.mean[None, :, None, None, None]

        # conv_in
        x = self.conv_in(x)
        _materialize(x)

        # Up blocks
        for i, (block, btype) in enumerate(zip(self.up_blocks, self.block_types)):
            x = block(x)
            _materialize(x)
            log.debug("Block %d/%d (%s) done, shape=%s", i + 1, len(self.up_blocks), btype, x.shape)

        # Final norm + scale/shift + activation
        x = _pixel_norm(x)
        shift = self.last_scale_shift_table[0][None, :, None, None, None]
        scale = 1 + self.last_scale_shift_table[1][None, :, None, None, None]
        x = x * scale + shift
        x = nn.silu(x)

        # conv_out
        x = self.conv_out(x)
        _materialize(x)

        # Unpatchify: (B, 48, F, H, W) -> (B, 3, F, H*4, W*4)
        x = unpatchify(x, self.patch_size)
        _materialize(x)

        return x


def load_vae_decoder(model_dir: Path) -> VideoDecoder:
    """Load VAE decoder from split safetensors file.

    Args:
        model_dir: Path to model directory containing vae_decoder.safetensors
                   and embedded_config.json.

    Returns:
        Loaded VideoDecoder ready for inference.
    """
    import json

    # Read decoder config
    config_path = model_dir / "embedded_config.json"
    decoder_blocks = _DECODER_BLOCKS
    if config_path.exists():
        with open(config_path) as f:
            embedded = json.load(f)
        vae_config = embedded.get("vae", {})
        if "decoder_blocks" in vae_config:
            decoder_blocks = vae_config["decoder_blocks"]
            log.info("Using decoder config from embedded_config.json")

    decoder = VideoDecoder(decoder_blocks=decoder_blocks)

    # Load weights
    weights_path = model_dir / "vae_decoder.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"vae_decoder.safetensors not found in {model_dir}")

    raw = mx.load(str(weights_path))
    prefix = "vae_decoder."

    # Load per-channel statistics
    mean_key = f"{prefix}per_channel_statistics.mean"
    std_key = f"{prefix}per_channel_statistics.std"
    if mean_key in raw:
        decoder.mean = raw[mean_key]
    if std_key in raw:
        decoder.std = raw[std_key]

    # Load conv_in
    for suffix in ("weight", "bias"):
        key = f"{prefix}conv_in.conv.{suffix}"
        if key in raw:
            setattr(decoder.conv_in, suffix, raw[key])

    # Load conv_out
    for suffix in ("weight", "bias"):
        key = f"{prefix}conv_out.conv.{suffix}"
        if key in raw:
            setattr(decoder.conv_out, suffix, raw[key])

    # Load up_blocks
    for block_idx, (block, btype) in enumerate(zip(decoder.up_blocks, decoder.block_types)):
        if btype == "res":
            for res_idx, res_block in enumerate(block.res_blocks):
                for conv_name in ("conv1", "conv2"):
                    conv = getattr(res_block, conv_name)
                    for suffix in ("weight", "bias"):
                        key = f"{prefix}up_blocks.{block_idx}.res_blocks.{res_idx}.{conv_name}.conv.{suffix}"
                        if key in raw:
                            setattr(conv, suffix, raw[key])
        else:  # upsample
            for suffix in ("weight", "bias"):
                key = f"{prefix}up_blocks.{block_idx}.conv.conv.{suffix}"
                if key in raw:
                    setattr(block.conv, suffix, raw[key])

    # Load last_scale_shift_table
    lst_key = f"{prefix}last_scale_shift_table"
    if lst_key in raw:
        decoder.last_scale_shift_table = raw[lst_key]

    del raw
    _materialize(decoder.parameters())

    loaded_mem = mx.get_active_memory() / (1024**3)
    log.info("VAE decoder loaded: %.2f GB active", loaded_mem)

    return decoder


def decode_video(
    latent: mx.array,
    decoder: VideoDecoder,
) -> mx.array:
    """Decode latent to uint8 video frames.

    Args:
        latent: (B, 128, F', H', W') or (128, F', H', W').
        decoder: Loaded VideoDecoder.

    Returns:
        (F, H, W, 3) uint8 frames in [0, 255].
    """
    if latent.ndim == 4:
        latent = latent[None]

    video = decoder(latent)

    # [-1, 1] -> [0, 255] uint8
    video = mx.clip((video + 1) / 2, 0, 1) * 255
    video = video.astype(mx.uint8)

    # (B, C, F, H, W) -> (F, H, W, C)
    video = video[0].transpose(1, 2, 3, 0)

    return video


def _find_ffmpeg() -> str:
    """Find ffmpeg binary, checking common Homebrew paths."""
    import shutil

    path = shutil.which("ffmpeg")
    if path:
        return path
    # Homebrew on Apple Silicon / Intel
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found. Install with: brew install ffmpeg"
    )


def streaming_decode_to_ffmpeg(
    latent: mx.array,
    decoder: VideoDecoder,
    output_path: str,
    fps: int = 24,
    progress_fn: callable | None = None,
) -> None:
    """Decode latent and stream frames directly to ffmpeg.

    Avoids holding all decoded frames in memory by piping raw RGB
    frame-by-frame to ffmpeg's stdin.

    Args:
        latent: (B, 128, F', H', W') latent tensor.
        decoder: Loaded VideoDecoder.
        output_path: Output MP4 file path.
        fps: Frames per second.
        progress_fn: Optional callback(frame_idx, total_frames).
    """
    import subprocess

    ffmpeg_bin = _find_ffmpeg()

    if latent.ndim == 4:
        latent = latent[None]

    # Decode full video (the decoder already materializes per-block)
    video = decoder(latent)

    # Get dimensions
    b, c, num_frames, height, width = video.shape

    log.info("Streaming %d frames (%dx%d) to ffmpeg (%s)", num_frames, width, height, ffmpeg_bin)

    ffmpeg_cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for i in range(num_frames):
            # Extract single frame: (C, H, W) -> uint8 (H, W, 3)
            frame = video[0, :, i, :, :]  # (C, H, W)
            frame = mx.clip((frame + 1) / 2, 0, 1) * 255
            frame = frame.astype(mx.uint8)
            frame = frame.transpose(1, 2, 0)  # (H, W, C)

            # Materialize before extracting bytes
            _materialize(frame)
            frame_bytes = bytes(memoryview(frame))
            proc.stdin.write(frame_bytes)

            # Free frame memory periodically
            del frame
            if i % 8 == 0:
                gc.collect()
                mx.clear_cache()

            if progress_fn is not None:
                progress_fn(i + 1, num_frames)

    finally:
        proc.stdin.close()
        proc.wait()

    # Free decoded video
    del video
    gc.collect()
    mx.clear_cache()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr[-500:]}")

    log.info("Video saved to %s", output_path)
