"""Video latent patchification for LTX-2.3.

Converts 5D video latents [B, C, F, H, W] to 3D patch sequences [B, T, D]
for transformer processing, and back.

Adapted from Acelogic/LTX-2-MLX (Apache-2.0).
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import mlx.core as mx
import numpy as np


class SpatioTemporalScaleFactors(NamedTuple):
    """VAE compression factors."""
    time: int = 8
    height: int = 32
    width: int = 32


class VideoLatentShape(NamedTuple):
    """Shape of video latent tensor [B, C, F, H, W]."""
    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.height, self.width)

    def mask_shape(self) -> "VideoLatentShape":
        return VideoLatentShape(self.batch, 1, self.frames, self.height, self.width)

    @staticmethod
    def from_pixel_shape(
        batch: int,
        channels: int,
        num_frames: int,
        height: int,
        width: int,
        scale_factors: SpatioTemporalScaleFactors = SpatioTemporalScaleFactors(),
    ) -> "VideoLatentShape":
        """Convert pixel dimensions to latent dimensions."""
        return VideoLatentShape(
            batch=batch,
            channels=channels,
            frames=(num_frames - 1) // scale_factors.time + 1,
            height=height // scale_factors.height,
            width=width // scale_factors.width,
        )

    def upscale(
        self,
        scale_factors: SpatioTemporalScaleFactors = SpatioTemporalScaleFactors(),
    ) -> Tuple[int, int, int, int, int]:
        """Convert latent dimensions back to pixel dimensions."""
        return (
            self.batch,
            self.channels,
            (self.frames - 1) * scale_factors.time + 1,
            self.height * scale_factors.height,
            self.width * scale_factors.width,
        )


class VideoLatentPatchifier:
    """Patchifies 5D video latents to 3D patch sequences for transformer input.

    With patch_size=1 (default for LTX-2.3), this is effectively a reshape:
    [B, C, F, H, W] → [B, F*H*W, C]
    """

    def __init__(self, patch_size: int = 1):
        self.patch_size = patch_size

    @property
    def num_patches(self) -> int:
        return self.patch_size ** 3

    def patchify(self, latent: mx.array) -> mx.array:
        """Convert [B, C, F, H, W] → [B, T, D].

        T = (F/p)*(H/p)*(W/p), D = C*p*p*p where p = patch_size.
        """
        b, c, f, h, w = latent.shape
        p = self.patch_size

        if p == 1:
            # Fast path: simple reshape
            return latent.transpose(0, 2, 3, 4, 1).reshape(b, f * h * w, c)

        # General case
        latent = latent.reshape(b, c, f // p, p, h // p, p, w // p, p)
        latent = latent.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        latent = latent.reshape(b, (f // p) * (h // p) * (w // p), c * p * p * p)
        return latent

    def unpatchify(self, latent: mx.array, shape: VideoLatentShape) -> mx.array:
        """Convert [B, T, D] → [B, C, F, H, W]."""
        b = shape.batch
        c = shape.channels
        f = shape.frames
        h = shape.height
        w = shape.width
        p = self.patch_size

        if p == 1:
            return latent.reshape(b, f, h, w, c).transpose(0, 4, 1, 2, 3)

        latent = latent.reshape(b, f // p, h // p, w // p, c, p, p, p)
        latent = latent.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        latent = latent.reshape(b, c, f, h, w)
        return latent

    def get_patch_grid_bounds(
        self, shape: VideoLatentShape
    ) -> mx.array:
        """Get position bounds for each patch.

        Returns shape (batch, 3, num_patches, 2) with [start, end) for each
        (frame, height, width) axis.
        """
        f, h, w = shape.frames, shape.height, shape.width
        p = self.patch_size

        # Create coordinate grids
        f_coords = np.arange(0, f, p)
        h_coords = np.arange(0, h, p)
        w_coords = np.arange(0, w, p)

        # Meshgrid: (f_grid, h_grid, w_grid) each of shape (F/p, H/p, W/p)
        f_grid, h_grid, w_grid = np.meshgrid(f_coords, h_coords, w_coords, indexing="ij")

        # Flatten to (num_patches,)
        f_flat = f_grid.reshape(-1)
        h_flat = h_grid.reshape(-1)
        w_flat = w_grid.reshape(-1)

        num_patches = len(f_flat)

        # Create bounds: [start, start+patch_size) for each axis
        # Shape: (3, num_patches, 2)
        bounds = np.zeros((3, num_patches, 2), dtype=np.float32)
        bounds[0, :, 0] = f_flat
        bounds[0, :, 1] = f_flat + p
        bounds[1, :, 0] = h_flat
        bounds[1, :, 1] = h_flat + p
        bounds[2, :, 0] = w_flat
        bounds[2, :, 1] = w_flat + p

        # Broadcast to batch: (batch, 3, num_patches, 2)
        bounds = np.broadcast_to(bounds[None], (shape.batch, 3, num_patches, 2))
        return mx.array(bounds)


def get_pixel_coords(
    grid_bounds: mx.array,
    scale_factors: SpatioTemporalScaleFactors = SpatioTemporalScaleFactors(),
    causal_fix: bool = True,
) -> mx.array:
    """Convert latent-space patch bounds to pixel-space coordinates.

    Args:
        grid_bounds: Shape (batch, 3, num_patches, 2) in latent space.
        scale_factors: VAE compression factors.
        causal_fix: If True, shift first frame handling for causal temporal VAE.

    Returns:
        Shape (batch, 3, num_patches, 2) in pixel space.
    """
    scales = mx.array([
        [scale_factors.time],
        [scale_factors.height],
        [scale_factors.width],
    ], dtype=mx.float32)  # Shape: (3, 1)

    # Scale all axes
    pixel_coords = grid_bounds * scales[None, :, None, :]

    if causal_fix:
        # Temporal axis: shift by -(scale-1) to handle first-frame offset
        temporal = pixel_coords[:, 0:1, :, :]
        temporal = temporal - (scale_factors.time - 1)
        # Clip to min=0 to avoid negative temporal positions
        temporal = mx.maximum(temporal, 0)
        pixel_coords = mx.concatenate([temporal, pixel_coords[:, 1:, :, :]], axis=1)

    return pixel_coords


def positions_from_shape(
    shape: VideoLatentShape,
    fps: float = 24.0,
    scale_factors: SpatioTemporalScaleFactors = SpatioTemporalScaleFactors(),
    causal_fix: bool = True,
) -> mx.array:
    """Compute position embeddings for a given latent shape.

    Returns shape (batch, 3, num_patches, 2) where temporal axis is in seconds.
    """
    patchifier = VideoLatentPatchifier(patch_size=1)
    grid_bounds = patchifier.get_patch_grid_bounds(shape)
    pixel_coords = get_pixel_coords(grid_bounds, scale_factors, causal_fix)

    # Convert temporal axis from frames to seconds
    pixel_coords_np = np.array(pixel_coords)
    pixel_coords_np[:, 0, :, :] = pixel_coords_np[:, 0, :, :] / fps

    return mx.array(pixel_coords_np)
