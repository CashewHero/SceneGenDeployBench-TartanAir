from __future__ import annotations

import numpy as np


def decode_float32_le_bgra(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("float32_le_bgra depth requires an 8-bit four-channel image")
    return np.squeeze(np.ascontiguousarray(image).view("<f4"), axis=-1)


def encode_float32_le_bgra(depth: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    packed = np.ascontiguousarray(depth.astype("<f4", copy=False))
    return packed.view(np.uint8).reshape(depth.shape + (4,))


def depth_to_ray_distance(depth: np.ndarray) -> np.ndarray:
    """Convert TartanAir 90-degree pinhole Z-depth to camera-ray distance."""
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    height, width = depth.shape
    focal = width / 2.0
    x = np.arange(width, dtype=np.float32) + 0.5 - width / 2.0
    y = np.arange(height, dtype=np.float32) + 0.5 - height / 2.0
    x_grid, y_grid = np.meshgrid(x, y)
    factor = np.sqrt(x_grid * x_grid + y_grid * y_grid + focal * focal) / focal
    return np.asarray(depth, dtype=np.float32) * factor
