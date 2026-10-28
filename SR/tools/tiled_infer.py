"""ERP-aware tiled super-resolution for full-frame panoramic inference.

The full LR sequence (960x1920) cannot be super-resolved in one pass on a 12 GiB
GPU (measured: BasicVSR++ OOMs above 320x640 LR). This module tiles the spatial
plane, runs a caller-supplied ``sr_fn`` on each tile, and stitches the central
HR frame with overlap-add feather blending.

ERP geometry:
- Width (longitude) is circular: horizontal tiles wrap modulo W, so a tile may
  straddle the seam and adjacent columns stay continuous across it.
- Height (latitude) is NOT circular: the poles are hard image boundaries and
  vertical tiles never wrap past row 0 or row H.

Correctness contract: for any spatially-local ``sr_fn`` and sufficient overlap,
the tiled result equals the full-frame result up to float rounding, because the
overlap-add weights are normalised by their accumulated sum (see the unit test
``tests/test_tiled_infer.py`` which checks this against bilinear upsampling).
"""

from __future__ import annotations

import numpy as np
import torch


def tile_starts_linear(length: int, tile: int, overlap: int) -> list[int]:
    """Start indices covering ``[0, length)`` on a NON-wrapping axis.

    The final tile is clamped to end exactly at ``length`` so the last rows are
    always covered even when ``length`` is not a multiple of the stride.
    """
    if tile >= length:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_starts_circular(length: int, tile: int, overlap: int) -> list[int]:
    """Start indices covering a WRAPPING axis with (near-)uniform stride.

    Tiles wrap modulo ``length``; the count is chosen so the stride is as even
    as possible and every column is covered by at least one tile interior.
    """
    if tile >= length:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile size")
    count = int(np.ceil(length / stride))
    return [int(round(i * length / count)) % length for i in range(count)]


def feather_window(size: int, ramp_left: int, ramp_right: int) -> np.ndarray:
    """Trapezoidal window, strictly positive, ramps only where a neighbour abuts.

    Ramps rise linearly from ``1/(ramp+1)`` to 1 so the window never reaches 0;
    combined with accumulated-weight normalisation this blends smoothly and
    stays exact where only one tile covers a pixel.
    """
    window = np.ones(size, dtype=np.float64)
    if ramp_left > 0:
        window[:ramp_left] = (np.arange(1, ramp_left + 1)) / (ramp_left + 1)
    if ramp_right > 0:
        window[size - ramp_right:] = (np.arange(ramp_right, 0, -1)) / (ramp_right + 1)
    return window


def tiled_sr(lr_seq: np.ndarray, sr_fn, scale: int, tile_h: int, tile_w: int,
             overlap: int) -> np.ndarray:
    """Super-resolve a full ERP LR sequence by ERP-aware tiling.

    Args:
        lr_seq: uint8 array ``(T, H, W, 3)`` in RGB, the full LR frame window.
        sr_fn: callable mapping a uint8 tile ``(T, th, tw, 3)`` to a float32 HR
            centre frame ``(scale*th, scale*tw, 3)`` in [0, 1].
        scale: integer upscale factor (3 here).
        tile_h, tile_w: LR tile size in pixels.
        overlap: LR overlap in pixels shared between neighbouring tiles.

    Returns:
        float32 HR centre frame ``(scale*H, scale*W, 3)`` in [0, 1].
    """
    t, height, width, _ = lr_seq.shape
    out_h, out_w = height * scale, width * scale
    accum = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float64)

    row_starts = tile_starts_linear(height, tile_h, overlap)
    col_starts = tile_starts_circular(width, tile_w, overlap)
    eff_h = min(tile_h, height)
    eff_w = min(tile_w, width)

    for ry in row_starts:
        # Vertical: clamp, no wrap. Ramp only on the side that has a neighbour.
        ramp_top = overlap if ry > 0 else 0
        ramp_bot = overlap if ry + eff_h < height else 0
        win_y = feather_window(eff_h * scale, ramp_top * scale, ramp_bot * scale)
        for cx in col_starts:
            cols = (cx + np.arange(eff_w)) % width  # circular longitude
            tile_lr = lr_seq[:, ry:ry + eff_h][:, :, cols, :]
            hr_tile = sr_fn(np.ascontiguousarray(tile_lr))  # (scale*eff_h, scale*eff_w, 3)
            win_x = feather_window(eff_w * scale, overlap * scale, overlap * scale)
            win = (win_y[:, None] * win_x[None, :])[:, :, None]
            hr_cols = (cx * scale + np.arange(eff_w * scale)) % out_w
            block = hr_tile.astype(np.float64) * win
            # Scatter-add into the wrapped HR column range, row range is direct.
            np.add.at(accum, (slice(ry * scale, ry * scale + eff_h * scale), hr_cols), block)
            np.add.at(weight, (slice(ry * scale, ry * scale + eff_h * scale), hr_cols),
                      win)
    if float(weight.min()) <= 0:
        raise RuntimeError("tiling left an uncovered pixel; increase tile count")
    return (accum / weight).astype(np.float32)
