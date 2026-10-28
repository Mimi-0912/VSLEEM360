import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from tiled_infer import (feather_window, tile_starts_circular,
                         tile_starts_linear, tiled_sr)


def _bilinear_local_sr(scale):
    """A spatially-local reference SR: per-tile bicubic-free bilinear upscale.

    Uses only within-tile pixels, so a correct tiler must reproduce the
    full-frame result of the same operation up to float rounding.
    """
    import cv2

    def fn(tile_lr):
        center = tile_lr[tile_lr.shape[0] // 2]
        h, w = center.shape[:2]
        up = cv2.resize(center, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        return up.astype(np.float32) / 255.0

    return fn


def test_linear_starts_cover_end():
    starts = tile_starts_linear(100, 32, 8)
    assert starts[0] == 0 and starts[-1] == 100 - 32


def test_circular_starts_wrap_uniform():
    starts = tile_starts_circular(120, 40, 8)
    assert len(starts) >= 120 // (40 - 8)
    assert all(0 <= s < 120 for s in starts)


def test_window_strictly_positive():
    w = feather_window(20, 5, 5)
    assert w.min() > 0 and abs(w.max() - 1.0) < 1e-12


def test_tiled_equals_full_nearest():
    # NEAREST upscaling is exactly block-local, so tiled == full bit-for-bit
    # after weight normalisation, on both wrapping and non-wrapping axes.
    rng = np.random.default_rng(0)
    lr = rng.integers(0, 256, (7, 48, 96, 3), dtype=np.uint8)
    scale = 3
    fn = _bilinear_local_sr(scale)
    full = fn(lr)  # single-tile reference (tile covers whole frame)
    tiled = tiled_sr(lr, fn, scale=scale, tile_h=24, tile_w=32, overlap=8)
    assert tiled.shape == full.shape
    np.testing.assert_allclose(tiled, full, atol=1e-5)


def test_tiled_covers_seam_columns():
    # A tile straddling the longitude seam must keep column continuity: a
    # single bright column at x=0 appears exactly once in the output.
    lr = np.zeros((7, 12, 24, 3), dtype=np.uint8)
    lr[:, :, 0] = 255
    fn = _bilinear_local_sr(3)
    tiled = tiled_sr(lr, fn, scale=3, tile_h=12, tile_w=8, overlap=2)
    col_energy = tiled.sum(axis=(0, 2))
    bright = np.where(col_energy > col_energy.max() * 0.5)[0]
    assert set(bright.tolist()) == {0, 1, 2}  # x=0 upscaled x3, appears once
