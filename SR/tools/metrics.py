"""Reference RGB and spherical ERP metrics for VSLEEM360."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _validate(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"expected equal HxWx3 arrays, got {pred.shape} and {gt.shape}")
    pred = pred.astype(np.float32) / 255.0 if pred.dtype == np.uint8 else pred.astype(np.float32)
    gt = gt.astype(np.float32) / 255.0 if gt.dtype == np.uint8 else gt.astype(np.float32)
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError("images contain NaN or infinity")
    if pred.min() < 0 or pred.max() > 1 or gt.min() < 0 or gt.max() > 1:
        raise ValueError("floating point inputs must be in [0, 1]")
    return pred, gt


def latitude_weights(height: int) -> np.ndarray:
    y = np.arange(height, dtype=np.float64)
    weights = np.cos(np.pi * (y + 0.5 - height / 2.0) / height)
    return weights / weights.sum()


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _validate(pred, gt)
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def ws_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _validate(pred, gt)
    row_mse = np.mean((pred - gt) ** 2, axis=(1, 2))
    mse = float(np.sum(row_mse * latitude_weights(pred.shape[0])))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def _filter_erp(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    # Horizontal is circular because ERP columns meet at the longitude seam;
    # vertical reflect avoids inventing a second pole beyond the image boundary.
    padded = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    padded[:, :radius] = padded[:, -2 * radius:-radius]
    padded[:, -radius:] = padded[:, radius:2 * radius]
    filtered = cv2.filter2D(padded, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return filtered[radius:-radius, radius:-radius]


def ssim_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred, gt = _validate(pred, gt)
    one_d = cv2.getGaussianKernel(11, 1.5, cv2.CV_64F)
    kernel = one_d @ one_d.T
    mu_x, mu_y = _filter_erp(pred, kernel), _filter_erp(gt, kernel)
    sigma_x = _filter_erp(pred * pred, kernel) - mu_x * mu_x
    sigma_y = _filter_erp(gt * gt, kernel) - mu_y * mu_y
    sigma_xy = _filter_erp(pred * gt, kernel) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    values = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return np.mean(values, axis=2)


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(ssim_map(pred, gt)))


def ws_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    rows = np.mean(ssim_map(pred, gt), axis=1)
    return float(np.sum(rows * latitude_weights(len(rows))))


# --- Y-channel (luma) variants: keep plain and WS metrics on the same channel.
# Y matches mmagic/mmcv exactly (BT.601, y_only). WS metrics operate on the FULL
# ERP frame (no crop) because latitude weighting already handles the poles and
# longitude is circular; plain PSNR/SSIM follow the SR-literature crop convention
# separately via the official mmagic functions.

def to_y01(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> luma Y float32 HxW in [0,1] (BT.601, mmcv-compatible)."""
    import mmcv
    if rgb_u8.dtype != np.uint8:
        raise ValueError("expected uint8 RGB")
    return mmcv.rgb2ycbcr(rgb_u8.astype(np.float32) / 255.0, y_only=True)


def ws_psnr_y(pred_y: np.ndarray, gt_y: np.ndarray) -> float:
    """WS-PSNR on a single Y channel (HxW float in [0,1]), peak = 1.0."""
    if pred_y.shape != gt_y.shape or pred_y.ndim != 2:
        raise ValueError(f"expected equal HxW arrays, got {pred_y.shape}, {gt_y.shape}")
    row_mse = np.mean((pred_y - gt_y) ** 2, axis=1)
    mse = float(np.sum(row_mse * latitude_weights(pred_y.shape[0])))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def _ssim_map_y(pred_y: np.ndarray, gt_y: np.ndarray) -> np.ndarray:
    """ERP-aware SSIM map on a single Y channel (HxW float in [0,1])."""
    one_d = cv2.getGaussianKernel(11, 1.5, cv2.CV_64F)
    kernel = one_d @ one_d.T
    mu_x, mu_y = _filter_erp(pred_y, kernel), _filter_erp(gt_y, kernel)
    sigma_x = _filter_erp(pred_y * pred_y, kernel) - mu_x * mu_x
    sigma_y = _filter_erp(gt_y * gt_y, kernel) - mu_y * mu_y
    sigma_xy = _filter_erp(pred_y * gt_y, kernel) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))


def ws_ssim_y(pred_y: np.ndarray, gt_y: np.ndarray) -> float:
    """WS-SSIM on a single Y channel (HxW float in [0,1])."""
    rows = np.mean(_ssim_map_y(pred_y, gt_y), axis=1)
    return float(np.sum(rows * latitude_weights(len(rows))))


def _reflect101(indices: np.ndarray, length: int) -> np.ndarray:
    if length < 2:
        return np.zeros_like(indices)
    period = 2 * length - 2
    values = np.mod(indices, period)
    return np.where(values < length, values, period - values)


def _ssim_core(pred: np.ndarray, gt: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    mu_x = cv2.filter2D(pred, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    mu_y = cv2.filter2D(gt, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    sigma_x = cv2.filter2D(pred * pred, -1, kernel, borderType=cv2.BORDER_CONSTANT) - mu_x * mu_x
    sigma_y = cv2.filter2D(gt * gt, -1, kernel, borderType=cv2.BORDER_CONSTANT) - mu_y * mu_y
    sigma_xy = cv2.filter2D(pred * gt, -1, kernel, borderType=cv2.BORDER_CONSTANT) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))


def distortion_metrics_striped(pred: np.ndarray, gt: np.ndarray, stripe_rows: int = 256) -> dict[str, float]:
    """Compute four distortion metrics exactly with bounded working memory."""
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"expected equal HxWx3 arrays, got {pred.shape} and {gt.shape}")
    height, width, _ = pred.shape
    weights = latitude_weights(height)
    one_d = cv2.getGaussianKernel(11, 1.5, cv2.CV_32F)
    kernel = one_d @ one_d.T
    radius = 5
    squared_sum = weighted_squared_sum = ssim_sum = weighted_ssim_sum = 0.0
    for y0 in range(0, height, stripe_rows):
        y1 = min(height, y0 + stripe_rows)
        core_pred = pred[y0:y1].astype(np.float32) / 255.0
        core_gt = gt[y0:y1].astype(np.float32) / 255.0
        row_mse = np.mean((core_pred - core_gt) ** 2, axis=(1, 2), dtype=np.float64)
        squared_sum += float(row_mse.sum())
        weighted_squared_sum += float(np.sum(row_mse * weights[y0:y1]))

        indices = _reflect101(np.arange(y0 - radius, y1 + radius), height)
        p = pred[indices].astype(np.float32) / 255.0
        g = gt[indices].astype(np.float32) / 255.0
        p = np.pad(p, ((0, 0), (radius, radius), (0, 0)), mode="wrap")
        g = np.pad(g, ((0, 0), (radius, radius), (0, 0)), mode="wrap")
        values = _ssim_core(p, g, kernel)[radius:-radius, radius:-radius]
        row_ssim = np.mean(values, axis=(1, 2), dtype=np.float64)
        ssim_sum += float(row_ssim.sum())
        weighted_ssim_sum += float(np.sum(row_ssim * weights[y0:y1]))
    mse = squared_sum / height
    return {
        "psnr": float("inf") if mse == 0 else -10.0 * math.log10(mse),
        "ssim": ssim_sum / height,
        "ws_psnr": float("inf") if weighted_squared_sum == 0 else -10.0 * math.log10(weighted_squared_sum),
        "ws_ssim": weighted_ssim_sum,
    }
