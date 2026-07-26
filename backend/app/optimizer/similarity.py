"""
Image similarity metrics — SSIM, PSNR, pixel/edge/color differences.

SSIM is implemented with NumPy (no scikit-image dependency).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img.astype(np.float64)
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return g


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Structural Similarity Index in [0, 1]."""
    a = _to_gray_f32(img_a)
    b = _to_gray_f32(img_b)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(a, -1, window)
    mu2 = cv2.filter2D(b, -1, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(a * a, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(b * b, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(a * b, -1, window) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.clip(np.mean(ssim_map), 0.0, 1.0))


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def pixel_difference(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Mean absolute pixel difference normalized to [0, 1]."""
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a - b)) / 255.0)


def edge_difference(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Normalized mean absolute difference of Canny edges."""
    ga = _to_gray_f32(img_a).astype(np.uint8)
    gb = _to_gray_f32(img_b).astype(np.uint8)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    ea = cv2.Canny(ga, 50, 150).astype(np.float64)
    eb = cv2.Canny(gb, 50, 150).astype(np.float64)
    return float(np.mean(np.abs(ea - eb)) / 255.0)


def color_difference(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Mean Lab ΔE-ish distance normalized roughly to [0, 1]."""
    a = img_a
    b = img_b
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float64)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float64)
    delta = np.sqrt(np.sum((la - lb) ** 2, axis=2))
    # Lab channels ~0–255 in OpenCV; max theoretical ~441
    return float(np.clip(np.mean(delta) / 100.0, 0.0, 1.0))


def mean_region_color(img: np.ndarray, bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    """Mean BGR color inside bbox [x1,y1,x2,y2]."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0])
    crop = img[y1:y2, x1:x2]
    return crop.reshape(-1, 3).mean(axis=0)


def region_color_delta(img_a: np.ndarray, img_b: np.ndarray, bbox: list[float]) -> float:
    ca = mean_region_color(img_a, bbox)
    cb = mean_region_color(img_b, bbox)
    return float(np.linalg.norm(ca - cb) / (255.0 * np.sqrt(3)))


def compute_all_metrics(
    original: np.ndarray,
    rendered: np.ndarray,
    *,
    text_bbox_diff: float = 0.0,
    alignment_diff: float = 0.0,
    object_position_error: float = 0.0,
    spacing_error: float = 0.0,
) -> dict[str, Any]:
    ssim = compute_ssim(original, rendered)
    psnr = compute_psnr(original, rendered)
    pix = pixel_difference(original, rendered)
    edge = edge_difference(original, rendered)
    color = color_difference(original, rendered)

    # Editable-PDF quality: structure matters as much as pixels (fonts rarely match 1:1).
    structural = (
        0.35 * (1.0 - min(object_position_error, 1.0))
        + 0.30 * (1.0 - min(text_bbox_diff, 1.0))
        + 0.20 * (1.0 - min(alignment_diff, 1.0))
        + 0.15 * (1.0 - min(spacing_error, 1.0))
    )
    perceptual = (
        0.40 * ssim
        + 0.20 * min(psnr / 40.0, 1.0)
        + 0.15 * (1.0 - pix)
        + 0.15 * (1.0 - edge)
        + 0.10 * (1.0 - color)
    )
    overall = float(np.clip((0.55 * structural + 0.45 * perceptual) * 100.0, 0.0, 100.0))

    return {
        "ssim": round(ssim, 6),
        "psnr": round(psnr, 3),
        "pixel_difference": round(pix, 6),
        "edge_difference": round(edge, 6),
        "color_difference": round(color, 6),
        "text_bbox_difference": round(float(text_bbox_diff), 6),
        "alignment_difference": round(float(alignment_diff), 6),
        "object_position_error": round(float(object_position_error), 6),
        "spacing_error": round(float(spacing_error), 6),
        "overall_similarity": round(overall, 3),
    }
