"""
Color sampling utilities for text and surrounding background pixels.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def rgb_to_hsv(r: int, g: int, b: int) -> list[float]:
    arr = np.uint8([[[b, g, r]]])  # OpenCV BGR
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0]
    # OpenCV: H 0-179, S/V 0-255 → normalize
    return [round(float(hsv[0]) * 2.0, 2), round(float(hsv[1]) / 255.0, 4), round(float(hsv[2]) / 255.0, 4)]


def relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG relative luminance."""

    def channel(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> float:
    l1 = relative_luminance(*fg)
    l2 = relative_luminance(*bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return round((lighter + 0.05) / (darker + 0.05), 3)


def _clip_box(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def _dominant_color(pixels: np.ndarray, k: int = 3) -> tuple[int, int, int]:
    """
    Dominant RGB from Nx3 uint8 pixel array (already RGB).
    Falls back to mean if too few pixels.
    """
    if pixels is None or len(pixels) == 0:
        return (0, 0, 0)
    if len(pixels) < 8:
        mean = pixels.mean(axis=0)
        return (int(mean[0]), int(mean[1]), int(mean[2]))

    # Quantize lightly for stability
    data = pixels.astype(np.float32)
    kk = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(data, kk, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=kk)
    center = centers[int(np.argmax(counts))]
    return (int(center[0]), int(center[1]), int(center[2]))


def extract_text_and_background_colors(
    bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    """
    Estimate dominant text color (ink) and local background color around a text box.

    Strategy:
      - Background: sample a padded ring outside the bbox
      - Text: pixels inside bbox that differ strongly from background (ink mask)
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return {
            "font_rgb": (20, 20, 20),
            "bg_rgb": (255, 255, 255),
            "opacity": 1.0,
        }

    pad = max(4, int(min(x2 - x1, y2 - y1) * 0.15))
    ox1, oy1, ox2, oy2 = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)

    outer = bgr[oy1:oy2, ox1:ox2].copy()
    # Mask out the inner text box from outer region
    inner_mask = np.zeros(outer.shape[:2], dtype=np.uint8)
    ix1, iy1 = x1 - ox1, y1 - oy1
    ix2, iy2 = x2 - ox1, y2 - oy1
    inner_mask[iy1:iy2, ix1:ix2] = 255
    ring_mask = cv2.bitwise_not(inner_mask)

    ring_pixels_bgr = outer[ring_mask > 0]
    if len(ring_pixels_bgr) < 5:
        ring_pixels_bgr = bgr[oy1:oy2, ox1:ox2].reshape(-1, 3)

    # Convert to RGB for API
    ring_rgb = ring_pixels_bgr[:, ::-1]
    bg_rgb = _dominant_color(ring_rgb, k=2)

    # Ink mask inside text box: pixels far from background in Lab space
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"font_rgb": (20, 20, 20), "bg_rgb": bg_rgb, "opacity": 1.0}

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_bgr = np.uint8([[bg_rgb[::-1]]])  # RGB→BGR
    bg_lab = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab, axis=2)

    # Adaptive threshold: top-contrast pixels as ink
    thr = max(18.0, float(np.percentile(dist, 70)))
    ink_mask = dist >= thr

    # Also use Otsu on grayscale difference for thin strokes
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg_gray = int(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
    diff = cv2.absdiff(gray, np.uint8(np.full_like(gray, bg_gray)))
    _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_mask = np.logical_or(ink_mask, otsu > 0)

    ink_pixels_bgr = roi[ink_mask]
    if len(ink_pixels_bgr) < 5:
        # Fallback: darkest or most saturated vs bg
        flat = roi.reshape(-1, 3)
        lumin = 0.114 * flat[:, 0] + 0.587 * flat[:, 1] + 0.299 * flat[:, 2]
        if bg_gray > 128:
            pick = flat[np.argsort(lumin)[: max(10, len(flat) // 10)]]
        else:
            pick = flat[np.argsort(-lumin)[: max(10, len(flat) // 10)]]
        ink_pixels_bgr = pick

    # Drop neon highlight fill pixels (lime/yellow accents behind text) from ink
    if len(ink_pixels_bgr) >= 8:
        bch = ink_pixels_bgr[:, 0].astype(np.float32)
        gch = ink_pixels_bgr[:, 1].astype(np.float32)
        rch = ink_pixels_bgr[:, 2].astype(np.float32)
        mx = np.maximum(np.maximum(rch, gch), bch)
        mn = np.minimum(np.minimum(rch, gch), bch)
        sat = (mx - mn) / np.maximum(mx, 1.0)
        lum = 0.299 * rch + 0.587 * gch + 0.114 * bch
        highlight = (sat > 0.28) & (lum > 90) & (gch >= rch + 15) & (gch >= bch + 10)
        kept = ink_pixels_bgr[~highlight]
        if len(kept) >= 5:
            ink_pixels_bgr = kept

    # Prefer median of mid-tone ink (avoid near-black AA fringes dominating gray text)
    ink_rgb = ink_pixels_bgr[:, ::-1].astype(np.float64)
    lumin = 0.299 * ink_rgb[:, 0] + 0.587 * ink_rgb[:, 1] + 0.114 * ink_rgb[:, 2]
    order = np.argsort(lumin)
    # On light backgrounds, bias toward lighter ink percentiles so soft gray
    # body copy isn't pulled toward black AA edges.
    if bg_gray >= 200:
        lo = max(0, int(len(order) * 0.35))
        hi = max(lo + 1, int(len(order) * 0.85))
    else:
        lo = max(0, int(len(order) * 0.15))
        hi = max(lo + 1, int(len(order) * 0.75))
    core = ink_rgb[order[lo:hi]]
    if len(core) >= 3:
        med = np.median(core, axis=0)
        font_rgb = (int(med[0]), int(med[1]), int(med[2]))
    else:
        font_rgb = _dominant_color(ink_pixels_bgr[:, ::-1], k=2)

    # If residual green cast remains (highlight bleed), snap dark ink to near-black
    fr, fg, fb = font_rgb
    fmx, fmn = max(fr, fg, fb), min(fr, fg, fb)
    fsat = (fmx - fmn) / max(fmx, 1)
    flum = 0.299 * fr + 0.587 * fg + 0.114 * fb
    if fsat > 0.18 and fg >= fr + 12 and fg >= fb + 8 and flum < 120:
        font_rgb = (22, 22, 22)

    # Opacity estimate: how solid ink pixels are vs background blend
    if len(ink_pixels_bgr) > 0:
        ink_mean = ink_pixels_bgr.mean(axis=0)
        bg_vec = np.array(bg_rgb[::-1], dtype=np.float32)
        target = np.array(font_rgb[::-1], dtype=np.float32)
        denom = np.linalg.norm(target - bg_vec) + 1e-6
        # If ink is between bg and solid target, estimate alpha
        proj = np.dot(ink_mean - bg_vec, target - bg_vec) / (denom * denom)
        opacity = float(np.clip(proj, 0.35, 1.0))
    else:
        opacity = 1.0

    return {
        "font_rgb": font_rgb,
        "bg_rgb": bg_rgb,
        "opacity": round(opacity, 3),
        "ink_mask": ink_mask,
        "roi_origin": (x1, y1),
    }
