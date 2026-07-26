"""
Gradient estimation for color regions — linear / radial (no rendering).
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from app.vector.models import GradientKind, GradientSpec


def _hex(bgr: tuple[float, float, float] | np.ndarray) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"


def estimate_gradient(
    bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    mask: Optional[np.ndarray] = None,
) -> Optional[GradientSpec]:
    """
    Estimate linear or radial gradient inside bbox.
    Returns None if the region is essentially flat.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    roi = bgr[y1:y2, x1:x2].astype(np.float32)
    rh, rw = roi.shape[:2]

    # Sample strips for linear gradient along X and Y
    left = np.median(roi[:, : max(1, rw // 8)], axis=(0, 1))
    right = np.median(roi[:, -max(1, rw // 8) :], axis=(0, 1))
    top = np.median(roi[: max(1, rh // 8), :], axis=(0, 1))
    bottom = np.median(roi[-max(1, rh // 8) :, :], axis=(0, 1))

    dx = float(np.linalg.norm(right - left))
    dy = float(np.linalg.norm(bottom - top))
    flat_std = float(np.mean(np.std(roi.reshape(-1, 3), axis=0)))

    if max(dx, dy) < 18 and flat_std < 12:
        return None  # flat fill

    # Radial: compare center vs corners
    cy, cx = rh // 2, rw // 2
    r = max(4, min(rh, rw) // 10)
    center = np.median(roi[cy - r : cy + r, cx - r : cx + r], axis=(0, 1))
    corners = np.stack(
        [
            np.median(roi[:r, :r], axis=(0, 1)),
            np.median(roi[:r, -r:], axis=(0, 1)),
            np.median(roi[-r:, :r], axis=(0, 1)),
            np.median(roi[-r:, -r:], axis=(0, 1)),
        ]
    )
    corner_mean = np.mean(corners, axis=0)
    radial_delta = float(np.linalg.norm(center - corner_mean))

    if radial_delta > max(dx, dy) * 1.15 and radial_delta > 22:
        return GradientSpec(
            kind=GradientKind.RADIAL,
            angle=0.0,
            start_color=_hex(center),
            end_color=_hex(corner_mean),
            center_x=float(x1 + cx),
            center_y=float(y1 + cy),
            confidence=float(np.clip(60 + radial_delta * 0.5, 60, 96)),
        )

    if dx >= dy:
        angle = 0.0
        start, end = left, right
        conf = dx
    else:
        angle = 90.0
        start, end = top, bottom
        conf = dy

    # Refine angle via Sobel on luminance
    gray = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mean_gx, mean_gy = float(np.mean(gx)), float(np.mean(gy))
    if abs(mean_gx) + abs(mean_gy) > 1e-3:
        angle = float(np.degrees(np.arctan2(mean_gy, mean_gx)) % 180)

    return GradientSpec(
        kind=GradientKind.LINEAR,
        angle=round(angle, 2),
        start_color=_hex(start),
        end_color=_hex(end),
        confidence=float(np.clip(55 + conf * 0.4, 55, 97)),
    )


def attach_gradients(
    bgr: np.ndarray,
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mutate region dicts with gradient specs when detected."""
    out: list[dict[str, Any]] = []
    for reg in regions:
        g = estimate_gradient(bgr, reg["bbox"])
        item = dict(reg)
        if g is not None:
            item["gradient"] = g
            item["type"] = item.get("type")  # may promote later
            item["meta"] = dict(item.get("meta") or {})
            item["meta"]["has_gradient"] = True
        out.append(item)
    return out
