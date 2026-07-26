"""
Color region extraction — flat fills, panels, large background areas.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.vector.models import VectorType


def _hex(bgr: tuple[int, int, int] | np.ndarray) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def detect_color_regions(
    bgr: np.ndarray,
    max_regions: int = 24,
    min_area_ratio: float = 0.004,
) -> list[dict[str, Any]]:
    """
    Segment large near-flat color panels via quantization + connected components.
    """
    h, w = bgr.shape[:2]
    page_area = float(h * w)
    min_area = max(80.0, page_area * min_area_ratio)

    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    quant = (blur // 24) * 24
    gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    solid = cv2.bitwise_not(edges)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    regions: list[dict[str, Any]] = []

    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 8 or bh < 8:
            continue

        mask = (labels == i).astype(np.uint8)
        pixels = bgr[mask > 0]
        if pixels.size == 0:
            continue
        fill_bgr = tuple(int(v) for v in np.median(pixels.reshape(-1, 3), axis=0))
        fill = _hex(fill_bgr)
        std = float(np.mean(np.std(pixels.reshape(-1, 3).astype(np.float32), axis=0)))
        aspect = bw / max(bh, 1)

        if area > page_area * 0.45:
            vtype = VectorType.COLOR_REGION
            layer = 1
        elif area > page_area * 0.03:
            vtype = VectorType.PANEL
            layer = 2
        else:
            vtype = VectorType.COLOR_REGION
            layer = 2

        regions.append(
            {
                "type": vtype,
                "bbox": (float(x), float(y), float(x + bw), float(y + bh)),
                "fill_color": fill,
                "stroke_color": None,
                "stroke_width": 0.0,
                "corner_radius": 0.0,
                "confidence": float(np.clip(95.0 - std * 0.8, 70.0, 99.0)),
                "layer": layer,
                "meta": {
                    "area": area,
                    "flatness_std": round(std, 2),
                    "aspect": round(aspect, 3),
                    "source": "color_region",
                },
            }
        )

    regions.sort(key=lambda r: r["meta"]["area"], reverse=True)
    return regions[:max_regions]
