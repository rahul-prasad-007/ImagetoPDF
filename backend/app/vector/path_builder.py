"""
Approximate contours as SVG-style path commands / Bezier control points.
No SVG file export — path data only.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.vector.models import ControlPoint, PathData


def contour_to_path(
    contour: np.ndarray,
    epsilon_ratio: float = 0.012,
    closed: bool = True,
) -> PathData:
    """
    Fit a simplified polyline / quadratic Bezier chain to a contour.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, epsilon_ratio * peri), True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return PathData(commands="", control_points=[], closed=closed, confidence=0.0)

    # Build SVG-like path: M + sequence of Q (quadratic) or L
    commands: list[str] = [f"M {pts[0, 0]:.2f} {pts[0, 1]:.2f}"]
    controls: list[ControlPoint] = [ControlPoint(x=float(pts[0, 0]), y=float(pts[0, 1]))]

    if len(pts) == 2:
        commands.append(f"L {pts[1, 0]:.2f} {pts[1, 1]:.2f}")
        controls.append(ControlPoint(x=float(pts[1, 0]), y=float(pts[1, 1])))
    else:
        # Quadratic Bezier through midpoints for smoother curves
        for i in range(1, len(pts) - 1):
            cx, cy = float(pts[i, 0]), float(pts[i, 1])
            nx, ny = float(pts[i + 1, 0]), float(pts[i + 1, 1])
            mx, my = (cx + nx) / 2.0, (cy + ny) / 2.0
            commands.append(f"Q {cx:.2f} {cy:.2f} {mx:.2f} {my:.2f}")
            controls.append(ControlPoint(x=cx, y=cy))
            controls.append(ControlPoint(x=mx, y=my))
        # Close to first point
        last = pts[-1]
        first = pts[0]
        commands.append(f"Q {last[0]:.2f} {last[1]:.2f} {first[0]:.2f} {first[1]:.2f}")
        controls.append(ControlPoint(x=float(last[0]), y=float(last[1])))

    if closed:
        commands.append("Z")

    # Confidence from how closely approx matches original
    area_orig = abs(cv2.contourArea(contour))
    area_approx = abs(cv2.contourArea(approx))
    if area_orig > 1:
        ratio = min(area_approx, area_orig) / max(area_approx, area_orig)
        conf = float(np.clip(ratio * 100.0, 50.0, 99.0))
    else:
        conf = 75.0

    return PathData(
        commands=" ".join(commands),
        control_points=controls,
        closed=closed,
        confidence=conf,
    )


def points_to_path(points: list[tuple[float, float]], closed: bool = False) -> PathData:
    if not points:
        return PathData(commands="", confidence=0.0)
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    ctrls = [ControlPoint(x=points[0][0], y=points[0][1])]
    for x, y in points[1:]:
        cmds.append(f"L {x:.2f} {y:.2f}")
        ctrls.append(ControlPoint(x=x, y=y))
    if closed:
        cmds.append("Z")
    return PathData(
        commands=" ".join(cmds),
        control_points=ctrls,
        closed=closed,
        confidence=90.0,
    )


def classify_decorative_path(
    contour: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    """
    Heuristic: wave / ribbon / curved band / generic path.
    """
    x1, y1, x2, y2 = bbox
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = w / h
    area = abs(cv2.contourArea(contour))
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    peri = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    path = contour_to_path(contour, epsilon_ratio=0.01, closed=True)

    if aspect > 3.5 and extent < 0.55:
        kind = "WAVE"
    elif aspect > 2.2 and 0.35 < extent < 0.75:
        kind = "RIBBON"
    elif 0.15 < circularity < 0.55 and extent < 0.65:
        kind = "CURVED_BAND"
    else:
        kind = "PATH"

    return {
        "decorative_kind": kind,
        "path": path,
        "extent": float(extent),
        "circularity": float(circularity),
    }
