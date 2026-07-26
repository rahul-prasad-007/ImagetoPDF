"""
OpenCV shape detection for vector reconstruction.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from app.vector.models import ControlPoint, VectorType
from app.vector.path_builder import classify_decorative_path, contour_to_path


def _hex(bgr: tuple[int, int, int] | np.ndarray) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def _fill_of(bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "#CCCCCC"
    # Sample interior (avoid border stroke)
    pad_y = max(1, (y2 - y1) // 6)
    pad_x = max(1, (x2 - x1) // 6)
    inner = roi[pad_y : max(pad_y + 1, roi.shape[0] - pad_y), pad_x : max(pad_x + 1, roi.shape[1] - pad_x)]
    sample = inner if inner.size else roi
    med = np.median(sample.reshape(-1, 3), axis=0)
    return _hex(med)


def _stroke_of(bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> str | None:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 6:
        return None
    edge = np.concatenate(
        [
            roi[0, :].reshape(-1, 3),
            roi[-1, :].reshape(-1, 3),
            roi[:, 0].reshape(-1, 3),
            roi[:, -1].reshape(-1, 3),
        ]
    )
    med = np.median(edge, axis=0)
    fill = _fill_of(bgr, bbox)
    stroke = _hex(med)
    # If stroke ≈ fill, omit
    fr = int(fill[1:3], 16)
    fg = int(fill[3:5], 16)
    fb = int(fill[5:7], 16)
    if abs(fr - med[2]) + abs(fg - med[1]) + abs(fb - med[0]) < 40:
        return None
    return stroke


def _estimate_corner_radius(contour: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    rect_peri = 2 * (w + h)
    peri = cv2.arcLength(contour, True)
    # Rounded rects have slightly shorter perimeter than sharp rect for same bbox
    diff = max(0.0, rect_peri - peri)
    r = min(w, h) * 0.08 + diff * 0.15
    return float(np.clip(r, 0.0, min(w, h) * 0.45))


def _is_arrow(approx: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
    if len(approx) not in (5, 6, 7):
        return False
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    aspect = max(w, h) / max(1.0, min(w, h))
    return aspect > 1.6


def detect_shapes(bgr: np.ndarray) -> list[dict[str, Any]]:
    """Detect rectangles, rounded rects, circles, ellipses, polygons, lines, decorative paths."""
    h, w = bgr.shape[:2]
    page_area = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shapes: list[dict[str, Any]] = []

    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < page_area * 0.0008 and area < 120:
            continue
        if area > page_area * 0.95:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        verts = len(approx)
        fill = _fill_of(bgr, bbox)
        stroke = _stroke_of(bgr, bbox)
        aspect = bw / max(bh, 1)

        # Circle / ellipse via fit — require substantial contour
        if len(cnt) >= 8 and area > page_area * 0.002:
            (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
            ellipse_area = math.pi * (ma / 2) * (mi / 2)
            if ellipse_area > 0 and 0.82 < area / ellipse_area < 1.18:
                ratio = min(ma, mi) / max(ma, mi)
                if min(ma, mi) < 18:
                    pass
                elif ratio > 0.9:
                    shapes.append(
                        {
                            "type": VectorType.CIRCLE,
                            "bbox": bbox,
                            "fill_color": fill,
                            "stroke_color": stroke,
                            "stroke_width": 1.5 if stroke else 0.0,
                            "corner_radius": 0.0,
                            "rotation": float(angle),
                            "confidence": 92.0,
                            "layer": 3,
                            "meta": {"source": "shape_detector", "cx": cx, "cy": cy, "r": (ma + mi) / 4},
                        }
                    )
                    continue
                elif ratio > 0.45:
                    shapes.append(
                        {
                            "type": VectorType.ELLIPSE,
                            "bbox": bbox,
                            "fill_color": fill,
                            "stroke_color": stroke,
                            "stroke_width": 1.5 if stroke else 0.0,
                            "corner_radius": 0.0,
                            "rotation": float(angle),
                            "confidence": 90.0,
                            "layer": 3,
                            "meta": {"source": "shape_detector"},
                        }
                    )
                    continue

        if verts == 3:
            pts = [ControlPoint(x=float(p[0][0]), y=float(p[0][1])) for p in approx]
            shapes.append(
                {
                    "type": VectorType.TRIANGLE,
                    "bbox": bbox,
                    "fill_color": fill,
                    "stroke_color": stroke,
                    "stroke_width": 1.0 if stroke else 0.0,
                    "corner_radius": 0.0,
                    "confidence": 93.0,
                    "layer": 3,
                    "points": pts,
                    "meta": {"source": "shape_detector"},
                }
            )
            continue

        if verts == 4:
            # Rectangle vs rounded
            rect_area = float(bw * bh)
            extent = area / rect_area if rect_area else 0
            corner_r = _estimate_corner_radius(cnt, bbox)
            # Angular check
            pts = approx.reshape(4, 2).astype(np.float32)
            angles_ok = True
            for i in range(4):
                a = pts[i] - pts[(i - 1) % 4]
                b = pts[(i + 1) % 4] - pts[i]
                cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
                ang = abs(math.degrees(math.acos(float(np.clip(cosang, -1, 1)))))
                if abs(ang - 90) > 22:
                    angles_ok = False
                    break

            if angles_ok and extent > 0.78:
                if corner_r > min(bw, bh) * 0.06:
                    shapes.append(
                        {
                            "type": VectorType.ROUNDED_RECTANGLE,
                            "bbox": bbox,
                            "fill_color": fill,
                            "stroke_color": stroke,
                            "stroke_width": 1.5 if stroke else 0.0,
                            "corner_radius": round(corner_r, 2),
                            "confidence": 94.0,
                            "layer": 2 if area > page_area * 0.04 else 3,
                            "meta": {"source": "shape_detector", "extent": extent},
                        }
                    )
                else:
                    # Border-like thin frame?
                    if extent < 0.35 and min(bw, bh) > 40:
                        vtype = VectorType.BORDER
                    else:
                        vtype = VectorType.RECTANGLE
                    shapes.append(
                        {
                            "type": vtype,
                            "bbox": bbox,
                            "fill_color": fill if vtype != VectorType.BORDER else None,
                            "stroke_color": stroke or fill,
                            "stroke_width": 2.0 if vtype == VectorType.BORDER else (1.0 if stroke else 0.0),
                            "corner_radius": 0.0,
                            "confidence": 95.0,
                            "layer": 2 if area > page_area * 0.04 else 3,
                            "meta": {"source": "shape_detector", "extent": extent},
                        }
                    )
                continue

            if _is_arrow(approx, bbox):
                shapes.append(
                    {
                        "type": VectorType.ARROW,
                        "bbox": bbox,
                        "fill_color": fill,
                        "stroke_color": stroke,
                        "stroke_width": 1.0 if stroke else 0.0,
                        "corner_radius": 0.0,
                        "confidence": 80.0,
                        "layer": 3,
                        "path": contour_to_path(cnt),
                        "meta": {"source": "shape_detector"},
                    }
                )
                continue

            shapes.append(
                {
                    "type": VectorType.POLYGON,
                    "bbox": bbox,
                    "fill_color": fill,
                    "stroke_color": stroke,
                    "stroke_width": 1.0 if stroke else 0.0,
                    "corner_radius": 0.0,
                    "confidence": 88.0,
                    "layer": 3,
                    "points": [ControlPoint(x=float(p[0][0]), y=float(p[0][1])) for p in approx],
                    "meta": {"source": "shape_detector"},
                }
            )
            continue

        if 5 <= verts <= 8 and _is_arrow(approx, bbox):
            shapes.append(
                {
                    "type": VectorType.ARROW,
                    "bbox": bbox,
                    "fill_color": fill,
                    "stroke_color": stroke,
                    "stroke_width": 1.0 if stroke else 0.0,
                    "confidence": 78.0,
                    "layer": 3,
                    "path": contour_to_path(cnt),
                    "meta": {"source": "shape_detector"},
                }
            )
            continue

        # Decorative / curved shapes (sparse)
        if verts > 8 and area > page_area * 0.004:
            deco = classify_decorative_path(cnt, bbox)
            kind = deco["decorative_kind"]
            if kind == "PATH" and deco["extent"] > 0.7:
                continue
            vtype = {
                "WAVE": VectorType.WAVE,
                "RIBBON": VectorType.RIBBON,
                "CURVED_BAND": VectorType.CURVED_BAND,
                "PATH": VectorType.PATH,
            }[kind]
            shapes.append(
                {
                    "type": vtype,
                    "bbox": bbox,
                    "fill_color": fill,
                    "stroke_color": stroke,
                    "stroke_width": 1.0 if stroke else 0.0,
                    "corner_radius": 0.0,
                    "confidence": float(deco["path"].confidence),
                    "layer": 3,
                    "path": deco["path"],
                    "meta": {"source": "shape_detector", "decorative": kind},
                }
            )

    # Hough lines → LINE vectors (strict + NMS)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=90,
        minLineLength=max(50, w // 12),
        maxLineGap=6,
    )
    line_cands: list[dict[str, Any]] = []
    if lines is not None:
        for ln in lines[:40]:
            x1, y1, x2, y2 = map(float, ln[0])
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                continue
            bx1, by1 = min(x1, x2), min(y1, y2)
            bx2, by2 = max(x1, x2), max(y1, y2)
            if abs(bx2 - bx1) < 3:
                bx1 -= 1
                bx2 += 1
            if abs(by2 - by1) < 3:
                by1 -= 1
                by2 += 1
            bbox = (bx1, by1, bx2, by2)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            stroke = _fill_of(bgr, bbox)
            line_cands.append(
                {
                    "type": VectorType.LINE,
                    "bbox": bbox,
                    "fill_color": None,
                    "stroke_color": stroke,
                    "stroke_width": 2.0,
                    "corner_radius": 0.0,
                    "rotation": angle,
                    "confidence": 88.0,
                    "layer": 3,
                    "points": [ControlPoint(x=x1, y=y1), ControlPoint(x=x2, y=y2)],
                    "meta": {"source": "hough_line", "length": length},
                }
            )

    # Suppress near-duplicate lines
    line_cands.sort(key=lambda d: d["meta"]["length"], reverse=True)
    kept_lines: list[dict[str, Any]] = []
    for cand in line_cands:
        if any(_line_similar(cand, k) for k in kept_lines):
            continue
        kept_lines.append(cand)
    shapes.extend(kept_lines[:20])

    return shapes


def _line_similar(a: dict[str, Any], b: dict[str, Any], ang_tol: float = 8.0, dist_tol: float = 12.0) -> bool:
    ra = float(a.get("rotation") or 0) % 180
    rb = float(b.get("rotation") or 0) % 180
    dang = min(abs(ra - rb), 180 - abs(ra - rb))
    if dang > ang_tol:
        return False
    ax = (a["bbox"][0] + a["bbox"][2]) / 2
    ay = (a["bbox"][1] + a["bbox"][3]) / 2
    bx = (b["bbox"][0] + b["bbox"][2]) / 2
    by = (b["bbox"][1] + b["bbox"][3]) / 2
    return math.hypot(ax - bx, ay - by) < dist_tol or (
        abs(ax - bx) < dist_tol and abs(ay - by) < dist_tol * 3
    )
