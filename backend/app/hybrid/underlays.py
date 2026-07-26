"""Hybrid raster underlays — logos, stamps, header art cropped from the source."""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from app.layout.detectors import detect_image_regions

logger = logging.getLogger(__name__)


def _ocr_boxes(ocr: Optional[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for b in (ocr or {}).get("text_blocks") or []:
        pts = b.get("bbox") or []
        if len(pts) >= 4 and isinstance(pts[0], (list, tuple)):
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        elif len(pts) >= 4:
            boxes.append((float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])))
    return boxes


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _text_coverage(
    bbox: tuple[float, float, float, float],
    text_boxes: list[tuple[float, float, float, float]],
) -> float:
    ax1, ay1, ax2, ay2 = bbox
    area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    covered = 0.0
    for tb in text_boxes:
        ix1 = max(ax1, tb[0])
        iy1 = max(ay1, tb[1])
        ix2 = min(ax2, tb[2])
        iy2 = min(ay2, tb[3])
        covered += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return min(1.0, covered / area)


def _header_band_underlay(
    bgr: np.ndarray,
    text_boxes: list[tuple[float, float, float, float]],
    *,
    max_height_ratio: float = 0.22,
) -> Optional[dict[str, Any]]:
    h, w = bgr.shape[:2]
    band_h = max(40, int(h * max_height_ratio))
    band = bgr[0:band_h, :]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    colorful = (sat > 45) & (val > 40) & (val < 250)
    ratio = float(np.count_nonzero(colorful) / max(1, colorful.size))
    if ratio < 0.08:
        return None

    row_score = colorful.mean(axis=1)
    active = np.where(row_score > 0.05)[0]
    if len(active) < 8:
        return None
    y1 = int(max(0, active[0] - 4))
    y2 = int(min(band_h, active[-1] + 8))
    if y2 - y1 < 28:
        return None

    bbox = (0.0, float(y1), float(w), float(y2))
    coverage = _text_coverage(bbox, text_boxes)
    if coverage > 0.55:
        return None

    return {
        "kind": "header_art",
        "bbox": bbox,
        "confidence": float(np.clip(55 + ratio * 80, 55, 92)),
        "meta": {
            "source": "hybrid_header",
            "color_ratio": round(ratio, 3),
            "text_coverage": round(coverage, 3),
        },
    }


def detect_hybrid_underlays(
    bgr: np.ndarray,
    ocr: Optional[dict[str, Any]] = None,
    *,
    mode: str = "designed_invoice",
    max_underlays: int = 8,
) -> list[dict[str, Any]]:
    """Find logo / stamp / photo / header regions for PDF image XObjects."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    text_boxes = _ocr_boxes(ocr)
    h, w = bgr.shape[:2]
    page_area = float(h * w)

    candidates: list[dict[str, Any]] = []

    for det in detect_image_regions(bgr, gray, text_boxes):
        x1, y1, x2, y2 = det.bbox
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        coverage = _text_coverage(det.bbox, text_boxes)
        if coverage > 0.28:
            continue
        if area < page_area * 0.004 or area > page_area * 0.35:
            continue
        if mode == "ruled_form":
            if y1 > h * 0.35 and area > page_area * 0.04:
                continue
            if bw > w * 0.55 and bh > h * 0.18:
                continue
        kind = det.kind if det.kind in {"logo", "icon", "photo", "qr_code", "image"} else "image"
        if mode == "ruled_form" and y1 < h * 0.28 and x1 < w * 0.45:
            kind = "logo"
        candidates.append(
            {
                "kind": kind,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": float(det.confidence or 0.7) * 100.0,
                "meta": {**(det.meta or {}), "source": "hybrid_texture"},
            }
        )

    if mode == "designed_invoice":
        header = _header_band_underlay(bgr, text_boxes)
        if header:
            candidates.append(header)

    candidates.sort(
        key=lambda c: (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for cur in candidates:
        if any(_iou(cur["bbox"], k["bbox"]) >= 0.55 for k in kept):
            continue
        kept.append(cur)
        if len(kept) >= max_underlays:
            break

    logger.info("Hybrid underlays mode=%s count=%d", mode, len(kept))
    return kept


def underlays_to_scene_dicts(
    underlays: list[dict[str, Any]],
    *,
    image_path: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    start_id: int = 9000,
) -> list[dict[str, Any]]:
    """Convert underlay detections into scene-object dicts (IMAGE/LOGO)."""
    objects: list[dict[str, Any]] = []
    for i, u in enumerate(underlays):
        x1, y1, x2, y2 = u["bbox"]
        sx = x1 * scale_x + offset_x
        sy = y1 * scale_y + offset_y
        sw = max(1.0, (x2 - x1) * scale_x)
        sh = max(1.0, (y2 - y1) * scale_y)
        kind = str(u.get("kind") or "image")
        otype = "LOGO" if kind == "logo" else "IMAGE"
        crop = {
            "x": float(x1),
            "y": float(y1),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        }
        objects.append(
            {
                "id": start_id + i,
                "parent": None,
                "children": [],
                "layer": 5,
                "type": otype,
                "x": round(sx, 2),
                "y": round(sy, 2),
                "width": round(sw, 2),
                "height": round(sh, 2),
                "rotation": 0.0,
                "opacity": 1.0,
                "visibility": True,
                "locked": False,
                "image_path": image_path,
                "crop": crop,
                "source": {"hybrid": True, "kind": kind},
                "meta": {
                    "hybrid_underlay": True,
                    "kind": kind,
                    "confidence": u.get("confidence"),
                    **(u.get("meta") or {}),
                    "render": {
                        "image": {
                            "image_path": image_path,
                            "crop_x": crop["x"],
                            "crop_y": crop["y"],
                            "crop_width": crop["width"],
                            "crop_height": crop["height"],
                        }
                    },
                },
            }
        )
    return objects
