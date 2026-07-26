"""
OpenCV hybrid detectors for document layout understanding.

Windows-friendly: contours, morphology, Hough lines, connected components,
color-panel segmentation, and texture-based image-region detection.

No Detectron2 / LayoutParser required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _sample_fill_bgr(
    bgr: np.ndarray, bbox: tuple[float, float, float, float]
) -> list[int]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return [200, 200, 200]
    pad_y = max(1, (y2 - y1) // 5)
    pad_x = max(1, (x2 - x1) // 5)
    inner = roi[
        pad_y : max(pad_y + 1, roi.shape[0] - pad_y),
        pad_x : max(pad_x + 1, roi.shape[1] - pad_x),
    ]
    sample = inner if inner.size else roi
    med = np.median(sample.reshape(-1, 3), axis=0)
    return [int(med[0]), int(med[1]), int(med[2])]


def _is_highlight_bgr(bgr_color: list[int] | tuple[int, ...] | None) -> bool:
    """Bright accent fills (lime/yellow highlights behind text)."""
    if not bgr_color or len(bgr_color) < 3:
        return False
    b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / max(mx, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 70 or lum > 245 or sat < 0.25:
        return False
    if g >= r + 25 and g >= b + 15:
        return True
    if r >= 160 and g >= 160 and b < 120 and sat > 0.3:
        return True
    return False


def _text_coverage_ratio(
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


@dataclass
class RawDetection:
    """Intermediate detection before final typing / hierarchy."""

    kind: str  # shape | line | circle | ellipse | panel | image_region | text
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2
    rotation: float = 0.0
    confidence: float = 1.0
    text: Optional[str] = None
    ocr_block_ids: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


def _clip_bbox(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[float, float, float, float]:
    return (
        float(max(0, min(x1, w - 1))),
        float(max(0, min(y1, h - 1))),
        float(max(0, min(x2, w - 1))),
        float(max(0, min(y2, h - 1))),
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    pad: float = 2.0,
) -> bool:
    return (
        inner[0] >= outer[0] - pad
        and inner[1] >= outer[1] - pad
        and inner[2] <= outer[2] + pad
        and inner[3] <= outer[3] + pad
    )


def _nms(dets: list[RawDetection], iou_thresh: float = 0.55) -> list[RawDetection]:
    """Greedy NMS preferring larger / higher-confidence detections."""
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: (d.confidence, d.area), reverse=True)
    kept: list[RawDetection] = []
    for det in ordered:
        if any(_iou(det.bbox, k.bbox) >= iou_thresh and det.kind == k.kind for k in kept):
            continue
        kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Line detection (Hough)
# ---------------------------------------------------------------------------
def detect_lines(gray: np.ndarray, page_w: int, page_h: int) -> list[RawDetection]:
    edges = cv2.Canny(gray, 60, 160, apertureSize=3)
    min_len = max(80, int(min(page_w, page_h) * 0.18))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=min_len,
        maxLineGap=8,
    )
    out: list[RawDetection] = []
    if lines is None:
        return out

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        orientation = (
            "horizontal"
            if angle < 12 or angle > 168
            else "vertical"
            if 78 < angle < 102
            else "diagonal"
        )
        if orientation == "diagonal":
            continue

        # Skip page-border frame fragments (handled as rectangles)
        margin = min(page_w, page_h) * 0.03
        near_border = (
            min(x1, x2) < margin
            or min(y1, y2) < margin
            or max(x1, x2) > page_w - margin
            or max(y1, y2) > page_h - margin
        )
        if near_border and length > min(page_w, page_h) * 0.5:
            continue

        pad = 2.0
        bx1, by1 = min(x1, x2) - pad, min(y1, y2) - pad
        bx2, by2 = max(x1, x2) + pad, max(y1, y2) + pad
        bbox = _clip_bbox(bx1, by1, bx2, by2, page_w, page_h)
        out.append(
            RawDetection(
                kind="line",
                bbox=bbox,
                rotation=float(math.degrees(math.atan2(dy, dx))),
                confidence=0.85,
                meta={"orientation": orientation, "length": round(length, 1)},
            )
        )

    # Stronger merge for near-duplicate lines
    return _nms(out, 0.25)[:20]


# ---------------------------------------------------------------------------
# Contour-based shapes (rects, rounded rects, circles, ellipses)
# ---------------------------------------------------------------------------
def detect_shapes(bgr: np.ndarray, gray: np.ndarray) -> list[RawDetection]:
    page_h, page_w = gray.shape[:2]
    page_area = float(page_w * page_h)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[RawDetection] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < page_area * 0.0015 or area > page_area * 0.92:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 40:
            continue
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12:
            continue

        bbox = _clip_bbox(x, y, x + w, y + h, page_w, page_h)
        aspect = w / float(h)
        extent = area / float(w * h + 1e-6)
        circularity = 4.0 * math.pi * area / (peri * peri + 1e-6)

        # Circle / ellipse (avoid huge page frames)
        if area < page_area * 0.55:
            if circularity > 0.72 and 0.75 <= aspect <= 1.33:
                out.append(
                    RawDetection(
                        kind="circle",
                        bbox=bbox,
                        confidence=min(1.0, circularity),
                        meta={"circularity": round(circularity, 3)},
                    )
                )
                continue
            if circularity > 0.55 and (aspect < 0.75 or aspect > 1.33) and extent > 0.55:
                out.append(
                    RawDetection(
                        kind="ellipse",
                        bbox=bbox,
                        confidence=min(1.0, circularity + 0.1),
                        meta={"circularity": round(circularity, 3), "aspect": round(aspect, 3)},
                    )
                )
                continue

        # Rectangle / rounded rectangle (including large borders/frames)
        if len(approx) <= 6 and extent > 0.45:
            rect_area = float(w * h)
            fill_ratio = area / (rect_area + 1e-6)
            # Hollow frame: contour area small relative to bbox but 4-ish corners
            is_frame = fill_ratio < 0.35 and area > page_area * 0.01
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            corner_pts = [
                (x + 2, y + 2),
                (x + w - 3, y + 2),
                (x + 2, y + h - 3),
                (x + w - 3, y + h - 3),
            ]
            corner_inside = sum(
                1
                for cx, cy in corner_pts
                if 0 <= cy < page_h and 0 <= cx < page_w and mask[cy, cx] > 0
            )
            is_rounded = (not is_frame) and corner_inside <= 2 and fill_ratio > 0.7
            # Prefer rectangle for large near-page outlines
            if len(approx) >= 4 or is_frame or (0.6 <= aspect <= 1.7 and extent > 0.5):
                kind = "rounded_rectangle" if is_rounded else "rectangle"
                color = _sample_fill_bgr(bgr, bbox)
                out.append(
                    RawDetection(
                        kind=kind,
                        bbox=bbox,
                        confidence=0.8 if is_rounded else 0.88,
                        meta={
                            "extent": round(extent, 3),
                            "fill_ratio": round(fill_ratio, 3),
                            "vertices": int(len(approx)),
                            "frame": is_frame,
                            "color_bgr": color,
                            "highlight": _is_highlight_bgr(color),
                        },
                    )
                )
                continue

        # Thin long contour → decorative line-like element
        if (w > page_w * 0.25 and h < 10) or (h > page_h * 0.25 and w < 10):
            out.append(
                RawDetection(
                    kind="line",
                    bbox=bbox,
                    confidence=0.7,
                    meta={"source": "contour"},
                )
            )

    return _nms(out, 0.5)


# ---------------------------------------------------------------------------
# Background / color panels via quantization + connected components
# ---------------------------------------------------------------------------
def detect_background_panels(bgr: np.ndarray) -> list[RawDetection]:
    page_h, page_w = bgr.shape[:2]
    page_area = float(page_w * page_h)

    # Downscale for speed
    scale = 0.35 if max(page_w, page_h) > 1200 else 0.5
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    data = small.reshape((-1, 3)).astype(np.float32)
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])
    centers_u8 = np.uint8(centers)

    out: list[RawDetection] = []
    for i in range(k):
        mask = np.uint8(labels == i) * 255
        # Clean small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lab in range(1, n_labels):
            x, y, w, h, area = stats[lab]
            # Map back to full resolution
            fx1 = x / scale
            fy1 = y / scale
            fx2 = (x + w) / scale
            fy2 = (y + h) / scale
            area_full = (fx2 - fx1) * (fy2 - fy1)
            if area_full < page_area * 0.04 or area_full > page_area * 0.95:
                continue
            # Skip very thin strips (likely borders already covered as lines)
            if (fx2 - fx1) < page_w * 0.15 and (fy2 - fy1) < page_h * 0.08:
                continue
            color = centers_u8[i].tolist()
            bbox = _clip_bbox(fx1, fy1, fx2, fy2, page_w, page_h)
            out.append(
                RawDetection(
                    kind="panel",
                    bbox=bbox,
                    confidence=0.75,
                    meta={
                        "color_bgr": color,
                        "role": "background_panel",
                    },
                )
            )

    return _nms(out, 0.6)


# ---------------------------------------------------------------------------
# Non-text image regions (photo / logo / icon / QR heuristic)
# ---------------------------------------------------------------------------
def detect_image_regions(
    bgr: np.ndarray,
    gray: np.ndarray,
    text_boxes: list[tuple[float, float, float, float]],
) -> list[RawDetection]:
    page_h, page_w = gray.shape[:2]
    page_area = float(page_w * page_h)

    # Texture map via Laplacian variance in sliding windows is expensive;
    # use MSER-ish approach: high-gradient blobs minus text masks.
    edges = cv2.Canny(gray, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dense = cv2.dilate(edges, kernel, iterations=2)
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Suppress known text areas
    text_mask = np.zeros(gray.shape, dtype=np.uint8)
    for tb in text_boxes:
        x1, y1, x2, y2 = map(int, tb)
        cv2.rectangle(text_mask, (x1, y1), (x2, y2), 255, -1)
    text_mask = cv2.dilate(text_mask, np.ones((7, 7), np.uint8), iterations=1)
    dense[text_mask > 0] = 0

    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[RawDetection] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(w * h)
        if area < page_area * 0.008 or area > page_area * 0.55:
            continue
        if w < 28 or h < 28:
            continue

        bbox = _clip_bbox(x, y, x + w, y + h, page_w, page_h)
        # Skip if mostly overlapping OCR text (union coverage, not single-box IoU)
        coverage = _text_coverage_ratio(bbox, text_boxes)
        if coverage > 0.22:
            continue
        if any(_iou(bbox, tb) > 0.45 for tb in text_boxes):
            continue

        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        mean_val = float(roi.mean())
        aspect = w / float(h)

        # Flat near-white/gray bands with only text-edge texture are not photos
        if mean_val > 200 and coverage > 0.08:
            continue
        if mean_val > 185 and aspect > 2.5 and coverage > 0.05:
            continue

        # Classification heuristics
        if 0.85 <= aspect <= 1.15 and 40 < w < page_w * 0.22 and lap_var > 80:
            kind = "qr_code" if lap_var > 400 and mean_val < 200 else "icon"
        elif area < page_area * 0.04 and max(w, h) < page_w * 0.2:
            kind = "logo" if y < page_h * 0.35 else "icon"
        elif lap_var > 120:
            kind = "photo"
        else:
            kind = "image"

        out.append(
            RawDetection(
                kind=kind,
                bbox=bbox,
                confidence=0.7,
                meta={
                    "laplacian_var": round(lap_var, 1),
                    "mean": round(mean_val, 1),
                    "aspect": round(aspect, 3),
                },
            )
        )

    return _nms(out, 0.5)


# ---------------------------------------------------------------------------
# Text region grouping from OCR blocks
# ---------------------------------------------------------------------------
def ocr_blocks_to_text_detections(ocr_payload: dict[str, Any]) -> list[RawDetection]:
    """Convert OCR JSON text_blocks into RawDetection list with reading order."""
    blocks = ocr_payload.get("text_blocks") or []
    page = ocr_payload.get("page") or {}
    page_h = float(page.get("height") or 1)

    dets: list[RawDetection] = []
    heights: list[float] = []
    for b in blocks:
        bbox_pts = b.get("bbox") or []
        if len(bbox_pts) >= 4:
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            # Fallback from center/size if present
            cx, cy = float(b.get("center_x", 0)), float(b.get("center_y", 0))
            w, h = float(b.get("width", 0)), float(b.get("height", 0))
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        heights.append(max(1.0, y2 - y1))
        dets.append(
            RawDetection(
                kind="text",
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                rotation=float(b.get("rotation") or 0.0),
                confidence=float(b.get("confidence") or 0.0),
                text=str(b.get("text") or ""),
                ocr_block_ids=[int(b.get("id"))] if b.get("id") is not None else [],
                meta={
                    "line": b.get("line"),
                    "paragraph": b.get("paragraph"),
                    "word": b.get("word"),
                },
            )
        )

    if not dets:
        return []

    median_h = float(np.median(heights)) if heights else 20.0

    for det in dets:
        det.kind = "text_block"
        det.meta["median_text_height"] = median_h
        det.meta["paragraph_hint"] = det.meta.get("paragraph")

    # TITLE = tallest text near the top (prefer higher placement over raw area,
    # so wide lower title lines don't steal the headline).
    # Never promote numbered list lines to title.
    upper = [
        d
        for d in dets
        if d.center[1] / page_h < 0.42
        and not (str(d.text or "").strip()[:2].rstrip(".").isdigit())
    ]
    pool = upper if upper else [
        d for d in dets if not (str(d.text or "").strip()[:2].rstrip(".").isdigit())
    ] or list(dets)
    if pool:
        # Rank by height, then by top position (smaller y wins)
        title = max(pool, key=lambda d: (d.height, -d.bbox[1], d.width))
        title.kind = "title"
        rest_upper = [d for d in upper if d is not title]
        if rest_upper:
            sub = max(rest_upper, key=lambda d: (d.height, -d.bbox[1]))
            if sub.height >= median_h * 1.05 and sub.bbox[1] < title.bbox[3] + median_h * 2:
                sub.kind = "subtitle"

    return dets


def _is_list_marker(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.startswith(("-", "•", "*", "·", "–", "—")):
        return True
    # Lone numerals / step numbers
    if len(t) <= 3 and t.rstrip(".").isdigit():
        return True
    if len(t) <= 4 and t[:-1].isdigit() and t[-1] in ".)":
        return True
    return False


def merge_paragraph_groups(text_dets: list[RawDetection]) -> list[RawDetection]:
    """
    Keep OCR lines as separate text objects by default.

    Only merge consecutive wrapped paragraph lines that share a left edge and
    have a tight vertical gap — never merge numbered lists or page-spanning groups.
    """
    titles = [d for d in text_dets if d.kind in {"title", "subtitle"}]
    others = [d for d in text_dets if d.kind not in {"title", "subtitle"}]
    others = sorted(others, key=lambda m: (m.bbox[1], m.bbox[0]))

    if not others:
        return list(titles)

    median_h = float(
        np.median([max(1.0, m.height) for m in others])
    ) if others else 20.0
    max_gap = median_h * 0.55
    max_left_delta = max(12.0, median_h * 0.35)

    merged: list[RawDetection] = list(titles)
    cluster: list[RawDetection] = []

    def flush(cluster_members: list[RawDetection]) -> None:
        if not cluster_members:
            return
        if len(cluster_members) == 1:
            m = cluster_members[0]
            m.kind = "text_block"
            merged.append(m)
            return
        # Refuse merges that look like numbered/bullet lists
        if any(_is_list_marker(m.text or "") for m in cluster_members):
            for m in cluster_members:
                m.kind = "text_block"
                merged.append(m)
            return
        span_h = max(m.bbox[3] for m in cluster_members) - min(m.bbox[1] for m in cluster_members)
        if span_h > median_h * 3.5:
            for m in cluster_members:
                m.kind = "text_block"
                merged.append(m)
            return

        x1 = min(m.bbox[0] for m in cluster_members)
        y1 = min(m.bbox[1] for m in cluster_members)
        x2 = max(m.bbox[2] for m in cluster_members)
        y2 = max(m.bbox[3] for m in cluster_members)
        text = "\n".join(m.text or "" for m in cluster_members if m.text)
        ocr_ids = [i for m in cluster_members for i in m.ocr_block_ids]
        avg_conf = float(np.mean([m.confidence for m in cluster_members]))
        merged.append(
            RawDetection(
                kind="paragraph",
                bbox=(x1, y1, x2, y2),
                confidence=avg_conf,
                text=text,
                ocr_block_ids=ocr_ids,
                meta={"member_count": len(cluster_members)},
            )
        )

    for d in others:
        if not cluster:
            cluster = [d]
            continue
        prev = cluster[-1]
        gap = d.bbox[1] - prev.bbox[3]
        left_delta = abs(d.bbox[0] - prev.bbox[0])
        # Same wrapped paragraph: tight gap, aligned left, similar height
        same_wrap = (
            gap >= -median_h * 0.15
            and gap <= max_gap
            and left_delta <= max_left_delta
            and not _is_list_marker(d.text or "")
            and not _is_list_marker(prev.text or "")
            and abs(d.height - prev.height) <= median_h * 0.55
        )
        if same_wrap:
            cluster.append(d)
        else:
            flush(cluster)
            cluster = [d]
    flush(cluster)

    return merged
