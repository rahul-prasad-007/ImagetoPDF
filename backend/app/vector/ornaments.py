"""
Decorative ornament detection — hearts + divider lines.

Conservative by design: never invent hearts over glyphs. Prefer OCR icon
tokens and text-free vertical gaps (title separators / signature).
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from app.vector.models import ControlPoint, PathData, VectorType


_HEART_OCR_TOKENS = {"❤", "♥", "♡", "❣️", "<3", "❤︎"}
# Lone "3" is ONLY accepted as a heart when it sits in a clear separator band
_HEART_OCR_AMBIGUOUS = {"3"}


def _bbox_from_ocr_block(block: dict[str, Any]) -> Optional[tuple[float, float, float, float]]:
    bb = block.get("bbox")
    if not bb:
        return None
    if isinstance(bb, list) and bb and isinstance(bb[0], (list, tuple)):
        xs = [float(p[0]) for p in bb]
        ys = [float(p[1]) for p in bb]
        return min(xs), min(ys), max(xs), max(ys)
    if len(bb) >= 4:
        return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    return None


def _ocr_boxes(ocr: Optional[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for block in (ocr or {}).get("text_blocks") or []:
        box = _bbox_from_ocr_block(block)
        if box:
            out.append(box)
    return out


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _max_text_overlap(
    box: tuple[float, float, float, float],
    text_boxes: list[tuple[float, float, float, float]],
) -> float:
    return max((_overlap_ratio(box, tb) for tb in text_boxes), default=0.0)


def _in_separator_band(
    y_mid: float,
    text_boxes: list[tuple[float, float, float, float]],
    page_h: float,
    *,
    min_gap: float = 10.0,
    ignore_box: tuple[float, float, float, float] | None = None,
) -> bool:
    """True if y sits in a vertical gap between text rows (or above/below all text)."""
    boxes = [
        b
        for b in text_boxes
        if ignore_box is None
        or abs(b[0] - ignore_box[0]) > 1
        or abs(b[1] - ignore_box[1]) > 1
        or abs(b[2] - ignore_box[2]) > 1
        or abs(b[3] - ignore_box[3]) > 1
    ]
    if not boxes:
        return True
    rows = sorted(((b[1], b[3]) for b in boxes), key=lambda r: r[0])
    if y_mid < rows[0][0] - 2:
        return True
    if y_mid > rows[-1][1] + 2:
        return True
    for i in range(len(rows) - 1):
        gap_top = rows[i][1]
        gap_bot = rows[i + 1][0]
        if gap_bot - gap_top >= min_gap and gap_top + 1 <= y_mid <= gap_bot - 1:
            return True
    # Near page top title ornaments (within top 22%)
    if y_mid < page_h * 0.22:
        for tb in boxes:
            # Strictly inside another text row
            if tb[1] + 1 < y_mid < tb[3] - 1:
                return False
        return True
    return False


def _is_page_centered(cx: float, page_w: float, tol: float = 0.14) -> bool:
    return abs(cx - page_w / 2.0) <= page_w * tol


def _heart_path(x1: float, y1: float, x2: float, y2: float) -> PathData:
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)
    unit = [
        (0.50, 0.92),
        (0.10, 0.55),
        (0.05, 0.28),
        (0.22, 0.08),
        (0.50, 0.28),
        (0.78, 0.08),
        (0.95, 0.28),
        (0.90, 0.55),
        (0.50, 0.92),
    ]
    pts = [(x1 + u * w, y1 + v * h) for u, v in unit]
    cmds = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for i in range(1, len(pts) - 1):
        cx, cy = pts[i]
        nx, ny = pts[i + 1]
        mx, my = (cx + nx) / 2, (cy + ny) / 2
        cmds.append(f"Q {cx:.2f} {cy:.2f} {mx:.2f} {my:.2f}")
    cmds.append(f"L {pts[-1][0]:.2f} {pts[-1][1]:.2f}")
    cmds.append("Z")
    return PathData(
        commands=" ".join(cmds),
        control_points=[ControlPoint(x=p[0], y=p[1]) for p in pts],
        closed=True,
        confidence=92.0,
    )


def _sample_dark_stroke(bgr: np.ndarray, y: float, x1: float, x2: float) -> str:
    h, w = bgr.shape[:2]
    yy = int(np.clip(round(y), 0, h - 1))
    xa, xb = int(np.clip(min(x1, x2), 0, w - 1)), int(np.clip(max(x1, x2), 0, w - 1))
    if xb <= xa:
        return "#2A2A2A"
    strip = bgr[max(0, yy - 2) : min(h, yy + 3), xa:xb]
    if strip.size == 0:
        return "#2A2A2A"
    flat = strip.reshape(-1, 3)
    lum = 0.114 * flat[:, 0] + 0.587 * flat[:, 1] + 0.299 * flat[:, 2]
    dark = flat[np.argsort(lum)[: max(5, len(flat) // 8)]]
    med = np.median(dark, axis=0)
    return f"#{int(med[2]):02X}{int(med[1]):02X}{int(med[0]):02X}"


def _divider_lines_around_heart(
    bgr: np.ndarray,
    hx1: float,
    hy1: float,
    hx2: float,
    hy2: float,
    page_w: float,
) -> list[dict[str, Any]]:
    cx = (hx1 + hx2) / 2.0
    cy = (hy1 + hy2) / 2.0
    hw = max(8.0, hx2 - hx1)
    gap = max(6.0, hw * 0.35)
    line_len = max(40.0, min(page_w * 0.16, hw * 4.5))
    stroke = _sample_dark_stroke(bgr, cy, cx - gap - line_len, cx + gap + line_len)
    try:
        r, g, b = int(stroke[1:3], 16), int(stroke[3:5], 16), int(stroke[5:7], 16)
        if 0.299 * r + 0.587 * g + 0.114 * b > 90:
            stroke = "#2C2C2C"
    except Exception:
        stroke = "#2C2C2C"

    out = []
    for box in (
        (cx - gap - line_len, cy - 1.0, cx - gap, cy + 1.0),
        (cx + gap, cy - 1.0, cx + gap + line_len, cy + 1.0),
    ):
        x1, y1, x2, y2 = box
        out.append(
            {
                "type": VectorType.LINE,
                "bbox": box,
                "fill_color": None,
                "stroke_color": stroke,
                "stroke_width": 1.6,
                "corner_radius": 0.0,
                "rotation": 0.0,
                "confidence": 94.0,
                "layer": 3,
                "points": [
                    ControlPoint(x=x1, y=(y1 + y2) / 2),
                    ControlPoint(x=x2, y=(y1 + y2) / 2),
                ],
                "meta": {"source": "heart_divider", "ornament": "line"},
            }
        )
    return out


def _make_heart(
    box: tuple[float, float, float, float],
    *,
    ornament: str,
    stroke_width: float = 1.8,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    return {
        "type": VectorType.HEART,
        "bbox": box,
        "fill_color": None,
        "stroke_color": "#2A2A2A",
        "stroke_width": stroke_width,
        "corner_radius": 0.0,
        "confidence": 95.0,
        "layer": 4,
        "path": _heart_path(x1, y1, x2, y2),
        "meta": {"source": "ornament", "ornament": ornament},
    }


def detect_heart_ornaments(
    bgr: np.ndarray,
    ocr: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Recover a few decorative hearts (title dividers / signature), never glyph overlays.
    """
    if bgr is None:
        return []
    h, w = bgr.shape[:2]
    text_boxes = _ocr_boxes(ocr)
    ornaments: list[dict[str, Any]] = []
    heart_boxes: list[tuple[float, float, float, float]] = []

    # --- 1) Explicit heart unicode / <3 tokens only ---
    for block in (ocr or {}).get("text_blocks") or []:
        text = str(block.get("text") or "").strip()
        if text not in _HEART_OCR_TOKENS and text not in _HEART_OCR_AMBIGUOUS:
            continue
        box = _bbox_from_ocr_block(block)
        if not box:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            continue
        if bw > w * 0.08 or bh > h * 0.045:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > 2.2:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Ambiguous "3": must be centered + in separator band + tiny icon
        if text in _HEART_OCR_AMBIGUOUS:
            if not _is_page_centered(cx, float(w), tol=0.12):
                continue
            if not _in_separator_band(
                cy, text_boxes, float(h), min_gap=6.0, ignore_box=box
            ):
                continue
            same_row = [
                tb
                for tb in text_boxes
                if abs(((tb[1] + tb[3]) / 2) - cy) < max(bh, 14)
                and not (abs(tb[0] - x1) < 1 and abs(tb[1] - y1) < 1)
            ]
            if len(same_row) >= 2:
                continue
        pad = max(2.0, min(bw, bh) * 0.1)
        hx1, hy1 = max(0.0, x1 - pad), max(0.0, y1 - pad)
        hx2, hy2 = min(float(w), x2 + pad), min(float(h), y2 + pad)
        cand = (hx1, hy1, hx2, hy2)
        foreign = [
            tb
            for tb in text_boxes
            if not (abs(tb[0] - x1) < 1 and abs(tb[1] - y1) < 1)
        ]
        if _max_text_overlap(cand, foreign) > 0.22:
            continue
        heart_boxes.append(cand)

    # --- 2) Geometric hearts ONLY in text-masked separator bands (max 2) ---
    if len(heart_boxes) < 2:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Mask out all OCR text so letter counters never match
        masked = gray.copy()
        for tb in text_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in tb]
            pad = 3
            cv2.rectangle(
                masked,
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(w - 1, x2 + pad), min(h - 1, y2 + pad)),
                255,
                -1,
            )
        blur = cv2.GaussianBlur(masked, (3, 3), 0)
        bw_img = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
        )
        bw_img = cv2.morphologyEx(bw_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        geo: list[tuple[float, tuple[float, float, float, float]]] = []
        for cnt in contours:
            area = abs(cv2.contourArea(cnt))
            if area < 80 or area > (w * h) * 0.002:
                continue
            x, y, bw_, bh_ = cv2.boundingRect(cnt)
            if bw_ < 12 or bh_ < 12 or bw_ > 48 or bh_ > 48:
                continue
            aspect = bw_ / max(bh_, 1)
            if not (0.75 <= aspect <= 1.25):
                continue
            cx, cy = x + bw_ / 2.0, y + bh_ / 2.0
            if not _is_page_centered(cx, float(w), tol=0.16):
                continue
            box = (float(x), float(y), float(x + bw_), float(y + bh_))
            if not _in_separator_band(
                cy, text_boxes, float(h), min_gap=12.0, ignore_box=box
            ):
                continue
            if _max_text_overlap(box, text_boxes) > 0.08:
                continue
            # Stronger cleft: mid valley clearly below both peaks
            mask = np.zeros((bh_, bw_), dtype=np.uint8)
            cnt_local = cnt.copy()
            cnt_local[:, 0, 0] -= x
            cnt_local[:, 0, 1] -= y
            cv2.drawContours(mask, [cnt_local], -1, 255, -1)
            top = mask[: max(1, bh_ // 3), :]
            col_sums = (top > 0).sum(axis=0).astype(np.float32)
            if col_sums.size < 8:
                continue
            third = max(1, bw_ // 3)
            left_peak = float(np.max(col_sums[:third]))
            mid_valley = float(np.max(col_sums[third : 2 * third]))
            right_peak = float(np.max(col_sums[2 * third :]))
            if left_peak < 2 or right_peak < 2:
                continue
            if mid_valley > min(left_peak, right_peak) * 0.75:
                continue
            # Bottom should come to a point (narrower than mid)
            bot = mask[int(bh_ * 0.7) :, :]
            if bot.size and (bot > 0).sum(axis=0).max() > bw_ * 0.55:
                continue
            score = (left_peak + right_peak) - 2 * mid_valley
            if any(abs(cx - (a + c) / 2) < 24 and abs(cy - (b + d) / 2) < 24 for a, b, c, d in heart_boxes):
                continue
            geo.append((score, box))
        geo.sort(key=lambda t: t[0], reverse=True)
        for _, box in geo[: max(0, 2 - len(heart_boxes))]:
            heart_boxes.append(box)

    # Cap total decorative hearts (dividers); signature added separately
    heart_boxes = heart_boxes[:3]

    for hx1, hy1, hx2, hy2 in heart_boxes:
        box = (hx1, hy1, hx2, hy2)
        ornaments.append(_make_heart(box, ornament="heart"))
        cx = (hx1 + hx2) / 2.0
        cy = (hy1 + hy2) / 2.0
        # Ignore any OCR box overlapping this heart when testing separator band
        ignore = None
        for tb in text_boxes:
            if _overlap_ratio(box, tb) > 0.3:
                ignore = tb
                break
        if (
            (hx2 - hx1) >= 12
            and (hy2 - hy1) >= 12
            and _is_page_centered(cx, float(w), tol=0.16)
            and _in_separator_band(cy, text_boxes, float(h), min_gap=4.0, ignore_box=ignore)
        ):
            ornaments.extend(_divider_lines_around_heart(bgr, hx1, hy1, hx2, hy2, float(w)))

    # --- 3) Optional signature heart to the RIGHT of a short bottom name ---
    # Only a single-word name (e.g. "Krisha"), never sentence fragments like "Notice them."
    name_cands: list[tuple[float, tuple[float, float, float, float]]] = []
    for block in (ocr or {}).get("text_blocks") or []:
        text = str(block.get("text") or "").strip()
        if not text or len(text) > 18:
            continue
        if text.isdigit() or text in _HEART_OCR_TOKENS | _HEART_OCR_AMBIGUOUS:
            continue
        # Reject sentences / multi-word lines
        words = [w for w in text.replace(".", " ").replace(",", " ").split() if w]
        if len(words) != 1:
            continue
        name = words[0]
        if not name.isalpha() or not (2 <= len(name) <= 16):
            continue
        # Names are usually Title Case or lowercase script — skip ALL CAPS brands
        if name.isupper() and len(name) > 3:
            continue
        box = _bbox_from_ocr_block(block)
        if not box:
            continue
        x1, y1, x2, y2 = box
        if y1 < h * 0.82:
            continue
        if x1 > w * 0.55:
            continue
        name_cands.append((y1, box))

    if name_cands:
        # Prefer the lowest name on the page (true sign-off)
        name_cands.sort(key=lambda t: t[0], reverse=True)
        _, (x1, y1, x2, y2) = name_cands[0]
        sx1 = x2 + max(4.0, (x2 - x1) * 0.08)
        sy1 = y1 + (y2 - y1) * 0.15
        size = max(10.0, min(22.0, (y2 - y1) * 0.7))
        sx2, sy2 = sx1 + size, sy1 + size
        sig = (sx1, sy1, min(float(w - 1), sx2), min(float(h - 1), sy2))
        if _max_text_overlap(sig, text_boxes) <= 0.05 and not any(
            abs(((a + c) / 2) - (sig[0] + sig[2]) / 2) < 30
            and abs(((b + d) / 2) - (sig[1] + sig[3]) / 2) < 30
            for a, b, c, d in heart_boxes
        ):
            ornaments.append(_make_heart(sig, ornament="signature_heart", stroke_width=1.4))

    return ornaments


def filter_parchment_noise(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop beige tear/edge false arrows & polygons from textured paper."""
    kept: list[dict[str, Any]] = []
    noisy = {
        VectorType.ARROW,
        VectorType.TRIANGLE,
        VectorType.POLYGON,
        VectorType.RIBBON,
        VectorType.WAVE,
        VectorType.CURVED_BAND,
    }
    page_area = 0.0
    for item in raw:
        bbox = item.get("bbox")
        if bbox and len(bbox) >= 4:
            page_area = max(page_area, abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

    for item in raw:
        vtype = item.get("type")
        if isinstance(vtype, str):
            try:
                vtype = VectorType(vtype)
            except Exception:
                vtype = None
        meta = item.get("meta") or {}
        if meta.get("ornament") or meta.get("source") in {"ornament", "heart_divider"}:
            kept.append(item)
            continue
        if vtype == VectorType.HEART:
            kept.append(item)
            continue
        fill = item.get("fill_color") or item.get("fill")
        bbox = item.get("bbox") or (0, 0, 0, 0)
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if len(bbox) >= 4 else 0.0
        if isinstance(fill, str) and fill.startswith("#") and len(fill) >= 7:
            try:
                r, g, b = int(fill[1:3], 16), int(fill[3:5], 16), int(fill[5:7], 16)
            except ValueError:
                r = g = b = 0
            mx, mn = max(r, g, b), min(r, g, b)
            sat = (mx - mn) / max(mx, 1)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            warm = lum >= 140 and sat <= 0.35 and r >= g - 5 and g >= b
            if (
                vtype in {VectorType.PANEL, VectorType.RECTANGLE, VectorType.ROUNDED_RECTANGLE}
                and warm
                and page_area > 0
                and area < page_area * 0.85
                and area > page_area * 0.08
            ):
                continue
            if vtype in noisy and warm and area < 25000:
                continue
        kept.append(item)
    return kept
