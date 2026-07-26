"""
Form / bill-book grid reconstruction.

Uses morphological H/V line extraction + OCR header cues so cash memos,
invoices, and similar tabular pages keep aligned column rules, measured
stroke widths, fill dotted lines, and a proprietor badge when present.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import cv2
import numpy as np

from app.vector.models import ControlPoint, VectorType

_FORM_KEYWORDS = {
    "bill",
    "cash",
    "memo",
    "invoice",
    "particulars",
    "quantity",
    "rate",
    "amount",
    "total",
    "sl.no",
    "sl no",
    "customer",
    "lorry",
    "gst",
    "proprietor",
    "signature",
}


def is_form_like_page(ocr: Optional[dict[str, Any]]) -> bool:
    texts = []
    for b in (ocr or {}).get("text_blocks") or []:
        t = str(b.get("text") or "").strip().lower()
        if t:
            texts.append(t)
    if not texts:
        return False
    blob = " ".join(texts)
    hits = sum(1 for k in _FORM_KEYWORDS if k in blob)
    # Table header row signal
    header_hits = sum(
        1
        for k in ("particulars", "quantity", "rate", "amount", "sl.no", "sl no")
        if k in blob
    )
    return hits >= 3 or header_hits >= 3


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


def _ocr_boxes(ocr: Optional[dict[str, Any]]) -> list[tuple[str, tuple[float, float, float, float]]]:
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    for block in (ocr or {}).get("text_blocks") or []:
        text = str(block.get("text") or "").strip()
        box = _bbox_from_ocr_block(block)
        if text and box:
            out.append((text, box))
    return out


def _cluster_1d(vals: list[float], tol: float) -> list[float]:
    if not vals:
        return []
    ordered = sorted(vals)
    clusters: list[list[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(sum(c) / len(c)) for c in clusters]


def _sample_stroke(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> str:
    h, w = bgr.shape[:2]
    xa, xb = int(np.clip(min(x1, x2), 0, w - 1)), int(np.clip(max(x1, x2), 0, w - 1))
    ya, yb = int(np.clip(min(y1, y2), 0, h - 1)), int(np.clip(max(y1, y2), 0, h - 1))
    if xb <= xa:
        xb = min(w - 1, xa + 1)
    if yb <= ya:
        yb = min(h - 1, ya + 1)
    # Dilate thin samples
    pad = 2
    roi = bgr[max(0, ya - pad) : min(h, yb + pad + 1), max(0, xa - pad) : min(w, xb + pad + 1)]
    if roi.size == 0:
        return "#1A1A1A"
    flat = roi.reshape(-1, 3)
    lum = 0.114 * flat[:, 0] + 0.587 * flat[:, 1] + 0.299 * flat[:, 2]
    dark = flat[np.argsort(lum)[: max(8, len(flat) // 10)]]
    med = np.median(dark, axis=0)
    return f"#{int(med[2]):02X}{int(med[1]):02X}{int(med[0]):02X}"


def _line_item(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    stroke_width: float,
    dashed: bool = False,
    role: str = "grid",
) -> dict[str, Any]:
    bx1, by1 = min(x1, x2), min(y1, y2)
    bx2, by2 = max(x1, x2), max(y1, y2)
    # Ensure non-zero bbox for thin lines
    if abs(bx2 - bx1) < 1.0:
        bx1 -= 0.5
        bx2 += 0.5
    if abs(by2 - by1) < 1.0:
        by1 -= 0.5
        by2 += 0.5
    meta: dict[str, Any] = {
        "source": "form_grid",
        "form": True,
        "role": role,
    }
    if dashed:
        meta["dash"] = [1.5, 2.2]
        meta["ornament"] = "dotted"
    return {
        "type": VectorType.LINE,
        "bbox": (bx1, by1, bx2, by2),
        "fill_color": None,
        "stroke_color": stroke,
        "stroke_width": float(stroke_width),
        "corner_radius": 0.0,
        "rotation": 0.0 if abs(y2 - y1) < abs(x2 - x1) else 90.0,
        "confidence": 96.0,
        "layer": 4,
        "points": [ControlPoint(x=x1, y=y1), ControlPoint(x=x2, y=y2)],
        "meta": meta,
    }


def _extract_morph_lines(
    binary: np.ndarray,
    *,
    horizontal: bool,
    min_len: int,
) -> list[tuple[float, float, float, float, float]]:
    """Return (x1,y1,x2,y2,thickness) segments."""
    h, w = binary.shape[:2]
    if horizontal:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 28), 1))
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 35)))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    # Slight dilate to reconnect gaps
    opened = cv2.dilate(opened, k, iterations=1)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs: list[tuple[float, float, float, float, float]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if horizontal:
            if bw < min_len or bh > 14:
                continue
            y_mid = y + bh / 2.0
            segs.append((float(x), y_mid, float(x + bw), y_mid, float(max(1.0, bh))))
        else:
            if bh < min_len or bw > 14:
                continue
            x_mid = x + bw / 2.0
            segs.append((x_mid, float(y), x_mid, float(y + bh), float(max(1.0, bw))))
    return segs


def _extend_segment(
    segs: list[tuple[float, float, float, float, float]],
    *,
    horizontal: bool,
    snap_tol: float,
) -> list[tuple[float, float, float, float, float]]:
    """Cluster nearly-collinear segments and merge into longer rules."""
    if not segs:
        return []
    if horizontal:
        keys = _cluster_1d([s[1] for s in segs], snap_tol)
        out: list[tuple[float, float, float, float, float]] = []
        for y in keys:
            group = [s for s in segs if abs(s[1] - y) <= snap_tol]
            if not group:
                continue
            x1 = min(s[0] for s in group)
            x2 = max(s[2] for s in group)
            th = float(np.median([s[4] for s in group]))
            out.append((x1, y, x2, y, th))
        return out
    keys = _cluster_1d([s[0] for s in segs], snap_tol)
    out = []
    for x in keys:
        group = [s for s in segs if abs(s[0] - x) <= snap_tol]
        if not group:
            continue
        y1 = min(s[1] for s in group)
        y2 = max(s[3] for s in group)
        th = float(np.median([s[4] for s in group]))
        out.append((x, y1, x, y2, th))
    return out


def _header_column_xs(
    ocr_items: list[tuple[str, tuple[float, float, float, float]]],
    page_w: float,
) -> list[float]:
    """Infer vertical column lines from table header labels."""
    labels = {
        "sl.no": None,
        "sl no": None,
        "particulars": None,
        "quantity": None,
        "rate": None,
        "amount": None,
        "rs.": None,
        "rs": None,
        "p": None,
    }
    for text, box in ocr_items:
        key = re.sub(r"\s+", " ", text.strip().lower())
        key = key.replace("sl. no", "sl.no").replace("sl no.", "sl.no")
        for lab in list(labels.keys()):
            if key == lab or key.startswith(lab):
                labels[lab] = box
    # Prefer the densest header band
    boxes = [b for b in labels.values() if b]
    if len(boxes) < 3:
        return []
    ys = [(b[1] + b[3]) / 2 for b in boxes]
    y_med = float(np.median(ys))
    band = [b for b in boxes if abs(((b[1] + b[3]) / 2) - y_med) < 40]
    if len(band) < 3:
        band = boxes

    # Column edges: left of first, mid between neighbors, right of last amount block
    ordered = sorted(band, key=lambda b: b[0])
    xs = [ordered[0][0] - 6.0]
    for i in range(len(ordered) - 1):
        gap_mid = (ordered[i][2] + ordered[i + 1][0]) / 2.0
        xs.append(gap_mid)
    xs.append(min(page_w - 4.0, ordered[-1][2] + 8.0))

    # Amount Rs/P split if both present
    rs = labels.get("rs.") or labels.get("rs")
    p = labels.get("p")
    if rs and p:
        xs.append((rs[2] + p[0]) / 2.0)

    return _cluster_1d(xs, tol=10.0)


def _detect_dotted_fill_lines(
    gray: np.ndarray,
    bgr: np.ndarray,
    ocr_items: list[tuple[str, tuple[float, float, float, float]]],
) -> list[dict[str, Any]]:
    """Recover dotted underline fields after Customer / Address / Lorry / Amount in Word."""
    h, w = gray.shape[:2]
    out: list[dict[str, Any]] = []
    for text, box in ocr_items:
        low = text.strip().lower().rstrip(".")
        # Exact-ish field labels only — avoid "Signature of Customer's"
        is_customer = low.startswith("customer") and "signature" not in low
        is_address = low.startswith("address")
        is_lorry = low.startswith("lorry")
        is_date = low == "date"
        is_words = "amount in word" in low
        if not (is_customer or is_address or is_lorry or is_date or is_words):
            continue
        y = (box[1] + box[3]) / 2.0
        x_start = box[2] + 8.0
        if is_date:
            x_end = float(w - 40)
        elif is_words:
            x_start = box[2] + 8.0
            x_end = float(w * 0.55)
            for dy in (0.0, (box[3] - box[1]) * 1.15):
                yy = min(h - 2.0, box[3] + 6.0 + dy)
                stroke = _sample_stroke(bgr, x_start, yy, x_end, yy)
                out.append(
                    _line_item(
                        x_start,
                        yy,
                        x_end,
                        yy,
                        stroke=stroke if _is_dark(stroke) else "#333333",
                        stroke_width=1.0,
                        dashed=True,
                        role="dotted_field",
                    )
                )
            continue
        else:
            x_end = float(w - 45)
            if is_lorry:
                for t2, b2 in ocr_items:
                    if t2.strip().lower() == "date" and abs(((b2[1] + b2[3]) / 2) - y) < 30:
                        x_end = float(b2[0] - 10)
                        break
        if x_end - x_start < 40:
            continue
        yy = y + (box[3] - box[1]) * 0.35
        stroke = _sample_stroke(bgr, x_start, yy, x_end, yy)
        out.append(
            _line_item(
                x_start,
                yy,
                x_end,
                yy,
                stroke=stroke if _is_dark(stroke) else "#333333",
                stroke_width=1.0,
                dashed=True,
                role="dotted_field",
            )
        )
    # Signature underlines (solid thin rules above the labels)
    for text, box in ocr_items:
        low = text.strip().lower()
        if "signature" not in low:
            continue
        x1, y1, x2, y2 = box
        yy = max(4.0, y1 - 10.0)
        stroke = _sample_stroke(bgr, x1, yy, x2, yy)
        out.append(
            _line_item(
                x1,
                yy,
                x2,
                yy,
                stroke=stroke if _is_dark(stroke) else "#333333",
                stroke_width=1.1,
                dashed=True,
                role="dotted_field",
            )
        )
    return out


def _is_dark(hex_color: str) -> bool:
    try:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return (0.299 * r + 0.587 * g + 0.114 * b) < 140
    except Exception:
        return True


def _detect_prop_badge(
    bgr: np.ndarray,
    ocr_items: list[tuple[str, tuple[float, float, float, float]]],
) -> list[dict[str, Any]]:
    """Black pill/ellipse behind 'Prop - …' when present."""
    prop_box = None
    for text, box in ocr_items:
        if text.strip().lower().startswith("prop"):
            prop_box = box
            break
    if not prop_box:
        return []
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = prop_box
    # Ink strip through glyph centers — badge fill is dark even if padding is lavender
    cy = int(round((y1 + y2) / 2.0))
    strip = bgr[max(0, cy - 3) : min(h, cy + 4), int(max(0, x1)) : int(min(w, x2))]
    if strip.size == 0:
        return []
    med = np.median(strip.reshape(-1, 3), axis=0)
    lum = float(0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2])
    if lum > 70:
        return []
    pad_x = max(22.0, (x2 - x1) * 0.10)
    pad_y = max(5.0, (y2 - y1) * 0.35)
    bx1 = max(0.0, x1 - pad_x)
    by1 = max(0.0, y1 - pad_y)
    bx2 = min(float(w - 1), x2 + pad_x)
    by2 = min(float(h - 1), y2 + pad_y)
    return [
        {
            "type": VectorType.ELLIPSE,
            "bbox": (bx1, by1, bx2, by2),
            "fill_color": "#111111",
            "stroke_color": None,
            "stroke_width": 0.0,
            "corner_radius": 0.0,
            "rotation": 0.0,
            "confidence": 94.0,
            "layer": 5,
            "meta": {"source": "form_grid", "form": True, "role": "prop_badge"},
        }
    ]


def _find_box(
    ocr_items: list[tuple[str, tuple[float, float, float, float]]],
    *preds: str,
) -> Optional[tuple[float, float, float, float]]:
    # Prefer exact token match first
    for text, box in ocr_items:
        low = text.strip().lower().rstrip(".")
        for p in preds:
            if low == p.rstrip("."):
                return box
    for text, box in ocr_items:
        low = text.strip().lower()
        for p in preds:
            if low == p or low.startswith(p):
                return box
    return None


def _lattice_column_xs(
    ocr_items: list[tuple[str, tuple[float, float, float, float]]],
    left: float,
    right: float,
) -> list[float]:
    """Strict column edges from header label gaps — always includes outer frame."""
    keys = [
        ("s.no.", "s.no", "sl.no", "sl.no.", "sl no", "sno", "sl"),
        ("particulars",),
        ("quantity", "qty"),
        ("rate",),
        ("amount",),
    ]
    centers: list[tuple[float, float, float]] = []  # cx, x1, x2
    for alts in keys:
        box = None
        for text, b in ocr_items:
            low = re.sub(r"\s+", "", text.strip().lower())
            for a in alts:
                a2 = a.replace(" ", "")
                if low == a2 or low.startswith(a2):
                    if a2.startswith("amount") and "word" in low:
                        continue
                    if a2 in {"sl", "sno"} and not low.startswith(("sl", "sno", "s.no")):
                        continue
                    box = b
                    break
            if box:
                break
        if box:
            centers.append(((box[0] + box[2]) / 2.0, box[0], box[2]))
    if len(centers) < 3:
        # Try morph-free fallback from particulars+rate+amount only
        for alts in (("particulars",), ("rate",), ("amount",)):
            for text, b in ocr_items:
                low = re.sub(r"\s+", "", text.strip().lower())
                if low.startswith(alts[0]) and "word" not in low:
                    centers.append(((b[0] + b[2]) / 2.0, b[0], b[2]))
                    break
        centers = sorted({(round(c[0], 1), c[1], c[2]) for c in centers}, key=lambda t: t[0])
        centers = [(c[0], c[1], c[2]) for c in centers]
    if len(centers) < 2:
        n = 4
        step = (right - left) / n
        return [left + i * step for i in range(n + 1)]

    centers.sort(key=lambda t: t[0])
    xs = [left]
    for i in range(len(centers) - 1):
        xs.append((centers[i][2] + centers[i + 1][1]) / 2.0)
    xs.append(right)

    # Rs / P split inside Amount
    rs = _find_box(ocr_items, "rs.", "rs")
    p = None
    for text, box in ocr_items:
        if text.strip().lower().rstrip(".") == "p":
            p = box
            break
    if rs and p:
        split = (rs[2] + p[0]) / 2.0
        if xs[-2] < split < xs[-1]:
            xs.insert(-1, split)
        elif left < split < right:
            xs.append(split)
            xs = sorted(_cluster_1d(xs, tol=6.0))

    return sorted(_cluster_1d(xs, tol=6.0))


def _detect_black_banners(
    bgr: np.ndarray,
    ocr_items: list[tuple[str, tuple[float, float, float, float]]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Solid black bars with light text (service banners). Returns vectors + matched texts."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out: list[dict[str, Any]] = []
    matched: set[str] = set()
    for text, box in ocr_items:
        x1, y1, x2, y2 = box
        if (x2 - x1) < w * 0.35:
            continue
        low = text.strip().lower()
        # Prefer known banner phrases; skip shop titles alone
        looks_banner = any(
            k in low for k in ("service station", "repair", "reparing", "authorised dealer")
        )
        pad_x = max(8.0, (x2 - x1) * 0.03)
        pad_y = max(4.0, (y2 - y1) * 0.45)
        bx1 = int(max(0, x1 - pad_x))
        by1 = int(max(0, y1 - pad_y))
        bx2 = int(min(w - 1, x2 + pad_x))
        by2 = int(min(h - 1, y2 + pad_y))
        roi = gray[by1:by2, bx1:bx2]
        if roi.size == 0:
            continue
        dark_ratio = float((roi < 70).mean())
        # Filled black banner: most of the band is dark (not just glyph strokes)
        if dark_ratio < 0.42 and not looks_banner:
            continue
        if dark_ratio < 0.35:
            continue
        if not looks_banner and dark_ratio < 0.55:
            continue
        out.append(
            {
                "type": VectorType.RECTANGLE,
                "bbox": (float(bx1), float(by1), float(bx2), float(by2)),
                "fill_color": "#000000",
                "stroke_color": None,
                "stroke_width": 0.0,
                "corner_radius": 0.0,
                "rotation": 0.0,
                "confidence": 93.0,
                "layer": 2,
                "meta": {"source": "form_grid", "form": True, "role": "banner"},
            }
        )
        matched.add(text.strip())
    return out, matched


def detect_form_grid(
    bgr: np.ndarray,
    ocr: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a clean orthogonal bill/invoice lattice (aligned H/V rules only).
    No page background fill — transparent/white page; ink + shapes only.
    """
    if bgr is None or not is_form_like_page(ocr):
        return []

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ocr_items = _ocr_boxes(ocr)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )
    text_mask = np.zeros_like(bw)
    for _, box in ocr_items:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(
            text_mask,
            (max(0, x1 - 1), max(0, y1 - 1)),
            (min(w - 1, x2 + 1), min(h - 1, y2 + 1)),
            255,
            -1,
        )
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    h_raw = _extract_morph_lines(ink, horizontal=True, min_len=max(40, w // 14))
    v_raw = _extract_morph_lines(ink, horizontal=False, min_len=max(40, h // 16))
    h_segs = _extend_segment(h_raw, horizontal=True, snap_tol=4.0)
    v_segs = _extend_segment(v_raw, horizontal=False, snap_tol=4.0)
    v_segs = [s for s in v_segs if 10 < s[0] < w - 10 and (s[3] - s[1]) > h * 0.08]
    h_segs = [s for s in h_segs if 4 < s[1] < h - 4 and (s[2] - s[0]) > w * 0.12]

    # --- Outer frame (morph + OCR extent) ---
    left_cands = [s[0] for s in v_segs if 12 < s[0] < w * 0.16 and (s[3] - s[1]) > h * 0.3]
    right_cands = [s[0] for s in v_segs if w * 0.84 < s[0] < w - 8 and (s[3] - s[1]) > h * 0.3]
    top_cands = [s[1] for s in h_segs if 4 < s[1] < h * 0.12 and (s[2] - s[0]) > w * 0.45]
    bot_cands = [s[1] for s in h_segs if h * 0.88 < s[1] < h - 4 and (s[2] - s[0]) > w * 0.45]
    left = float(min(left_cands)) if left_cands else float(min(b[0] for _, b in ocr_items) - 6)
    right = float(max(right_cands)) if right_cands else float(max(b[2] for _, b in ocr_items) + 6)
    top = float(min(top_cands)) if top_cands else float(min(b[1] for _, b in ocr_items) - 10)
    bottom = float(max(bot_cands)) if bot_cands else float(max(b[3] for _, b in ocr_items) + 10)
    left = float(np.clip(left, 8, w * 0.2))
    right = float(np.clip(right, w * 0.8, w - 8))
    top = float(np.clip(top, 4, h * 0.15))
    bottom = float(np.clip(bottom, h * 0.85, h - 4))
    for text, box in ocr_items:
        if "signature" in text.lower():
            bottom = max(bottom, min(float(h - 4), box[3] + 10.0))

    # --- Table band ---
    header_boxes = []
    for text, box in ocr_items:
        low = text.strip().lower()
        if (
            low in {"sl.no", "s.no", "s.no.", "particulars", "quantity", "rate", "amount", "rs.", "rs", "p", "p."}
            or low.startswith("sl")
            or low.startswith("s.no")
        ):
            if "word" in low or "total amount" in low:
                continue
            header_boxes.append(box)
    if not header_boxes:
        return []

    # Optional vehicle/type/meter header row above particulars
    veh = None
    typ = None
    meter = None
    for text, box in ocr_items:
        low = text.strip().lower()
        if low.startswith("vehicle"):
            veh = box
        elif low == "type":
            typ = box
        elif "meter" in low:
            meter = box

    table_top = min(b[1] for b in header_boxes) - 6.0
    if veh:
        table_top = min(table_top, veh[1] - 6.0)
    header_bot = max(b[3] for b in header_boxes) + 4.0
    near_hb = [s[1] for s in h_segs if abs(s[1] - header_bot) < 18 and (s[2] - s[0]) > (right - left) * 0.5]
    if near_hb:
        header_bot = float(np.median(near_hb))
    near_ht = [s[1] for s in h_segs if abs(s[1] - table_top) < 18 and (s[2] - s[0]) > (right - left) * 0.5]
    if near_ht:
        table_top = float(np.median(near_ht))

    # Mid line between vehicle row and particulars row
    vehicle_mid_y = None
    if veh and header_boxes:
        vehicle_mid_y = (max(b[3] for b in (veh, typ, meter) if b) + min(b[1] for b in header_boxes)) / 2.0

    words = _find_box(ocr_items, "amount in word")
    total = _find_box(ocr_items, "total")  # exact TOTAL preferred
    table_bot = float(h * 0.92)
    if words:
        table_bot = words[1] - 4.0
    elif total:
        # TOTAL sits inside last row — floor just below it
        table_bot = total[3] + 8.0
    near_tb = [s[1] for s in h_segs if abs(s[1] - table_bot) < 24 and (s[2] - s[0]) > (right - left) * 0.45]
    if near_tb:
        table_bot = float(np.median(near_tb))

    # Footer notes / signatory outside the main border (common on garage bills)
    footer_outside = False
    for text, box in ocr_items:
        low = text.strip().lower()
        if any(
            k in low
            for k in ("e.&o.e", "e. & o.e", "jurisdiction", "authorised", "authorized", "owners risk")
        ):
            footer_outside = True
            break
        if low.startswith("total amount") and box[1] > table_bot - 20:
            footer_outside = True
    if footer_outside:
        bottom = float(table_bot)
    else:
        table_bot = min(table_bot, bottom - 8.0)
        for text, box in ocr_items:
            if "signature" in text.lower():
                bottom = max(bottom, min(float(h - 4), box[3] + 10.0))

    table_top = max(table_top, top + 8.0)
    bottom = max(bottom, table_bot)

    # Customer box sits between address block and table
    cust = _find_box(ocr_items, "customer")
    addr = _find_box(ocr_items, "address")
    lorry = _find_box(ocr_items, "lorry")
    date = _find_box(ocr_items, "date")
    cust_top = (cust[1] - 8.0) if cust else (table_top - 90.0)
    if lorry or date:
        cust_bot = max((lorry or date)[3], (date or lorry)[3]) + 10.0
        gap_lines = [
            s[1]
            for s in h_segs
            if cust_bot - 5 < s[1] < table_top + 5 and (s[2] - s[0]) > (right - left) * 0.5
        ]
        if gap_lines:
            table_top = float(min(gap_lines))

    col_xs = _lattice_column_xs(ocr_items, left, right)
    if abs(col_xs[0] - left) > 3:
        col_xs = [left] + col_xs
    if abs(col_xs[-1] - right) > 3:
        col_xs = col_xs + [right]
    col_xs = sorted(_cluster_1d(col_xs, tol=5.0))

    amount_box = _find_box(ocr_items, "amount")
    rs_box = _find_box(ocr_items, "rs.", "rs")
    amount_split_y = None
    if amount_box and rs_box:
        amount_split_y = (amount_box[3] + rs_box[1]) / 2.0
    elif amount_box:
        amount_split_y = (amount_box[3] + header_bot) / 2.0

    total_row_y = None
    if total and not words:
        total_row_y = total[1] - 4.0
        near = [s[1] for s in h_segs if abs(s[1] - total_row_y) < 16 and (s[2] - s[0]) > (right - left) * 0.25]
        if near:
            total_row_y = float(np.median(near))
    elif words:
        total_row_y = words[1] - 6.0

    stroke = "#000000"
    border_w = 2.0
    grid_w = 1.15
    vectors: list[dict[str, Any]] = []

    # Rounded outer border (matches printed bill books)
    radius = max(6.0, min(14.0, (right - left) * 0.02))
    vectors.append(
        {
            "type": VectorType.ROUNDED_RECTANGLE,
            "bbox": (left, top, right, bottom),
            "fill_color": None,
            "stroke_color": stroke,
            "stroke_width": border_w,
            "corner_radius": radius,
            "rotation": 0.0,
            "confidence": 97.0,
            "layer": 4,
            "meta": {"source": "form_grid", "form": True, "role": "border"},
        }
    )

    banners, banner_texts = _detect_black_banners(bgr, ocr_items)
    vectors.extend(banners)

    # Customer info box top
    if cust or addr:
        if abs(cust_top - table_top) > 8:
            vectors.append(
                _line_item(left, cust_top, right, cust_top, stroke=stroke, stroke_width=grid_w, role="grid_h")
            )

    # Horizontals — full width, shared endpoints
    y_horiz = [table_top, header_bot, table_bot]
    if vehicle_mid_y and table_top + 6 < vehicle_mid_y < header_bot - 4:
        y_horiz.append(vehicle_mid_y)
    if total_row_y and table_top + 8 < total_row_y < table_bot - 6:
        y_horiz.append(total_row_y)
    y_horiz = sorted(_cluster_1d(y_horiz, tol=4.0))
    for y in y_horiz:
        vectors.append(
            _line_item(left, y, right, y, stroke=stroke, stroke_width=grid_w, role="grid_h")
        )

    # Amount Rs/P sub-header
    if amount_split_y is not None and len(col_xs) >= 2:
        ax0 = col_xs[-2]
        if amount_box:
            acx = (amount_box[0] + amount_box[2]) / 2.0
            for i in range(len(col_xs) - 1):
                if col_xs[i] <= acx <= col_xs[i + 1]:
                    ax0 = col_xs[i]
                    break
        vectors.append(
            _line_item(ax0, amount_split_y, right, amount_split_y, stroke=stroke, stroke_width=grid_w, role="grid_h")
        )

    # Vertical column rules — same y-span (body). Outer edges skipped (rounded border).
    body_top = vehicle_mid_y if vehicle_mid_y else table_top
    for x in col_xs:
        if abs(x - left) < 2.5 or abs(x - right) < 2.5:
            continue
        # S.No divider often only through header; still run full body for clarity
        vectors.append(
            _line_item(x, table_top, x, table_bot, stroke=stroke, stroke_width=grid_w, role="grid_v")
        )

    # Vehicle-row vertical splits (Type / Meter) when present — header band only
    if veh and typ and vehicle_mid_y:
        vx = (veh[2] + typ[0]) / 2.0
        vectors.append(
            _line_item(vx, table_top, vx, vehicle_mid_y, stroke=stroke, stroke_width=grid_w, role="grid_v")
        )
        if meter:
            mx = (typ[2] + meter[0]) / 2.0
            vectors.append(
                _line_item(mx, table_top, mx, vehicle_mid_y, stroke=stroke, stroke_width=grid_w, role="grid_v")
            )

    # Address underline under location line
    loc = None
    for text, box in ocr_items:
        if "medinipur" in text.lower() or "shyamchak" in text.lower() or "delhi" in text.lower():
            if box[1] < table_top:
                loc = box
                break
    if loc:
        yy = loc[3] + 2.0
        vectors.append(
            _line_item(loc[0], yy, loc[2], yy, stroke=stroke, stroke_width=1.1, role="grid_h")
        )

    vectors.extend(_detect_dotted_fill_lines(gray, bgr, ocr_items))
    # Extra form fields: Bill No / To / Vehicle / Type / Meter / TOTAL AMOUNT
    for text, box in ocr_items:
        low = text.strip().lower()
        x1, y1, x2, y2 = box
        yy = (y1 + y2) / 2.0 + (y2 - y1) * 0.2
        if low.startswith("bill no") or low.startswith("bill no."):
            vectors.append(
                _line_item(x2 + 4, yy, min(right - 8, x2 + (right - left) * 0.35), yy, stroke=stroke, stroke_width=1.0, dashed=True, role="dotted_field")
            )
        elif low.startswith("to") and len(low) <= 5:
            vectors.append(
                _line_item(x2 + 4, yy, right - 8, yy, stroke=stroke, stroke_width=1.0, dashed=True, role="dotted_field")
            )
        elif low.startswith("date"):
            vectors.append(
                _line_item(x2 + 4, yy, right - 8, yy, stroke=stroke, stroke_width=1.0, dashed=True, role="dotted_field")
            )
        elif low.startswith("total amount"):
            vectors.append(
                _line_item(x2 + 6, yy, right - 8, yy, stroke=stroke, stroke_width=1.0, dashed=True, role="dotted_field")
            )
        elif low.startswith("vehicle") or low == "type" or "meter" in low:
            # short fill within cell — stop before next label
            x_end = right - 8
            for t2, b2 in ocr_items:
                if b2 is box:
                    continue
                if abs(((b2[1] + b2[3]) / 2) - yy) < 14 and b2[0] > x2 + 10:
                    x_end = min(x_end, b2[0] - 6)
            if x_end > x2 + 20:
                vectors.append(
                    _line_item(x2 + 4, yy, x_end, yy, stroke=stroke, stroke_width=1.0, dashed=True, role="dotted_field")
                )

    # Signatory rule
    for text, box in ocr_items:
        low = text.strip().lower()
        if "signatory" in low or "signature" in low:
            x1, y1, x2, y2 = box
            yy = y1 - 8.0
            vectors.append(
                _line_item(x1, yy, x2, yy, stroke=stroke, stroke_width=1.1, dashed=False, role="grid_h")
            )

    for item in vectors:
        meta = item.get("meta") or {}
        if meta.get("role") != "dotted_field":
            continue
        pts = item.get("points") or []
        if len(pts) < 2:
            continue
        x1, y1 = float(pts[0].x), float(pts[0].y)
        x2, y2 = float(pts[1].x), float(pts[1].y)
        if abs(y2 - y1) < 2 and x2 > x1:
            x2 = min(x2, right - 4)
            item["points"] = [ControlPoint(x=x1, y=y1), ControlPoint(x=x2, y=y1)]
            item["bbox"] = (x1, y1 - 0.5, x2, y1 + 0.5)

    vectors.extend(_detect_prop_badge(bgr, ocr_items))
    # Stash banner texts on a sentinel meta for scene refine via side channel? 
    # Scene refine detects banners by dark sampling instead.
    return vectors


def filter_non_form_noise(raw: list[dict[str, Any]], form_active: bool) -> list[dict[str, Any]]:
    """When a form lattice is present, keep only form ink + safe accents."""
    if not form_active:
        return raw
    kept: list[dict[str, Any]] = []
    for item in raw:
        meta = item.get("meta") or {}
        if meta.get("form") or meta.get("source") == "form_grid":
            # Never keep page washes
            if meta.get("role") == "page_wash":
                continue
            kept.append(item)
            continue
        # Drop everything else on form pages (lattice is authoritative)
        continue
    return kept
