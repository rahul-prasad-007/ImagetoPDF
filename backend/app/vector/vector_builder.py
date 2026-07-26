"""
Vector reconstruction orchestrator.

Inputs: processed image + scene graph + layout (+ typography optional)
Output: results/vector_<uuid>.json + debug/vector_<uuid>.png

NO PDF / SVG file export — vector data only.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import Settings
from app.ocr.ocr_service import resolve_processed_image
from app.classify.document_type import ensure_document_mode, load_document_mode
from app.vector.color_regions import detect_color_regions
from app.vector.gradient_detector import attach_gradients
from app.vector.models import (
    ControlPoint,
    GradientKind,
    GradientSpec,
    PathData,
    VectorCounts,
    VectorObject,
    VectorSummary,
    VectorType,
)
from app.vector.shape_detector import detect_shapes
from app.vector.ornaments import detect_heart_ornaments, filter_parchment_noise
from app.vector.form_grid import detect_form_grid, filter_non_form_noise, is_form_like_page
from app.scene.refine import refine_vector_separators

logger = logging.getLogger(__name__)

_SCENE_VECTOR_TYPES = {
    "RECTANGLE",
    "ROUNDED_RECTANGLE",
    "LINE",
    "ELLIPSE",
    "CIRCLE",
    "POLYGON",
    "PATH",
    "BACKGROUND",
}

_TYPE_MAP_SCENE = {
    "RECTANGLE": VectorType.RECTANGLE,
    "ROUNDED_RECTANGLE": VectorType.ROUNDED_RECTANGLE,
    "LINE": VectorType.LINE,
    "ELLIPSE": VectorType.ELLIPSE,
    "CIRCLE": VectorType.CIRCLE,
    "POLYGON": VectorType.POLYGON,
    "PATH": VectorType.PATH,
    "BACKGROUND": VectorType.COLOR_REGION,
}


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed reading %s", path)
        return None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _near(a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float) -> bool:
    ae = (a[0] - gap, a[1] - gap, a[2] + gap, a[3] + gap)
    return not (ae[2] < b[0] or b[2] < ae[0] or ae[3] < b[1] or b[3] < ae[1])


def _expand(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _same_color(c1: Optional[str], c2: Optional[str], tol: int = 28) -> bool:
    if not c1 or not c2 or len(c1) < 7 or len(c2) < 7:
        return c1 == c2
    try:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2) <= tol
    except Exception:
        return c1 == c2


def merge_shapes(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge adjacent same-color panels/rects, duplicate boxes, continuous lines."""
    mergeable = {
        VectorType.RECTANGLE,
        VectorType.ROUNDED_RECTANGLE,
        VectorType.PANEL,
        VectorType.COLOR_REGION,
        VectorType.BORDER,
    }
    others = [r for r in raw if r.get("type") not in mergeable]
    candidates = [dict(r) for r in raw if r.get("type") in mergeable]
    merged_count = 0

    # Deduplicate high IoU
    candidates.sort(
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for cur in candidates:
        dup = False
        for k in kept:
            if _iou(cur["bbox"], k["bbox"]) >= 0.88 and _same_color(
                cur.get("fill_color"), k.get("fill_color")
            ):
                k.setdefault("merged_from_tmp", []).append(cur.get("_tmp_id"))
                merged_count += 1
                dup = True
                break
        if not dup:
            kept.append(cur)
    candidates = kept

    # Adjacent panel merge
    changed = True
    while changed:
        changed = False
        n = len(candidates)
        used = [False] * n
        nxt: list[dict[str, Any]] = []
        for i in range(n):
            if used[i]:
                continue
            cur = candidates[i]
            cb = cur["bbox"]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                other = candidates[j]
                if not _same_color(cur.get("fill_color"), other.get("fill_color")):
                    continue
                if not _near(cb, other["bbox"], gap=14.0):
                    continue
                # Prefer merge when IoU soft-overlap or touching
                used[j] = True
                cb = _expand(cb, other["bbox"])
                cur = dict(cur)
                cur["bbox"] = cb
                cur["confidence"] = max(float(cur.get("confidence") or 0), float(other.get("confidence") or 0))
                if cur.get("type") == VectorType.COLOR_REGION and other.get("type") == VectorType.PANEL:
                    cur["type"] = VectorType.PANEL
                merged_count += 1
                changed = True
            used[i] = True
            nxt.append(cur)
        candidates = nxt

    # Continuous collinear lines
    lines = [dict(r) for r in others if r.get("type") == VectorType.LINE]
    rest = [r for r in others if r.get("type") != VectorType.LINE]
    changed = True
    while changed and len(lines) > 1:
        changed = False
        n = len(lines)
        used = [False] * n
        nxt = []
        for i in range(n):
            if used[i]:
                continue
            cur = lines[i]
            cb = cur["bbox"]
            rot = float(cur.get("rotation") or 0)
            horiz = abs(rot) < 30 or abs(abs(rot) - 180) < 30
            for j in range(i + 1, n):
                if used[j]:
                    continue
                other = lines[j]
                if not _same_color(cur.get("stroke_color"), other.get("stroke_color"), tol=40):
                    continue
                ob = other["bbox"]
                if horiz:
                    if abs(((cb[1] + cb[3]) / 2) - ((ob[1] + ob[3]) / 2)) > 10:
                        continue
                    if ob[0] > cb[2] + 12 or cb[0] > ob[2] + 12:
                        continue
                else:
                    if abs(((cb[0] + cb[2]) / 2) - ((ob[0] + ob[2]) / 2)) > 10:
                        continue
                    if ob[1] > cb[3] + 12 or cb[1] > ob[3] + 12:
                        continue
                used[j] = True
                cb = _expand(cb, ob)
                cur = dict(cur)
                cur["bbox"] = cb
                pts = list(cur.get("points") or []) + list(other.get("points") or [])
                cur["points"] = pts
                merged_count += 1
                changed = True
            used[i] = True
            nxt.append(cur)
        lines = nxt

    # Nested rectangles: drop inner if same color and largely contained
    final_rects: list[dict[str, Any]] = []
    ordered = sorted(
        candidates,
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    for cur in ordered:
        drop = False
        for outer in final_rects:
            if not _same_color(cur.get("fill_color"), outer.get("fill_color")):
                continue
            ob, ib = outer["bbox"], cur["bbox"]
            if ib[0] >= ob[0] - 2 and ib[1] >= ob[1] - 2 and ib[2] <= ob[2] + 2 and ib[3] <= ob[3] + 2:
                if _iou(ob, ib) > 0.35:
                    drop = True
                    merged_count += 1
                    break
        if not drop:
            final_rects.append(cur)

    return final_rects + rest + lines, merged_count


def _seed_from_scene(scene: Optional[dict[str, Any]], src_w: float, src_h: float) -> list[dict[str, Any]]:
    if not scene:
        return []
    page = scene.get("page") or {}
    scale_x = float(page.get("scale_x") or 1.0) or 1.0
    scale_y = float(page.get("scale_y") or 1.0) or 1.0
    ox = float(page.get("offset_x") or 0.0)
    oy = float(page.get("offset_y") or 0.0)

    out: list[dict[str, Any]] = []
    for obj in scene.get("objects") or []:
        otype = str(obj.get("type") or "")
        if otype not in _SCENE_VECTOR_TYPES:
            continue
        # Default slate placeholders are almost never real ink — prefer Hough/ornament lines
        stroke = (obj.get("stroke_color") or obj.get("stroke") or "").upper()
        fill = (obj.get("fill_color") or obj.get("fill") or "").upper()
        if otype == "LINE" and stroke in {"#64748B", "#94A3B8"} and fill in {"", "#E2E8F0", "#F1F5F9", "#CBD5E1"}:
            continue
        # Map page coords → source image coords
        x = (float(obj.get("x") or 0) - ox) / scale_x
        y = (float(obj.get("y") or 0) - oy) / scale_y
        w = float(obj.get("width") or 0) / scale_x
        h = float(obj.get("height") or 0) / scale_y
        if w < 1 or h < 1:
            continue
        bbox = (
            float(np.clip(x, 0, src_w)),
            float(np.clip(y, 0, src_h)),
            float(np.clip(x + w, 0, src_w)),
            float(np.clip(y + h, 0, src_h)),
        )
        vtype = _TYPE_MAP_SCENE.get(otype, VectorType.RECTANGLE)
        fill = obj.get("fill_color") or obj.get("fill")
        stroke = obj.get("stroke_color") or obj.get("stroke")
        out.append(
            {
                "type": vtype,
                "bbox": bbox,
                "fill_color": fill,
                "stroke_color": stroke,
                "stroke_width": float(obj.get("stroke_width") or 0),
                "corner_radius": float(obj.get("corner_radius") or 0) / scale_x,
                "rotation": float(obj.get("rotation") or 0),
                "opacity": float(obj.get("opacity") or 1),
                "confidence": 96.0,
                "layer": int(obj.get("layer") or 3),
                "meta": {"source": "scene", "scene_id": obj.get("id"), "scene_type": otype},
            }
        )
    return out


def _seed_from_layout(layout: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not layout:
        return []
    shape_types = {
        "RECTANGLE": VectorType.RECTANGLE,
        "ROUNDED_RECTANGLE": VectorType.ROUNDED_RECTANGLE,
        "LINE": VectorType.LINE,
        "CIRCLE": VectorType.CIRCLE,
        "ELLIPSE": VectorType.ELLIPSE,
        "BACKGROUND_SHAPE": VectorType.PANEL,
    }
    out: list[dict[str, Any]] = []
    for obj in layout.get("objects") or []:
        otype = str(obj.get("type") or "")
        if otype not in shape_types:
            continue
        bbox = obj.get("bbox") or []
        if len(bbox) < 4:
            continue
        meta = obj.get("meta") or {}
        fill = None
        if meta.get("color_bgr"):
            b, g, r = meta["color_bgr"][:3]
            fill = f"#{int(r):02X}{int(g):02X}{int(b):02X}"
        conf = float(obj.get("confidence") or 0.9)
        if conf <= 1.0:
            conf *= 100.0
        out.append(
            {
                "type": shape_types[otype],
                "bbox": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                "fill_color": fill,
                "stroke_color": "#64748B" if otype == "LINE" else None,
                "stroke_width": 2.0 if otype == "LINE" else 0.0,
                "corner_radius": 8.0 if otype == "ROUNDED_RECTANGLE" else 0.0,
                "rotation": float(obj.get("rotation") or 0),
                "confidence": conf,
                "layer": 2 if otype == "BACKGROUND_SHAPE" or meta.get("highlight") else 3,
                "meta": {
                    "source": "layout",
                    "layout_id": obj.get("id"),
                    "highlight": bool(meta.get("highlight")),
                },
            }
        )
    return out


def _to_vector_object(idx: int, raw: dict[str, Any]) -> VectorObject:
    bbox = raw["bbox"]
    x1, y1, x2, y2 = bbox
    vtype = raw["type"]
    if isinstance(vtype, str):
        vtype = VectorType(vtype)

    gradient = raw.get("gradient")
    if isinstance(gradient, dict):
        gradient = GradientSpec(**gradient)

    path = raw.get("path")
    if isinstance(path, dict):
        path = PathData(**path)

    points = raw.get("points") or []
    norm_pts: list[ControlPoint] = []
    for p in points:
        if isinstance(p, ControlPoint):
            norm_pts.append(p)
        elif isinstance(p, dict):
            norm_pts.append(ControlPoint(**p))

    fill = raw.get("fill_color")
    stroke = raw.get("stroke_color")

    # Promote color region with gradient
    if gradient and getattr(gradient, "kind", GradientKind.NONE) != GradientKind.NONE:
        if vtype in {VectorType.COLOR_REGION, VectorType.PANEL}:
            vtype = VectorType.GRADIENT_REGION

    return VectorObject(
        id=idx,
        type=vtype,
        x=round(x1, 2),
        y=round(y1, 2),
        width=round(max(0.0, x2 - x1), 2),
        height=round(max(0.0, y2 - y1), 2),
        rotation=round(float(raw.get("rotation") or 0), 2),
        fill_color=fill,
        stroke_color=stroke,
        fill=fill,
        stroke=stroke,
        stroke_width=round(float(raw.get("stroke_width") or 0), 2),
        corner_radius=round(float(raw.get("corner_radius") or 0), 2),
        opacity=float(raw.get("opacity") or 1.0),
        layer=int(raw["layer"]) if raw.get("layer") is not None else 3,
        path=path,
        gradient=gradient,
        points=norm_pts,
        confidence=round(float(raw.get("confidence") or 90), 2),
        merged_from=[],
        source=dict(raw.get("meta") or {}),
        meta=dict(raw.get("meta") or {}),
    )


def _build_counts(vectors: list[VectorObject], merged: int) -> VectorCounts:
    c = VectorCounts(total=len(vectors), merged_shapes=merged)
    for v in vectors:
        t = v.type
        if t == VectorType.RECTANGLE:
            c.rectangles += 1
        elif t == VectorType.ROUNDED_RECTANGLE:
            c.rounded_rectangles += 1
        elif t == VectorType.LINE or t == VectorType.BORDER:
            c.lines += 1
        elif t == VectorType.PATH:
            c.paths += 1
            c.curve_count += 1
        elif t in {VectorType.WAVE, VectorType.RIBBON, VectorType.CURVED_BAND}:
            c.paths += 1
            c.curve_count += 1
            if t == VectorType.WAVE:
                c.waves += 1
            elif t == VectorType.RIBBON:
                c.ribbons += 1
        elif t == VectorType.GRADIENT_REGION:
            c.gradients += 1
            c.color_regions += 1
        elif t in {VectorType.COLOR_REGION, VectorType.PANEL}:
            c.color_regions += 1
            if v.gradient and v.gradient.kind != GradientKind.NONE:
                c.gradients += 1
        elif t == VectorType.CIRCLE:
            c.circles += 1
        elif t == VectorType.ELLIPSE:
            c.ellipses += 1
        elif t == VectorType.POLYGON:
            c.polygons += 1
        elif t == VectorType.TRIANGLE:
            c.triangles += 1
        elif t == VectorType.ARROW:
            c.arrows += 1
            c.paths += 1
        elif t == VectorType.HEART:
            c.paths += 1
            c.curve_count += 1
        if v.path and v.path.commands and t not in {
            VectorType.PATH,
            VectorType.WAVE,
            VectorType.RIBBON,
            VectorType.CURVED_BAND,
            VectorType.ARROW,
            VectorType.HEART,
        }:
            c.curve_count += 1
    return c


def _score(counts: VectorCounts, avg_conf: float) -> float:
    # Recovery score: prefer presence of geometry + confidence
    geometric = (
        counts.rectangles
        + counts.rounded_rectangles
        + counts.lines
        + counts.circles
        + counts.ellipses
        + counts.polygons
        + counts.triangles
        + counts.paths
        + counts.color_regions
    )
    base = min(100.0, 40.0 + geometric * 3.5 + counts.gradients * 2.0)
    score = 0.55 * base + 0.45 * avg_conf
    return float(np.clip(score, 0, 100))


def _ocr_text_boxes(ocr: Optional[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not ocr:
        return boxes
    for b in ocr.get("text_blocks") or []:
        pts = b.get("bbox") or []
        if len(pts) >= 4 and isinstance(pts[0], (list, tuple)):
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        elif len(pts) >= 4 and not isinstance(pts[0], (list, tuple)):
            boxes.append((float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3])))
    return boxes


def _box_overlap_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection over area of a."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _mask_text_regions(bgr: np.ndarray, text_boxes: list[tuple[float, float, float, float]], pad: int = 3) -> np.ndarray:
    """Paint OCR text regions with local background so glyph strokes aren't vectorized."""
    out = bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in text_boxes:
        xi1 = max(0, int(x1) - pad)
        yi1 = max(0, int(y1) - pad)
        xi2 = min(w, int(x2) + pad)
        yi2 = min(h, int(y2) + pad)
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        # Sample border pixels around the box for fill color
        band = []
        if yi1 > 0:
            band.append(out[max(0, yi1 - 2) : yi1, xi1:xi2])
        if yi2 < h:
            band.append(out[yi2 : min(h, yi2 + 2), xi1:xi2])
        if xi1 > 0:
            band.append(out[yi1:yi2, max(0, xi1 - 2) : xi1])
        if xi2 < w:
            band.append(out[yi1:yi2, xi2 : min(w, xi2 + 2)])
        if band:
            sample = np.concatenate([b.reshape(-1, 3) for b in band if b.size], axis=0)
            fill = np.median(sample, axis=0).astype(np.uint8) if len(sample) else np.array([255, 255, 255], dtype=np.uint8)
        else:
            fill = np.array([255, 255, 255], dtype=np.uint8)
        out[yi1:yi2, xi1:xi2] = fill
    return out


def _filter_vectors_over_text(
    raw: list[dict[str, Any]],
    text_boxes: list[tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    """Drop lines/arrows/small shapes that mostly sit on OCR text (serif false positives)."""
    if not text_boxes:
        return raw
    kept: list[dict[str, Any]] = []
    risky = {
        VectorType.ARROW,
        VectorType.LINE,
        VectorType.BORDER,
        VectorType.POLYGON,
        VectorType.PATH,
        VectorType.RIBBON,
        VectorType.WAVE,
    }
    for item in raw:
        vtype = item.get("type")
        if isinstance(vtype, str):
            try:
                vtype = VectorType(vtype)
            except Exception:
                vtype = None
        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            kept.append(item)
            continue
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        max_overlap = 0.0
        for tb in text_boxes:
            max_overlap = max(max_overlap, _box_overlap_ratio(box, tb))

        # Always keep large panels / color regions
        if vtype in {VectorType.COLOR_REGION, VectorType.PANEL, VectorType.GRADIENT_REGION}:
            kept.append(item)
            continue
        meta = item.get("meta") or {}
        if vtype == VectorType.HEART or meta.get("ornament") in {"heart", "signature_heart", "line", "dotted"}:
            # Ignore the OCR icon box that *is* this heart (often "3" / ♡)
            heart_overlap = 0.0
            for tb in text_boxes:
                # Token mostly covered by the heart → it's the icon itself
                if _box_overlap_ratio(tb, box) >= 0.65:
                    continue
                heart_overlap = max(heart_overlap, _box_overlap_ratio(box, tb))
            if vtype == VectorType.HEART and meta.get("ornament") in {"heart", "signature_heart"}:
                if meta.get("ornament") == "signature_heart":
                    kept.append(item)
                    continue
                if heart_overlap >= 0.35:
                    continue
                kept.append(item)
                continue
            if meta.get("ornament") == "line" and heart_overlap >= 0.45:
                continue
            if meta.get("ornament") in {"line", "dotted"}:
                kept.append(item)
                continue
            kept.append(item)
            continue
        # Form grid lines must survive even when they cross label boxes
        if meta.get("form") or meta.get("source") == "form_grid":
            kept.append(item)
            continue
        if meta.get("source") == "scene" and str(meta.get("scene_type") or "") == "BACKGROUND":
            kept.append(item)
            continue

        fill = item.get("fill_color") or item.get("fill")
        is_highlight = False
        if isinstance(fill, str) and fill.startswith("#") and len(fill) >= 7:
            try:
                r, g, b = int(fill[1:3], 16), int(fill[3:5], 16), int(fill[5:7], 16)
                mx, mn = max(r, g, b), min(r, g, b)
                sat = (mx - mn) / max(mx, 1)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                if sat > 0.25 and 70 < lum < 245 and g >= r + 20 and g >= b + 10:
                    is_highlight = True
            except ValueError:
                pass
        if (item.get("meta") or {}).get("highlight"):
            is_highlight = True
        if is_highlight:
            kept.append(item)
            continue

        # Drop generic false arrows on illustrated / parchment pages
        if vtype == VectorType.ARROW and (item.get("meta") or {}).get("source") == "shape_detector":
            if max_overlap >= 0.05 or bw * bh < 8000:
                continue

        # Diagonal hough strokes are usually illustration edges, not rules
        if vtype == VectorType.LINE and (item.get("meta") or {}).get("source") == "hough_line":
            rot = abs(float(item.get("rotation") or 0.0)) % 180.0
            near_axis = min(rot, abs(rot - 90), abs(rot - 180)) <= 12.0
            if not near_axis:
                continue
            # Cap noisy edge fragments: keep only near page borders or long rules
            pts = item.get("points") or []
            if len(pts) >= 2:
                p0, p1 = pts[0], pts[1]
                x0 = float(getattr(p0, "x", None) if not isinstance(p0, dict) else p0.get("x") or 0)
                y0 = float(getattr(p0, "y", None) if not isinstance(p0, dict) else p0.get("y") or 0)
                x1 = float(getattr(p1, "x", None) if not isinstance(p1, dict) else p1.get("x") or 0)
                y1 = float(getattr(p1, "y", None) if not isinstance(p1, dict) else p1.get("y") or 0)
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                # Estimate page from text boxes
                page_w = max((tb[2] for tb in text_boxes), default=mx) * 1.15
                page_h = max((tb[3] for tb in text_boxes), default=my) * 1.15
                near_edge = (
                    mx < page_w * 0.08
                    or mx > page_w * 0.92
                    or my < page_h * 0.06
                    or my > page_h * 0.94
                )
                if length < page_w * 0.35 and not near_edge and max_overlap < 0.01:
                    # Interior short axis-aligned fragments often from texture
                    if length < 120:
                        continue

        if vtype in risky and max_overlap >= 0.35:
            continue
        # Horizontal rules through body copy
        if vtype == VectorType.LINE and max_overlap >= 0.18 and bw >= bh * 3:
            if (item.get("meta") or {}).get("ornament") != "line":
                continue
        # Tiny rectangles overlapping text glyphs
        if vtype in {VectorType.RECTANGLE, VectorType.ROUNDED_RECTANGLE} and max_overlap >= 0.45 and bw * bh < 20000:
            continue
        # Thin vertical/horizontal strokes overlapping text
        if max_overlap >= 0.5 and min(bw, bh) <= 8 and max(bw, bh) < 220:
            # Keep tall thin accent bars (green separators) even if near text
            if bh >= bw * 4 and bw <= 12:
                kept.append(item)
                continue
            continue
        # Large gray panels over body text
        if (
            vtype in {VectorType.RECTANGLE, VectorType.ROUNDED_RECTANGLE, VectorType.PANEL}
            and max_overlap >= 0.25
            and bw * bh > 5000
            and not is_highlight
        ):
            continue
        kept.append(item)
    return kept


def _dedupe_horizontal_rules(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a single thin dark horizontal rule per y-band; drop near-duplicates."""
    lines = []
    rest = []
    for item in raw:
        vtype = item.get("type")
        if isinstance(vtype, str):
            try:
                vtype = VectorType(vtype)
            except Exception:
                vtype = None
        bbox = item.get("bbox")
        if vtype != VectorType.LINE or not bbox or len(bbox) < 4:
            rest.append(item)
            continue
        x1, y1, x2, y2 = map(float, bbox[:4])
        bw, bh = abs(x2 - x1), abs(y2 - y1)
        is_ornament = (item.get("meta") or {}).get("ornament") == "line"
        if bh <= 12 and (bw >= 40 or is_ornament):
            lines.append(item)
        else:
            rest.append(item)

    if len(lines) <= 1:
        return rest + lines

    lines.sort(key=lambda it: ((float(it["bbox"][1]) + float(it["bbox"][3])) / 2.0))
    kept_lines: list[dict[str, Any]] = []
    for cur in lines:
        cy = (float(cur["bbox"][1]) + float(cur["bbox"][3])) / 2.0
        cx1, cx2 = float(cur["bbox"][0]), float(cur["bbox"][2])
        dup = False
        for k in kept_lines:
            ky = (float(k["bbox"][1]) + float(k["bbox"][3])) / 2.0
            kx1, kx2 = float(k["bbox"][0]), float(k["bbox"][2])
            x_overlap = min(cx2, kx2) - max(cx1, kx1)
            # Same y-band but separate x spans (left/right ornaments) — keep both
            if abs(cy - ky) <= 8.0 and x_overlap > -4.0:
                cur_stroke = (cur.get("stroke_color") or cur.get("fill_color") or "#888888").upper()
                k_stroke = (k.get("stroke_color") or k.get("fill_color") or "#888888").upper()
                cur_bh = abs(float(cur["bbox"][3]) - float(cur["bbox"][1]))
                k_bh = abs(float(k["bbox"][3]) - float(k["bbox"][1]))
                prefer_cur = cur_bh < k_bh or (cur_bh == k_bh and cur_stroke < k_stroke)
                cur_orn = (cur.get("meta") or {}).get("ornament") == "line"
                k_orn = (k.get("meta") or {}).get("ornament") == "line"
                if k_orn and not cur_orn:
                    prefer_cur = False
                if cur_orn and not k_orn:
                    prefer_cur = True
                if prefer_cur:
                    meta = dict(k.get("meta") or {})
                    k.clear()
                    k.update(cur)
                    k["meta"] = {**meta, **dict(cur.get("meta") or {})}
                    k["stroke_color"] = cur.get("stroke_color") or "#2A2A2A"
                    k["fill_color"] = None
                    k["stroke_width"] = max(1.0, min(2.5, cur_bh or 1.5))
                dup = True
                break
        if not dup:
            cur = dict(cur)
            if (cur.get("meta") or {}).get("ornament") != "line":
                cur["stroke_color"] = cur.get("stroke_color") or "#2A2A2A"
            cur["fill_color"] = None
            bh = abs(float(cur["bbox"][3]) - float(cur["bbox"][1]))
            cur["stroke_width"] = max(1.0, min(2.5, float(cur.get("stroke_width") or bh or 1.5)))
            kept_lines.append(cur)
    return rest + kept_lines


def _dedupe_vertical_rules(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a single thin vertical rule per x-band (bill column lines)."""
    lines = []
    rest = []
    for item in raw:
        vtype = item.get("type")
        if isinstance(vtype, str):
            try:
                vtype = VectorType(vtype)
            except Exception:
                vtype = None
        bbox = item.get("bbox")
        if vtype != VectorType.LINE or not bbox or len(bbox) < 4:
            rest.append(item)
            continue
        x1, y1, x2, y2 = map(float, bbox[:4])
        bw, bh = abs(x2 - x1), abs(y2 - y1)
        is_form = bool((item.get("meta") or {}).get("form"))
        if bw <= 12 and (bh >= 40 or is_form):
            lines.append(item)
        else:
            rest.append(item)

    if len(lines) <= 1:
        return rest + lines

    lines.sort(key=lambda it: ((float(it["bbox"][0]) + float(it["bbox"][2])) / 2.0))
    kept_lines: list[dict[str, Any]] = []
    for cur in lines:
        cx = (float(cur["bbox"][0]) + float(cur["bbox"][2])) / 2.0
        cy1, cy2 = float(cur["bbox"][1]), float(cur["bbox"][3])
        dup = False
        for k in kept_lines:
            kx = (float(k["bbox"][0]) + float(k["bbox"][2])) / 2.0
            ky1, ky2 = float(k["bbox"][1]), float(k["bbox"][3])
            y_overlap = min(cy2, ky2) - max(cy1, ky1)
            if abs(cx - kx) <= 12.0 and y_overlap > -8.0:
                cur_form = bool((cur.get("meta") or {}).get("form"))
                k_form = bool((k.get("meta") or {}).get("form"))
                prefer_cur = cur_form and not k_form
                if prefer_cur or (cur_form == k_form and abs(cy2 - cy1) >= abs(ky2 - ky1)):
                    meta = dict(k.get("meta") or {})
                    k.clear()
                    k.update(cur)
                    k["meta"] = {**meta, **dict(cur.get("meta") or {})}
                dup = True
                break
        if not dup:
            kept_lines.append(dict(cur))
    return rest + kept_lines


def build_vectors(image_id: str, settings: Settings) -> dict[str, Any]:
    image_path = resolve_processed_image(image_id, settings)

    try:
        with Image.open(image_path) as img:
            img.load()
            src_w, src_h = float(img.size[0]), float(img.size[1])
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable image: {exc}") from exc

    scene = _load_json(settings.results_path / f"scene_{image_id}.json")
    if not scene:
        raise FileNotFoundError(
            f"Scene graph not found for image_id={image_id}. Run /api/scene first."
        )

    layout = _load_json(settings.results_path / f"layout_{image_id}.json")
    ocr = _load_json(settings.results_path / f"{image_id}.json")
    # typography optional
    _ = _load_json(settings.results_path / f"typography_{image_id}.json")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        pil = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    started = time.perf_counter()
    text_boxes = _ocr_text_boxes(ocr)
    masked = _mask_text_regions(bgr, text_boxes, pad=4)

    mode_info = load_document_mode(image_id, settings) or ensure_document_mode(
        image_id, settings, bgr, ocr
    )
    doc_mode = str(mode_info.get("mode") or "poster")
    hybrid = mode_info.get("hybrid") or {}
    use_form_grid = bool(hybrid.get("use_form_grid", doc_mode == "ruled_form"))
    use_color = bool(hybrid.get("use_color_regions", doc_mode != "ruled_form"))
    preserve_ornaments = bool(hybrid.get("preserve_ornaments", doc_mode == "poster"))

    # Fallback if classifier file missing older heuristics
    if not mode_info.get("mode"):
        use_form_grid = is_form_like_page(ocr)

    raw: list[dict[str, Any]] = []
    form_active = use_form_grid or (doc_mode == "ruled_form")
    form_vectors = detect_form_grid(bgr, ocr) if form_active else []

    # Optional structure borders from layout-phase cache (no second PP-Structure run)
    structure_path = settings.results_path / f"structure_{image_id}.json"
    if structure_path.is_file():
        try:
            structure = json.loads(structure_path.read_text(encoding="utf-8"))
        except Exception:
            structure = {}
        if structure.get("ok"):
            for t in structure.get("tables") or []:
                bbox = t.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = map(float, bbox[:4])
                raw.append(
                    {
                        "type": VectorType.BORDER,
                        "bbox": (x1, y1, x2, y2),
                        "fill_color": None,
                        "stroke_color": "#222222",
                        "stroke_width": 1.5,
                        "corner_radius": 0.0,
                        "confidence": 88.0,
                        "layer": 3,
                        "meta": {"source": "pp_structure", "form": True},
                    }
                )

    if doc_mode == "ruled_form":
        # Lattice is authoritative — skip scene/layout/shape noise & color washes
        raw.extend(form_vectors)
        raw = filter_non_form_noise(raw, form_active=True)
        raw = _filter_vectors_over_text(raw, text_boxes)
        merged_raw, merged_count = merge_shapes(raw)
        merged_raw = _dedupe_horizontal_rules(merged_raw)
        merged_raw = _dedupe_vertical_rules(merged_raw)
    elif doc_mode == "designed_invoice":
        # Hybrid: keep color panels + optional form lattice + light shape seed
        raw.extend(_seed_from_scene(scene, src_w, src_h))
        raw.extend(_seed_from_layout(layout))
        if form_vectors:
            raw.extend(form_vectors)
        raw.extend(detect_shapes(masked))
        raw = filter_non_form_noise(raw, form_active=False)
        raw = _filter_vectors_over_text(raw, text_boxes)
        if use_color:
            regions = detect_color_regions(bgr)
            regions = attach_gradients(bgr, regions)
            raw.extend(regions)
        merged_raw, merged_count = merge_shapes(raw)
        refine_vector_separators(
            merged_raw,
            scene.get("objects") or [],
            scene.get("page") or {},
        )
        merged_raw = _dedupe_horizontal_rules(merged_raw)
        merged_raw = _dedupe_vertical_rules(merged_raw)
    else:
        # Poster / card / flyer / simple text letter
        # Detect clean B&W text page → keep outer border only, skip grey panels/ornaments
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        colorfulness = float(np.std(bgr.astype(np.float32).reshape(-1, 3), axis=0).mean())
        blob = " ".join(
            str(b.get("text") or "") for b in (ocr or {}).get("text_blocks") or []
        )
        has_deva = any("\u0900" <= ch <= "\u097F" for ch in blob)
        clean_text_page = colorfulness < 18.0 or has_deva

        raw.extend(_seed_from_scene(scene, src_w, src_h))
        raw.extend(_seed_from_layout(layout))
        if not clean_text_page:
            raw.extend(detect_shapes(masked))
            if preserve_ornaments:
                raw.extend(detect_heart_ornaments(bgr, ocr))
            raw = filter_parchment_noise(raw)
            raw = filter_non_form_noise(raw, form_active=False)
            raw = _filter_vectors_over_text(raw, text_boxes)
            if use_color:
                regions = detect_color_regions(bgr)
                regions = attach_gradients(bgr, regions)
                raw.extend(regions)
        else:
            # Only keep long axis-aligned border-like lines near page edges
            shapes = detect_shapes(masked)
            h_img, w_img = gray.shape[:2]
            kept_shapes = []
            for s in shapes:
                vtype = s.get("type")
                # Never keep ellipses/circles/arrows on clean text pages (glyph false positives)
                type_name = getattr(vtype, "value", str(vtype)).upper()
                if type_name in {
                    "ELLIPSE",
                    "CIRCLE",
                    "ARROW",
                    "POLYGON",
                    "PATH",
                    "HEART",
                    "TRIANGLE",
                    "PANEL",
                    "COLOR_REGION",
                    "GRADIENT_REGION",
                }:
                    continue
                bbox = s.get("bbox") or (0, 0, 0, 0)
                x1, y1, x2, y2 = map(float, bbox[:4])
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                near_edge = (
                    x1 < w_img * 0.04
                    or y1 < h_img * 0.04
                    or x2 > w_img * 0.96
                    or y2 > h_img * 0.96
                )
                is_h_rule = bh <= 6 and bw > w_img * 0.55
                is_v_rule = bw <= 6 and bh > h_img * 0.55
                is_frame = near_edge and (
                    is_h_rule
                    or is_v_rule
                    or (bw > w_img * 0.85 and bh > h_img * 0.85 and (s.get("stroke_width") or 0) <= 6)
                )
                # Keep only page-edge frame / long rules (title underline near top OK if edge-ish)
                if is_frame or (is_h_rule and (y1 < h_img * 0.12 or y1 > h_img * 0.85)):
                    kept_shapes.append(s)
            raw.extend(kept_shapes)
            # Always add a thin page border for Hindi letter-style pages if none detected
            if has_deva:
                has_frame = any(
                    getattr(s.get("type"), "value", str(s.get("type") or "")).upper()
                    in {"BORDER", "RECTANGLE", "LINE"}
                    and (s.get("meta") or {}).get("form") is not False
                    for s in kept_shapes
                )
                # If we don't have both H and V edge rules, synthesize a border rect
                h_rules = sum(
                    1
                    for s in kept_shapes
                    if abs(float((s.get("bbox") or [0, 0, 0, 0])[3]) - float((s.get("bbox") or [0, 0, 0, 0])[1]))
                    <= 6
                )
                v_rules = sum(
                    1
                    for s in kept_shapes
                    if abs(float((s.get("bbox") or [0, 0, 0, 0])[2]) - float((s.get("bbox") or [0, 0, 0, 0])[0]))
                    <= 6
                )
                if h_rules < 2 or v_rules < 2:
                    margin = max(6.0, min(w_img, h_img) * 0.012)
                    raw.append(
                        {
                            "type": VectorType.BORDER,
                            "bbox": (margin, margin, w_img - margin, h_img - margin),
                            "fill_color": None,
                            "stroke_color": "#111111",
                            "stroke_width": 1.4,
                            "corner_radius": 0.0,
                            "confidence": 90.0,
                            "layer": 3,
                            "meta": {"source": "hindi_page_border", "form": False},
                        }
                    )
            # Drop scene-seeded ellipses / gray fills on clean text pages
            raw = [
                r
                for r in raw
                if getattr(r.get("type"), "value", str(r.get("type") or "")).upper()
                not in {
                    "ELLIPSE",
                    "CIRCLE",
                    "ARROW",
                    "HEART",
                    "PANEL",
                    "COLOR_REGION",
                    "GRADIENT_REGION",
                }
            ]
            raw = _filter_vectors_over_text(raw, text_boxes)

        merged_raw, merged_count = merge_shapes(raw)
        if not clean_text_page:
            refine_vector_separators(
                merged_raw,
                scene.get("objects") or [],
                scene.get("page") or {},
            )
        merged_raw = _dedupe_horizontal_rules(merged_raw)

    vectors: list[VectorObject] = []
    for i, item in enumerate(merged_raw, start=1):
        vectors.append(_to_vector_object(i, item))

    # Stable order by layer then position
    vectors.sort(key=lambda v: (v.layer, v.y, v.x, v.id))
    for i, v in enumerate(vectors, start=1):
        v.id = i

    counts = _build_counts(vectors, merged_count)
    confs = [v.confidence for v in vectors]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    elapsed = (time.perf_counter() - started) * 1000
    summary = VectorSummary(
        counts=counts,
        vector_score=round(_score(counts, avg_conf), 1),
        average_confidence=round(avg_conf, 2),
        processing_time_ms=round(elapsed, 1),
    )

    logger.info(
        "Vector reconstruction id=%s mode=%s total=%d rect=%d round=%d lines=%d paths=%d "
        "gradients=%d regions=%d merged=%d curves=%d score=%.1f time=%.1fms",
        image_id,
        doc_mode,
        counts.total,
        counts.rectangles,
        counts.rounded_rectangles,
        counts.lines,
        counts.paths,
        counts.gradients,
        counts.color_regions,
        counts.merged_shapes,
        counts.curve_count,
        summary.vector_score,
        elapsed,
    )

    results_path = settings.results_path / f"vector_{image_id}.json"
    debug_path = settings.debug_path / f"vector_{image_id}.png"
    _draw_debug(bgr, vectors, debug_path)

    payload = {
        "success": True,
        "image_id": image_id,
        "document_mode": doc_mode,
        "page": {"width": src_w, "height": src_h},
        "vectors": [v.model_dump(mode="json") for v in vectors],
        "summary": summary.model_dump(mode="json"),
        "processing_time_ms": round(elapsed, 1),
        "results_file": str(results_path),
        "debug_image": str(debug_path),
        "message": "Vector reconstruction completed successfully.",
    }
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Vector JSON saved -> %s", results_path)
    return payload


def _draw_debug(bgr: np.ndarray, vectors: list[VectorObject], output_path: Path) -> None:
    image = bgr.copy()
    colors = {
        VectorType.RECTANGLE: (255, 90, 20),
        VectorType.ROUNDED_RECTANGLE: (255, 140, 40),
        VectorType.LINE: (0, 200, 255),
        VectorType.BORDER: (0, 165, 255),
        VectorType.PANEL: (200, 100, 0),
        VectorType.COLOR_REGION: (0, 140, 255),
        VectorType.GRADIENT_REGION: (180, 0, 200),
        VectorType.CIRCLE: (0, 220, 0),
        VectorType.ELLIPSE: (0, 180, 80),
        VectorType.POLYGON: (200, 0, 200),
        VectorType.TRIANGLE: (0, 100, 255),
        VectorType.ARROW: (50, 50, 255),
        VectorType.RIBBON: (180, 80, 255),
        VectorType.WAVE: (255, 0, 180),
        VectorType.CURVED_BAND: (160, 60, 220),
        VectorType.PATH: (255, 0, 128),
        VectorType.HEART: (40, 40, 220),
    }

    for v in vectors:
        color = colors.get(v.type, (128, 128, 128))
        x1, y1 = int(v.x), int(v.y)
        x2, y2 = int(v.x + v.width), int(v.y + v.height)
        if v.type == VectorType.LINE and len(v.points) >= 2:
            p1 = (int(v.points[0].x), int(v.points[0].y))
            p2 = (int(v.points[-1].x), int(v.points[-1].y))
            cv2.line(image, p1, p2, color, max(2, int(v.stroke_width or 2)))
        elif v.type in {VectorType.CIRCLE, VectorType.ELLIPSE}:
            cv2.ellipse(
                image,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                v.rotation,
                0,
                360,
                color,
                2,
            )
        else:
            thickness = 3 if v.type in {VectorType.PANEL, VectorType.COLOR_REGION} else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Path control points
        if v.path and v.path.control_points:
            for cp in v.path.control_points[:: max(1, len(v.path.control_points) // 12)]:
                cv2.circle(image, (int(cp.x), int(cp.y)), 3, (0, 255, 255), -1)

        # Fill swatch
        if v.fill_color and len(v.fill_color) >= 7:
            try:
                r = int(v.fill_color[1:3], 16)
                g = int(v.fill_color[3:5], 16)
                b = int(v.fill_color[5:7], 16)
                cv2.rectangle(image, (x1, y1), (min(x1 + 18, x2), min(y1 + 18, y2)), (b, g, r), -1)
            except Exception:
                pass

        label = f"#{v.id} {v.type.value}"
        cv2.putText(
            image,
            label[:40],
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label[:40],
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info("Vector debug image saved -> %s", output_path)
