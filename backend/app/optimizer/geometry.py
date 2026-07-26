"""
Geometry optimization — snap rectangles, borders, panels, lines to detected edges.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_dominant_edges(
    image: np.ndarray,
    *,
    max_lines: int = 80,
) -> dict[str, list[float]]:
    """
    Detect strong horizontal/vertical edge positions (in image pixel coords).
    Returns {"x": [...], "y": [...]}.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=min(gray.shape[:2]) // 8,
        maxLineGap=8,
    )
    xs: list[float] = []
    ys: list[float] = []
    if lines is None:
        return {"x": xs, "y": ys}

    for line in lines[:max_lines]:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 3:
            xs.append(float((x1 + x2) / 2))
        elif abs(y2 - y1) < 3:
            ys.append(float((y1 + y2) / 2))

    def _cluster(vals: list[float], tol: float = 4.0) -> list[float]:
        if not vals:
            return []
        vals = sorted(vals)
        clusters: list[list[float]] = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - clusters[-1][-1]) <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [float(sum(c) / len(c)) for c in clusters if len(c) >= 1]

    return {"x": _cluster(xs), "y": _cluster(ys)}


def _nearest(val: float, candidates: list[float], max_dist: float) -> float | None:
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c - val))
    if abs(best - val) <= max_dist:
        return best
    return None


def source_to_scene_xy(
    sx: float,
    sy: float,
    page: dict[str, Any],
) -> tuple[float, float]:
    return (
        float(page.get("offset_x") or 0) + sx * float(page.get("scale_x") or 1),
        float(page.get("offset_y") or 0) + sy * float(page.get("scale_y") or 1),
    )


def snap_shapes_to_edges(
    objects: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
    original: np.ndarray,
    page: dict[str, Any],
    *,
    max_snap_px: float = 6.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Snap shape/vector edges toward detected image edges (mapped into scene space)."""
    fixes: list[str] = []
    edges = detect_dominant_edges(original)
    sx = float(page.get("scale_x") or 1.0)
    sy = float(page.get("scale_y") or 1.0)
    ox = float(page.get("offset_x") or 0.0)
    oy = float(page.get("offset_y") or 0.0)

    edge_x_scene = [ox + e * sx for e in edges["x"]]
    edge_y_scene = [oy + e * sy for e in edges["y"]]
    max_snap = max_snap_px * max(sx, sy)

    shape_types = {
        "RECTANGLE",
        "ROUNDED_RECTANGLE",
        "LINE",
        "ELLIPSE",
        "CIRCLE",
        "POLYGON",
        "PATH",
        "BACKGROUND",
    }

    for o in objects:
        if str(o.get("type")) not in shape_types:
            continue
        x = float(o.get("x", 0))
        y = float(o.get("y", 0))
        w = float(o.get("width", 0))
        h = float(o.get("height", 0))
        changed = False

        nx = _nearest(x, edge_x_scene, max_snap)
        if nx is not None and abs(nx - x) > 0.4:
            dw = x - nx
            o["x"] = round(nx, 3)
            o["width"] = round(max(1.0, w + dw), 3)
            changed = True

        nr = _nearest(x + w, edge_x_scene, max_snap)
        if nr is not None and abs(nr - (x + w)) > 0.4:
            o["width"] = round(max(1.0, nr - float(o["x"])), 3)
            changed = True

        ny = _nearest(y, edge_y_scene, max_snap)
        if ny is not None and abs(ny - y) > 0.4:
            dh = y - ny
            o["y"] = round(ny, 3)
            o["height"] = round(max(1.0, h + dh), 3)
            changed = True

        nb = _nearest(y + h, edge_y_scene, max_snap)
        if nb is not None and abs(nb - (y + h)) > 0.4:
            o["height"] = round(max(1.0, nb - float(o["y"])), 3)
            changed = True

        if changed:
            fixes.append(f"geometry_snap:{o.get('id')}")

    for v in vectors:
        bbox = v.get("bbox") or v.get("source_bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(c) for c in bbox[:4]]
        changed = False
        nx1 = _nearest(x1, edges["x"], max_snap_px)
        if nx1 is not None and abs(nx1 - x1) > 0.5:
            x1 = nx1
            changed = True
        nx2 = _nearest(x2, edges["x"], max_snap_px)
        if nx2 is not None and abs(nx2 - x2) > 0.5:
            x2 = nx2
            changed = True
        ny1 = _nearest(y1, edges["y"], max_snap_px)
        if ny1 is not None and abs(ny1 - y1) > 0.5:
            y1 = ny1
            changed = True
        ny2 = _nearest(y2, edges["y"], max_snap_px)
        if ny2 is not None and abs(ny2 - y2) > 0.5:
            y2 = ny2
            changed = True
        if changed and x2 > x1 and y2 > y1:
            new_bbox = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
            if "bbox" in v:
                v["bbox"] = new_bbox
            if "source_bbox" in v:
                v["source_bbox"] = new_bbox
            if "x" in v:
                sx1, sy1 = source_to_scene_xy(x1, y1, page)
                sx2, sy2 = source_to_scene_xy(x2, y2, page)
                v["x"], v["y"] = round(sx1, 3), round(sy1, 3)
                v["width"], v["height"] = round(sx2 - sx1, 3), round(sy2 - sy1, 3)
            fixes.append(f"vector_geometry_snap:{v.get('id')}")

    return objects, vectors, fixes


def correct_bbox_drift(
    objects: list[dict[str, Any]],
    diffs: list[dict[str, Any]],
    *,
    max_correct: float = 8.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply measured offset corrections (negate drift) for small errors."""
    fixes: list[str] = []
    by_id = {int(d["object_id"]): d for d in diffs if "object_id" in d}
    for o in objects:
        oid = o.get("id")
        if oid is None or int(oid) not in by_id:
            continue
        d = by_id[int(oid)]
        ox = float(d.get("offset_x") or 0)
        oy = float(d.get("offset_y") or 0)
        if abs(ox) < 0.5 and abs(oy) < 0.5:
            continue
        if abs(ox) > max_correct or abs(oy) > max_correct:
            continue
        o["x"] = round(float(o.get("x", 0)) - ox, 3)
        o["y"] = round(float(o.get("y", 0)) - oy, 3)
        dw = float(d.get("width_difference") or 0)
        dh = float(d.get("height_difference") or 0)
        if abs(dw) <= max_correct:
            o["width"] = round(max(1.0, float(o.get("width", 1)) - dw), 3)
        if abs(dh) <= max_correct:
            o["height"] = round(max(1.0, float(o.get("height", 1)) - dh), 3)
        fixes.append(f"bbox_drift:{oid}")
    return objects, fixes
