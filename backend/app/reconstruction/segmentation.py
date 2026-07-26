"""
Merge / dedupe helpers for reconstruction planning.

- Merge nearby same-color rectangles / background panels
- Merge continuous collinear lines
- Drop near-duplicate objects
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _bbox(obj: dict[str, Any]) -> tuple[float, float, float, float]:
    b = obj.get("bbox") or [0, 0, 0, 0]
    if len(b) >= 4:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return 0.0, 0.0, 0.0, 0.0


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _color_key(obj: dict[str, Any], quant: int = 24) -> tuple[int, int, int] | None:
    meta = obj.get("meta") or {}
    color = meta.get("color_bgr") or meta.get("color_rgb")
    if not color or len(color) < 3:
        return None
    # Quantize to merge near-identical fills
    return (
        int(color[0]) // quant * quant,
        int(color[1]) // quant * quant,
        int(color[2]) // quant * quant,
    )


def _expand(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _near(a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float) -> bool:
    # Expand both by gap and check overlap
    ae = (a[0] - gap, a[1] - gap, a[2] + gap, a[3] + gap)
    return _iou(ae, b) > 0 or (
        not (ae[2] < b[0] or b[2] < ae[0] or ae[3] < b[1] or b[3] < ae[1])
    )


def remove_duplicates(objects: list[dict[str, Any]], iou_thresh: float = 0.85) -> list[dict[str, Any]]:
    """Keep larger object when two same-type boxes heavily overlap."""
    ordered = sorted(objects, key=lambda o: _area(_bbox(o)), reverse=True)
    kept: list[dict[str, Any]] = []
    for obj in ordered:
        ob = _bbox(obj)
        otype = obj.get("type")
        dup = False
        for k in kept:
            if k.get("type") != otype:
                continue
            if _iou(ob, _bbox(k)) >= iou_thresh:
                dup = True
                # Track merge provenance
                merged = list(k.get("_merged_from") or [])
                merged.append(int(obj.get("id") or 0))
                k["_merged_from"] = merged
                break
        if not dup:
            obj = dict(obj)
            obj["_merged_from"] = list(obj.get("_merged_from") or [])
            kept.append(obj)
    return kept


def merge_same_color_rectangles(
    objects: list[dict[str, Any]],
    gap: float = 12.0,
) -> list[dict[str, Any]]:
    """
    Merge nearby RECTANGLE / BACKGROUND_SHAPE / ROUNDED_RECTANGLE of same quantized color.
    """
    mergeable_types = {"RECTANGLE", "BACKGROUND_SHAPE", "ROUNDED_RECTANGLE"}
    others = [o for o in objects if o.get("type") not in mergeable_types]
    candidates = [dict(o) for o in objects if o.get("type") in mergeable_types]

    changed = True
    while changed:
        changed = False
        n = len(candidates)
        used = [False] * n
        merged: list[dict[str, Any]] = []
        for i in range(n):
            if used[i]:
                continue
            cur = candidates[i]
            cb = _bbox(cur)
            ck = _color_key(cur)
            ctype = cur.get("type")
            merged_ids = list(cur.get("_merged_from") or [])
            if cur.get("id") is not None:
                # keep self out of merged_from until partners added
                pass
            for j in range(i + 1, n):
                if used[j]:
                    continue
                other = candidates[j]
                if other.get("type") not in mergeable_types:
                    continue
                # Allow BACKGROUND_SHAPE ↔ RECTANGLE merge if same color
                ok = _color_key(other)
                if ck is not None and ok is not None and ck != ok:
                    continue
                if ck is not None and ok is None:
                    continue
                if ck is None and ok is not None:
                    continue
                if not _near(cb, _bbox(other), gap):
                    continue
                # Merge
                used[j] = True
                cb = _expand(cb, _bbox(other))
                merged_ids.append(int(other.get("id") or 0))
                merged_ids.extend(other.get("_merged_from") or [])
                # Prefer more specific type
                if ctype == "BACKGROUND_SHAPE" and other.get("type") != "BACKGROUND_SHAPE":
                    ctype = other.get("type")
                changed = True
            used[i] = True
            out = dict(cur)
            out["type"] = ctype
            out["bbox"] = [round(cb[0], 2), round(cb[1], 2), round(cb[2], 2), round(cb[3], 2)]
            out["width"] = round(cb[2] - cb[0], 2)
            out["height"] = round(cb[3] - cb[1], 2)
            out["area"] = round(_area(cb), 2)
            out["center_x"] = round((cb[0] + cb[2]) / 2, 2)
            out["center_y"] = round((cb[1] + cb[3]) / 2, 2)
            out["_merged_from"] = [m for m in merged_ids if m and m != out.get("id")]
            merged.append(out)
        candidates = merged

    return others + candidates


def merge_continuous_lines(objects: list[dict[str, Any]], gap: float = 10.0) -> list[dict[str, Any]]:
    """Merge collinear / nearly continuous LINE objects."""
    others = [o for o in objects if o.get("type") != "LINE"]
    lines = [dict(o) for o in objects if o.get("type") == "LINE"]
    if len(lines) <= 1:
        return objects

    def orientation(obj: dict[str, Any]) -> str:
        meta = obj.get("meta") or {}
        if meta.get("orientation") in {"horizontal", "vertical"}:
            return meta["orientation"]
        x1, y1, x2, y2 = _bbox(obj)
        return "horizontal" if abs(x2 - x1) >= abs(y2 - y1) else "vertical"

    changed = True
    while changed:
        changed = False
        n = len(lines)
        used = [False] * n
        merged: list[dict[str, Any]] = []
        for i in range(n):
            if used[i]:
                continue
            cur = lines[i]
            ori = orientation(cur)
            cb = _bbox(cur)
            merged_ids = list(cur.get("_merged_from") or [])
            for j in range(i + 1, n):
                if used[j]:
                    continue
                other = lines[j]
                if orientation(other) != ori:
                    continue
                ob = _bbox(other)
                if ori == "horizontal":
                    # Same row band and x-gap small
                    if abs(((cb[1] + cb[3]) / 2) - ((ob[1] + ob[3]) / 2)) > gap:
                        continue
                    if ob[0] > cb[2] + gap or cb[0] > ob[2] + gap:
                        continue
                else:
                    if abs(((cb[0] + cb[2]) / 2) - ((ob[0] + ob[2]) / 2)) > gap:
                        continue
                    if ob[1] > cb[3] + gap or cb[1] > ob[3] + gap:
                        continue
                used[j] = True
                cb = _expand(cb, ob)
                merged_ids.append(int(other.get("id") or 0))
                merged_ids.extend(other.get("_merged_from") or [])
                changed = True
            used[i] = True
            out = dict(cur)
            out["bbox"] = [round(cb[0], 2), round(cb[1], 2), round(cb[2], 2), round(cb[3], 2)]
            out["width"] = round(cb[2] - cb[0], 2)
            out["height"] = round(cb[3] - cb[1], 2)
            out["area"] = round(_area(cb), 2)
            out["_merged_from"] = [m for m in merged_ids if m and m != out.get("id")]
            meta = dict(out.get("meta") or {})
            meta["orientation"] = ori
            out["meta"] = meta
            merged.append(out)
        lines = merged

    return others + lines


def prepare_objects_for_planning(layout_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full merge pipeline before decision rules."""
    objs = [dict(o) for o in layout_objects]
    objs = remove_duplicates(objs, iou_thresh=0.88)
    objs = merge_same_color_rectangles(objs, gap=14.0)
    objs = merge_continuous_lines(objs, gap=12.0)
    objs = remove_duplicates(objs, iou_thresh=0.9)
    return objs
