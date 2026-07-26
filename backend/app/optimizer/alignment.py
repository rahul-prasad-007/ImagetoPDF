"""
Alignment analysis — text/object column & baseline snap metrics and corrections.
"""

from __future__ import annotations

from typing import Any


def _center_x(obj: dict[str, Any]) -> float:
    return float(obj.get("x", 0)) + float(obj.get("width", 0)) / 2.0


def _left(obj: dict[str, Any]) -> float:
    return float(obj.get("x", 0))


def _right(obj: dict[str, Any]) -> float:
    return float(obj.get("x", 0)) + float(obj.get("width", 0))


def _top(obj: dict[str, Any]) -> float:
    return float(obj.get("y", 0))


def _bottom(obj: dict[str, Any]) -> float:
    return float(obj.get("y", 0)) + float(obj.get("height", 0))


def measure_alignment_error(objects: list[dict[str, Any]], tolerance: float = 4.0) -> float:
    """
    Normalized alignment error: fraction of left/center/right edges that
    fail to cluster within tolerance of a dominant column.
    """
    texts = [o for o in objects if str(o.get("type")) == "TEXT" and float(o.get("width", 0)) > 0]
    if len(texts) < 2:
        return 0.0

    lefts = sorted(_left(o) for o in texts)
    # Cluster left edges
    clusters: list[list[float]] = []
    for v in lefts:
        placed = False
        for c in clusters:
            if abs(c[0] - v) <= tolerance * 3:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])

    # Error = average distance to nearest cluster median, normalized
    medians = [sorted(c)[len(c) // 2] for c in clusters]
    errs = []
    for o in texts:
        x = _left(o)
        d = min(abs(x - m) for m in medians)
        errs.append(min(d / max(tolerance * 10, 1.0), 1.0))
    return float(sum(errs) / len(errs)) if errs else 0.0


def snap_text_alignment(
    objects: list[dict[str, Any]],
    *,
    tolerance: float = 14.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Snap near-aligned text left edges and baselines; return (objects, fix labels)."""
    fixes: list[str] = []
    texts = [o for o in objects if str(o.get("type")) == "TEXT"]
    if len(texts) < 2:
        return objects, fixes

    # Snap left edges — larger tolerance for page-space scene coords
    page_span = max((float(o.get("x", 0)) + float(o.get("width", 0)) for o in texts), default=1000.0)
    tol = max(tolerance, page_span * 0.01)
    lefts = sorted((_left(o), int(o.get("id") or 0), o) for o in texts)
    clusters: list[list[tuple[float, dict]]] = []
    for v, _oid, o in lefts:
        placed = False
        for c in clusters:
            if abs(c[0][0] - v) <= tol:
                c.append((v, o))
                placed = True
                break
        if not placed:
            clusters.append([(v, o)])

    for c in clusters:
        if len(c) < 2:
            continue
        target = sorted(v for v, _ in c)[len(c) // 2]
        for v, o in c:
            if abs(v - target) > 0.5:
                o["x"] = round(target, 3)
                fixes.append(f"text_align_left:{o.get('id')}")

    # Snap digit columns separately (stronger)
    numbers = [
        o
        for o in texts
        if (str(o.get("content") or "").strip().rstrip(".").isdigit()
            and len(str(o.get("content") or "").strip()) <= 3)
    ]
    if len(numbers) >= 2:
        target = sorted(_left(o) for o in numbers)[len(numbers) // 2]
        for o in numbers:
            if abs(_left(o) - target) > 0.5:
                o["x"] = round(target, 3)
                fixes.append(f"number_col:{o.get('id')}")

    # Snap baselines (bottom edges) within rows
    texts_sorted = sorted(texts, key=_top)
    row: list[dict] = []
    rows: list[list[dict]] = []
    for o in texts_sorted:
        if not row:
            row = [o]
            continue
        if abs(_top(o) - _top(row[0])) <= tol * 1.5:
            row.append(o)
        else:
            rows.append(row)
            row = [o]
    if row:
        rows.append(row)

    for row_objs in rows:
        if len(row_objs) < 2:
            continue
        bottoms = [_bottom(o) for o in row_objs]
        target_b = sorted(bottoms)[len(bottoms) // 2]
        for o in row_objs:
            b = _bottom(o)
            if abs(b - target_b) > 0.5 and abs(b - target_b) <= tol * 2:
                o["y"] = round(target_b - float(o.get("height", 0)), 3)
                fixes.append(f"text_baseline:{o.get('id')}")

    return objects, fixes


def snap_object_alignment(
    objects: list[dict[str, Any]],
    *,
    tolerance: float = 5.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Snap non-text shape left/top edges that are nearly co-linear."""
    fixes: list[str] = []
    shapes = [
        o
        for o in objects
        if str(o.get("type"))
        in {
            "RECTANGLE",
            "ROUNDED_RECTANGLE",
            "LINE",
            "ELLIPSE",
            "CIRCLE",
            "POLYGON",
            "PATH",
            "PANEL",
            "BACKGROUND",
        }
    ]
    if len(shapes) < 2:
        return objects, fixes

    for attr, getter in (("x", _left), ("y", _top)):
        items = sorted((getter(o), int(o.get("id") or 0), o) for o in shapes)
        clusters: list[list[tuple[float, dict]]] = []
        for v, _oid, o in items:
            placed = False
            for c in clusters:
                if abs(c[0][0] - v) <= tolerance:
                    c.append((v, o))
                    placed = True
                    break
            if not placed:
                clusters.append([(v, o)])
        for c in clusters:
            if len(c) < 2:
                continue
            target = sorted(v for v, _ in c)[len(c) // 2]
            for v, o in c:
                if 0.5 < abs(v - target) <= tolerance:
                    o[attr] = round(target, 3)
                    fixes.append(f"object_align_{attr}:{o.get('id')}")

    return objects, fixes
