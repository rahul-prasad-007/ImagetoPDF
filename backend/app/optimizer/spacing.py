"""
Spacing optimization — paragraph gaps, character spacing, margin/padding, overlap.
"""

from __future__ import annotations

from typing import Any


def _box(obj: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(obj.get("x", 0))
    y = float(obj.get("y", 0))
    w = float(obj.get("width", 0))
    h = float(obj.get("height", 0))
    return x, y, x + w, y + h


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def measure_spacing_error(objects: list[dict[str, Any]]) -> float:
    """Normalized irregularity of vertical gaps between stacked text blocks."""
    texts = sorted(
        [o for o in objects if str(o.get("type")) == "TEXT"],
        key=lambda o: float(o.get("y", 0)),
    )
    if len(texts) < 3:
        return 0.0
    gaps = []
    for i in range(len(texts) - 1):
        _, _, _, b1 = _box(texts[i])
        _, y2, _, _ = _box(texts[i + 1])
        gaps.append(max(0.0, y2 - b1))
    if not gaps:
        return 0.0
    mean = sum(gaps) / len(gaps)
    if mean < 1e-6:
        return 0.0
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    cv = (var ** 0.5) / mean
    return float(min(cv / 2.0, 1.0))


def fix_paragraph_spacing(
    objects: list[dict[str, Any]],
    *,
    min_gap: float = 2.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Push overlapping / too-tight text blocks apart vertically."""
    fixes: list[str] = []
    texts = sorted(
        [o for o in objects if str(o.get("type")) == "TEXT"],
        key=lambda o: (float(o.get("y", 0)), float(o.get("x", 0))),
    )
    for i in range(len(texts) - 1):
        a, b = texts[i], texts[i + 1]
        ax1, ay1, ax2, ay2 = _box(a)
        bx1, by1, bx2, by2 = _box(b)
        # Same column-ish
        if ax2 < bx1 or bx2 < ax1:
            continue
        gap = by1 - ay2
        if gap < min_gap:
            shift = min_gap - gap
            b["y"] = round(float(b.get("y", 0)) + shift, 3)
            fixes.append(f"paragraph_spacing:{b.get('id')}")
    return objects, fixes


def fix_character_spacing(objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Clamp extreme character_spacing values in text meta/render."""
    fixes: list[str] = []
    for o in objects:
        if str(o.get("type")) != "TEXT":
            continue
        meta = dict(o.get("meta") or {})
        render = dict(meta.get("render") or {})
        text_spec = dict(render.get("text") or {})
        cs = text_spec.get("character_spacing")
        if cs is None:
            continue
        try:
            val = float(cs)
        except (TypeError, ValueError):
            continue
        clamped = max(-1.0, min(val, 8.0))
        if abs(clamped - val) > 1e-6:
            text_spec["character_spacing"] = clamped
            render["text"] = text_spec
            meta["render"] = render
            o["meta"] = meta
            fixes.append(f"char_spacing:{o.get('id')}")
    return objects, fixes


def fix_overlaps(
    objects: list[dict[str, Any]],
    *,
    types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve small overlaps by nudging the later (higher y) object down."""
    fixes: list[str] = []
    types = types or {"TEXT", "RECTANGLE", "ROUNDED_RECTANGLE", "IMAGE", "LOGO", "ICON"}
    candidates = [o for o in objects if str(o.get("type")) in types]
    candidates = sorted(candidates, key=lambda o: (int(o.get("layer", 0)), float(o.get("y", 0))))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if str(a.get("type")) == "BACKGROUND" or str(b.get("type")) == "BACKGROUND":
                continue
            ba, bb = _box(a), _box(b)
            area = _overlap_area(ba, bb)
            if area <= 0:
                continue
            aw = max(ba[2] - ba[0], 1.0) * max(ba[3] - ba[1], 1.0)
            bw = max(bb[2] - bb[0], 1.0) * max(bb[3] - bb[1], 1.0)
            ratio = area / min(aw, bw)
            if ratio < 0.08:
                continue
            # Prefer shifting text / later object
            shift = min(12.0, (ba[3] - bb[1]) + 2.0) if bb[1] < ba[3] else 2.0
            if shift > 0:
                b["y"] = round(float(b.get("y", 0)) + shift, 3)
                fixes.append(f"overlap:{a.get('id')}->{b.get('id')}")
    return objects, fixes


def fix_margins(
    objects: list[dict[str, Any]],
    page: dict[str, Any],
    *,
    min_margin: float = 8.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pull objects that sit slightly outside page content margins back in."""
    fixes: list[str] = []
    pw = float(page.get("width") or 0)
    ph = float(page.get("height") or 0)
    if pw <= 0 or ph <= 0:
        return objects, fixes
    margins = page.get("margins") or {}
    left = float(margins.get("left") or min_margin)
    top = float(margins.get("top") or min_margin)
    right = pw - float(margins.get("right") or min_margin)
    bottom = ph - float(margins.get("bottom") or min_margin)

    for o in objects:
        if str(o.get("type")) == "BACKGROUND":
            continue
        x, y, w, h = float(o.get("x", 0)), float(o.get("y", 0)), float(o.get("width", 0)), float(
            o.get("height", 0)
        )
        nx, ny = x, y
        if x < left - 1 and x > left - 20:
            nx = left
        if y < top - 1 and y > top - 20:
            ny = top
        if x + w > right + 1 and x + w < right + 20:
            nx = right - w
        if y + h > bottom + 1 and y + h < bottom + 20:
            ny = bottom - h
        if abs(nx - x) > 0.5 or abs(ny - y) > 0.5:
            o["x"] = round(nx, 3)
            o["y"] = round(ny, 3)
            fixes.append(f"margin:{o.get('id')}")
    return objects, fixes
