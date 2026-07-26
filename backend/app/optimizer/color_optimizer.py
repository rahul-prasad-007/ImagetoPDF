"""
Color optimization — correct RGB, gradient, and opacity mismatches vs original.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.optimizer.similarity import mean_region_color


def _bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(np.clip(round(float(c)), 0, 255)) for c in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def _parse_hex(value: str | None) -> np.ndarray | None:
    if not value or not isinstance(value, str):
        return None
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return np.array([b, g, r], dtype=np.float64)
    except ValueError:
        return None


def _blend_toward(current: np.ndarray | None, target: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    if current is None:
        return target
    return current * (1.0 - alpha) + target * alpha


def optimize_object_colors(
    objects: list[dict[str, Any]],
    original: np.ndarray,
    page: dict[str, Any],
    *,
    threshold: float = 12.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Sample original image under each object's source bbox and correct fill/font colors.
    """
    fixes: list[str] = []
    sx = float(page.get("scale_x") or 1.0)
    sy = float(page.get("scale_y") or 1.0)
    ox = float(page.get("offset_x") or 0.0)
    oy = float(page.get("offset_y") or 0.0)
    h, w = original.shape[:2]

    # Global page/background color from border median (robust for solid posters)
    border = np.concatenate(
        [
            original[0:4, :].reshape(-1, 3),
            original[-4:, :].reshape(-1, 3),
            original[:, 0:4].reshape(-1, 3),
            original[:, -4:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float64)
    page_bg = np.median(border, axis=0)

    for o in objects:
        src = o.get("source") or {}
        bbox = src.get("source_bbox") or src.get("bbox")
        if not bbox or len(bbox) < 4:
            x = (float(o.get("x", 0)) - ox) / max(sx, 1e-6)
            y = (float(o.get("y", 0)) - oy) / max(sy, 1e-6)
            bw = float(o.get("width", 0)) / max(sx, 1e-6)
            bh = float(o.get("height", 0)) / max(sy, 1e-6)
            bbox = [x, y, x + bw, y + bh]

        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        if x2i - x1i < 2 or y2i - y1i < 2:
            continue

        target = mean_region_color(original, [x1i, y1i, x2i, y2i])
        otype = str(o.get("type") or "")

        if otype == "TEXT":
            crop = original[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue
            flat = crop.reshape(-1, 3).astype(np.float64)
            lum = 0.114 * flat[:, 0] + 0.587 * flat[:, 1] + 0.299 * flat[:, 2]
            order = np.argsort(lum)
            dark = flat[order[: max(1, len(order) // 5)]].mean(axis=0)
            light = flat[order[-max(1, len(order) // 5) :]].mean(axis=0)
            bg = flat.mean(axis=0)
            text_col = dark if np.linalg.norm(dark - bg) >= np.linalg.norm(light - bg) else light
            current = _parse_hex(o.get("font_color"))
            if current is not None and np.linalg.norm(current - text_col) < threshold:
                continue
            blended = _blend_toward(current, text_col, 0.85)
            o["font_color"] = _bgr_to_hex(blended)
            fixes.append(f"text_color:{o.get('id')}")
        else:
            # Large backgrounds: snap strongly to page/region color
            area = (x2i - x1i) * (y2i - y1i)
            is_bg = otype == "BACKGROUND" or area > 0.35 * w * h
            sample = page_bg if is_bg else target
            blend = 0.9 if is_bg else 0.7
            current = _parse_hex(o.get("fill_color"))
            if current is not None and np.linalg.norm(current - sample) < threshold:
                pass
            elif o.get("fill_color") or otype in {
                "RECTANGLE",
                "ROUNDED_RECTANGLE",
                "ELLIPSE",
                "CIRCLE",
                "BACKGROUND",
                "POLYGON",
            }:
                blended = _blend_toward(current, sample, blend)
                o["fill_color"] = _bgr_to_hex(blended)
                fixes.append(f"fill_color:{o.get('id')}")

            op = o.get("opacity")
            if op is not None:
                try:
                    opf = float(op)
                    clamped = max(0.05, min(1.0, opf))
                    if abs(clamped - opf) > 1e-6:
                        o["opacity"] = round(clamped, 3)
                        fixes.append(f"opacity:{o.get('id')}")
                except (TypeError, ValueError):
                    o["opacity"] = 1.0
                    fixes.append(f"opacity:{o.get('id')}")

    return objects, fixes


def optimize_vector_colors(
    vectors: list[dict[str, Any]],
    original: np.ndarray,
    *,
    threshold: float = 18.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Correct fill/stroke/gradient stop colors on vector objects."""
    fixes: list[str] = []
    h, w = original.shape[:2]

    for v in vectors:
        bbox = v.get("bbox") or v.get("source_bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(c) for c in bbox[:4]]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        if x2i - x1i < 2 or y2i - y1i < 2:
            continue
        target = mean_region_color(original, [x1i, y1i, x2i, y2i])

        fill = v.get("fill") or v.get("fill_color")
        current = _parse_hex(fill if isinstance(fill, str) else None)
        if current is None or np.linalg.norm(current - target) >= threshold:
            new_hex = _bgr_to_hex(_blend_toward(current, target, 0.55))
            if "fill" in v:
                v["fill"] = new_hex
            if "fill_color" in v:
                v["fill_color"] = new_hex
            if "fill" not in v and "fill_color" not in v:
                v["fill_color"] = new_hex
            fixes.append(f"vector_fill:{v.get('id')}")

        # Gradient stops
        grad = v.get("gradient") or {}
        stops = grad.get("stops") if isinstance(grad, dict) else None
        if isinstance(stops, list) and stops:
            # Sample left/right thirds for linear gradients
            mid_y = (y1i + y2i) // 2
            left = mean_region_color(original, [x1i, max(y1i, mid_y - 2), x1i + max(2, (x2i - x1i) // 3), min(h, mid_y + 2)])
            right = mean_region_color(
                original,
                [x1i + 2 * (x2i - x1i) // 3, max(y1i, mid_y - 2), x2i, min(h, mid_y + 2)],
            )
            samples = [left, right]
            for i, stop in enumerate(stops):
                if not isinstance(stop, dict):
                    continue
                col = stop.get("color")
                cur = _parse_hex(col if isinstance(col, str) else None)
                samp = samples[min(i, len(samples) - 1)]
                if cur is None or np.linalg.norm(cur - samp) >= threshold * 0.7:
                    stop["color"] = _bgr_to_hex(_blend_toward(cur, samp, 0.6))
                    fixes.append(f"gradient_stop:{v.get('id')}:{i}")
            grad["stops"] = stops
            v["gradient"] = grad

        op = v.get("opacity")
        if op is not None:
            try:
                opf = float(op)
                clamped = max(0.05, min(1.0, opf))
                if abs(clamped - opf) > 1e-6:
                    v["opacity"] = round(clamped, 3)
                    fixes.append(f"vector_opacity:{v.get('id')}")
            except (TypeError, ValueError):
                pass

    return vectors, fixes


def measure_object_color_diff(
    original: np.ndarray,
    rendered_content: np.ndarray,
    bbox_src: list[float],
    content_origin: tuple[float, float],
    content_scale: tuple[float, float],
) -> float:
    """Color delta for one object between original source bbox and mapped rendered region."""
    ox, oy = content_origin
    sx, sy = content_scale
    x1, y1, x2, y2 = [float(v) for v in bbox_src[:4]]
    # Original sample
    ca = mean_region_color(original, [x1, y1, x2, y2])
    # Rendered content is already cropped to source area at original resolution ideally
    rx1 = (x1 - 0) * sx
    ry1 = (y1 - 0) * sy
    rx2 = (x2 - 0) * sx
    ry2 = (y2 - 0) * sy
    cb = mean_region_color(rendered_content, [rx1, ry1, rx2, ry2])
    return float(np.linalg.norm(ca - cb) / (255.0 * np.sqrt(3)))
