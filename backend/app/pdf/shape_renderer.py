"""
Vector / shape rendering — real PDF path & geometry operators (never rasterized).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from reportlab.lib.colors import Color
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfgen.pathobject import PDFPathObject

from app.pdf.page_renderer import PageContext, parse_color

logger = logging.getLogger(__name__)

_SHAPE_TYPES = {
    "RECTANGLE",
    "ROUNDED_RECTANGLE",
    "LINE",
    "BORDER",
    "PANEL",
    "COLOR_REGION",
    "CIRCLE",
    "ELLIPSE",
    "POLYGON",
    "TRIANGLE",
    "ARROW",
    "RIBBON",
    "WAVE",
    "CURVED_BAND",
    "PATH",
    "HEART",
    "GRADIENT_REGION",
    "BACKGROUND",
}


def _apply_fill_stroke(
    canvas: Canvas,
    fill: Optional[str],
    stroke: Optional[str],
    stroke_width: float,
    opacity: float,
) -> tuple[bool, bool]:
    do_fill = bool(fill)
    do_stroke = bool(stroke) and stroke_width > 0
    if do_fill:
        canvas.setFillColor(parse_color(fill, alpha=opacity))
        try:
            canvas.setFillAlpha(opacity)
        except Exception:
            pass
    if do_stroke:
        canvas.setStrokeColor(parse_color(stroke, alpha=opacity))
        try:
            canvas.setStrokeAlpha(opacity)
        except Exception:
            pass
        canvas.setLineWidth(max(0.25, stroke_width))
    return do_fill, do_stroke


def _paint(canvas: Canvas, do_fill: bool, do_stroke: bool) -> None:
    if do_fill and do_stroke:
        canvas.fillStroke()
    elif do_fill:
        canvas.fill()
    elif do_stroke:
        canvas.stroke()


def _draw_linear_gradient_approx(
    canvas: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    start: str,
    end: str,
    angle: float,
    opacity: float,
    steps: int = 24,
) -> None:
    """Approximate gradient with thin strips (still vector rects, not a bitmap)."""
    c0 = parse_color(start, alpha=opacity)
    c1 = parse_color(end, alpha=opacity)
    horizontal = abs(angle % 180) < 45 or abs(angle % 180) > 135
    for i in range(steps):
        t = i / max(1, steps - 1)
        color = Color(
            c0.red + (c1.red - c0.red) * t,
            c0.green + (c1.green - c0.green) * t,
            c0.blue + (c1.blue - c0.blue) * t,
            alpha=opacity,
        )
        canvas.setFillColor(color)
        if horizontal:
            sx = x + w * (i / steps)
            canvas.rect(sx, y, w / steps + 0.5, h, stroke=0, fill=1)
        else:
            sy = y + h * (i / steps)
            canvas.rect(x, sy, w, h / steps + 0.5, stroke=0, fill=1)


def _parse_svg_path_to_pdf(
    canvas: Canvas,
    commands: str,
    ctx: PageContext,
    from_source: bool,
) -> Optional[PDFPathObject]:
    if not commands:
        return None
    tokens = re.findall(r"[MmLlQqCcZz]|-?\d*\.?\d+(?:e[-+]?\d+)?", commands)
    if not tokens:
        return None

    path = canvas.beginPath()
    i = 0
    cmd = "M"
    cx = cy = 0.0

    def map_pt(x: float, y: float) -> tuple[float, float]:
        if from_source:
            return ctx.source_point_to_pdf(x, y)
        px = x * ctx.scale_x
        py = ctx.pdf_height - y * ctx.scale_y
        return px, py

    def read_num() -> float:
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        t = tokens[i]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            i += 1
            if cmd in {"Z", "z"}:
                path.close()
                continue
        try:
            if cmd in {"M", "m"}:
                x, y = read_num(), read_num()
                if cmd == "m":
                    x, y = cx + x, cy + y
                cx, cy = x, y
                px, py = map_pt(x, y)
                path.moveTo(px, py)
                cmd = "L" if cmd == "M" else "l"
            elif cmd in {"L", "l"}:
                x, y = read_num(), read_num()
                if cmd == "l":
                    x, y = cx + x, cy + y
                cx, cy = x, y
                px, py = map_pt(x, y)
                path.lineTo(px, py)
            elif cmd in {"Q", "q"}:
                x1, y1 = read_num(), read_num()
                x, y = read_num(), read_num()
                if cmd == "q":
                    x1, y1 = cx + x1, cy + y1
                    x, y = cx + x, cy + y
                # Convert quadratic to cubic for ReportLab
                c1x = cx + 2 / 3 * (x1 - cx)
                c1y = cy + 2 / 3 * (y1 - cy)
                c2x = x + 2 / 3 * (x1 - x)
                c2y = y + 2 / 3 * (y1 - y)
                p1 = map_pt(c1x, c1y)
                p2 = map_pt(c2x, c2y)
                p3 = map_pt(x, y)
                path.curveTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
                cx, cy = x, y
            elif cmd in {"C", "c"}:
                x1, y1 = read_num(), read_num()
                x2, y2 = read_num(), read_num()
                x, y = read_num(), read_num()
                if cmd == "c":
                    x1, y1 = cx + x1, cy + y1
                    x2, y2 = cx + x2, cy + y2
                    x, y = cx + x, cy + y
                p1 = map_pt(x1, y1)
                p2 = map_pt(x2, y2)
                p3 = map_pt(x, y)
                path.curveTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
                cx, cy = x, y
            else:
                i += 1
        except (IndexError, ValueError):
            break
    return path


def render_vector_object(
    canvas: Canvas,
    obj: dict[str, Any],
    ctx: PageContext,
    *,
    from_source: bool = True,
) -> bool:
    """
    Render a vector JSON object (source coords) or scene shape (scene coords).
    """
    vtype = str(obj.get("type") or "").upper()
    if vtype not in _SHAPE_TYPES:
        return False

    opacity = float(obj.get("opacity") or 1.0)
    fill = obj.get("fill_color") or obj.get("fill")
    stroke = obj.get("stroke_color") or obj.get("stroke")
    stroke_width = float(obj.get("stroke_width") or 0.0)
    if from_source:
        stroke_width_pdf = stroke_width * ctx.scale_x * ctx.src_scale_x
    else:
        stroke_width_pdf = stroke_width * ctx.scale_x

    # Geometry
    if from_source:
        x = float(obj.get("x") or 0)
        y = float(obj.get("y") or 0)
        w = float(obj.get("width") or 0)
        h = float(obj.get("height") or 0)
        px, py, pw, ph = ctx.source_rect_to_pdf(x, y, w, h)
    else:
        x = float(obj.get("x") or 0)
        y = float(obj.get("y") or 0)
        w = float(obj.get("width") or 0)
        h = float(obj.get("height") or 0)
        px, py, pw, ph = ctx.scene_rect_to_pdf(x, y, w, h)

    if pw < 0.2 and ph < 0.2 and vtype != "LINE":
        return False

    canvas.saveState()
    try:
        try:
            canvas.setFillAlpha(opacity)
            canvas.setStrokeAlpha(opacity)
        except Exception:
            pass

        gradient = obj.get("gradient")
        if isinstance(gradient, dict) and gradient.get("kind") in {"LINEAR", "RADIAL"}:
            _draw_linear_gradient_approx(
                canvas,
                px,
                py,
                max(pw, 1),
                max(ph, 1),
                gradient.get("start_color") or fill or "#FFFFFF",
                gradient.get("end_color") or "#CCCCCC",
                float(gradient.get("angle") or 0),
                opacity,
            )
            if stroke and stroke_width_pdf > 0:
                canvas.setStrokeColor(parse_color(stroke, alpha=opacity))
                canvas.setLineWidth(stroke_width_pdf)
                canvas.rect(px, py, pw, ph, stroke=1, fill=0)
            return True

        if vtype == "HEART":
            # Parametric heart outline (reliable vs noisy contour paths)
            path = canvas.beginPath()
            steps = 64
            first = True
            for i in range(steps + 1):
                t = math.pi * 2 * i / steps
                hx = 16 * math.sin(t) ** 3
                hy = (
                    13 * math.cos(t)
                    - 5 * math.cos(2 * t)
                    - 2 * math.cos(3 * t)
                    - math.cos(4 * t)
                )
                nx = (hx + 16) / 32.0
                ny = (hy + 17) / 29.0
                qx = px + nx * pw
                qy = py + ny * ph
                if first:
                    path.moveTo(qx, qy)
                    first = False
                else:
                    path.lineTo(qx, qy)
            path.close()
            do_fill, do_stroke = _apply_fill_stroke(
                canvas, fill, stroke or "#2A2A2A", max(0.8, stroke_width_pdf or 1.2), opacity
            )
            canvas.drawPath(path, fill=1 if do_fill else 0, stroke=1 if do_stroke else 0)
            return True

        path_data = obj.get("path")
        if isinstance(path_data, dict) and path_data.get("commands") and vtype in {
            "PATH",
            "WAVE",
            "RIBBON",
            "CURVED_BAND",
            "ARROW",
            "POLYGON",
            "TRIANGLE",
        }:
            path = _parse_svg_path_to_pdf(canvas, path_data["commands"], ctx, from_source)
            if path is not None:
                do_fill, do_stroke = _apply_fill_stroke(
                    canvas, fill, stroke, stroke_width_pdf, opacity
                )
                canvas.drawPath(path, fill=1 if do_fill else 0, stroke=1 if do_stroke else 0)
                return True

        points = obj.get("points") or []
        if vtype == "LINE" and len(points) >= 2:
            if from_source:
                p1 = ctx.source_point_to_pdf(float(points[0]["x"]), float(points[0]["y"]))
                p2 = ctx.source_point_to_pdf(float(points[-1]["x"]), float(points[-1]["y"]))
            else:
                p1 = (points[0]["x"] * ctx.scale_x, ctx.pdf_height - points[0]["y"] * ctx.scale_y)
                p2 = (points[-1]["x"] * ctx.scale_x, ctx.pdf_height - points[-1]["y"] * ctx.scale_y)
            canvas.setStrokeColor(parse_color(stroke or fill or "#333333", alpha=opacity))
            canvas.setLineWidth(max(0.5, stroke_width_pdf or 1.0))
            dash = (obj.get("meta") or {}).get("dash")
            if dash:
                try:
                    canvas.setDash(list(dash))
                except Exception:
                    canvas.setDash(1.5, 2.2)
            canvas.line(p1[0], p1[1], p2[0], p2[1])
            if dash:
                canvas.setDash([])
            return True

        if vtype == "LINE":
            canvas.setStrokeColor(parse_color(stroke or fill or "#333333", alpha=opacity))
            canvas.setLineWidth(max(0.5, stroke_width_pdf or 1.0))
            dash = (obj.get("meta") or {}).get("dash")
            if dash:
                try:
                    canvas.setDash(list(dash))
                except Exception:
                    canvas.setDash(1.5, 2.2)
            if pw >= ph:
                mid = py + ph / 2
                canvas.line(px, mid, px + pw, mid)
            else:
                mid = px + pw / 2
                canvas.line(mid, py, mid, py + ph)
            if dash:
                canvas.setDash([])
            return True

        if vtype in {"CIRCLE", "ELLIPSE"}:
            do_fill, do_stroke = _apply_fill_stroke(canvas, fill, stroke, stroke_width_pdf, opacity)
            canvas.ellipse(px, py, px + pw, py + ph, stroke=1 if do_stroke else 0, fill=1 if do_fill else 0)
            return True

        if vtype in {"POLYGON", "TRIANGLE"} and len(points) >= 3:
            path = canvas.beginPath()
            first = True
            for p in points:
                if from_source:
                    qx, qy = ctx.source_point_to_pdf(float(p["x"]), float(p["y"]))
                else:
                    qx, qy = p["x"] * ctx.scale_x, ctx.pdf_height - p["y"] * ctx.scale_y
                if first:
                    path.moveTo(qx, qy)
                    first = False
                else:
                    path.lineTo(qx, qy)
            path.close()
            do_fill, do_stroke = _apply_fill_stroke(canvas, fill, stroke, stroke_width_pdf, opacity)
            canvas.drawPath(path, fill=1 if do_fill else 0, stroke=1 if do_stroke else 0)
            return True

        if vtype == "ROUNDED_RECTANGLE":
            radius = float(obj.get("corner_radius") or 0.0)
            if from_source:
                radius_pdf = radius * ctx.scale_x * ctx.src_scale_x
            else:
                radius_pdf = radius * ctx.scale_x
            radius_pdf = max(0.0, min(radius_pdf, min(pw, ph) / 2))
            do_fill, do_stroke = _apply_fill_stroke(canvas, fill, stroke, stroke_width_pdf, opacity)
            canvas.roundRect(px, py, pw, ph, radius_pdf, stroke=1 if do_stroke else 0, fill=1 if do_fill else 0)
            return True

        # Default rectangle / panel / border / background / color region
        do_fill, do_stroke = _apply_fill_stroke(
            canvas,
            fill if vtype != "BORDER" else None,
            stroke or (fill if vtype == "BORDER" else None),
            stroke_width_pdf if vtype == "BORDER" else (stroke_width_pdf or 0),
            opacity,
        )
        if vtype == "BORDER":
            canvas.setStrokeColor(parse_color(stroke or fill or "#000000", alpha=opacity))
            canvas.setLineWidth(max(1.0, stroke_width_pdf or 2.0))
            canvas.rect(px, py, pw, ph, stroke=1, fill=0)
        else:
            canvas.rect(px, py, pw, ph, stroke=1 if do_stroke else 0, fill=1 if do_fill else 0)
        return True
    finally:
        canvas.restoreState()


def render_scene_shape(canvas: Canvas, obj: dict[str, Any], ctx: PageContext) -> bool:
    """Render a scene graph shape object (already in scene coordinates)."""
    return render_vector_object(canvas, obj, ctx, from_source=False)
