"""
Editable text rendering — real PDF text objects (never rasterized).
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

from reportlab.pdfgen.canvas import Canvas

from app.pdf.font_manager import normalize_font_family, resolve_aatext_font_path, resolve_font
from app.pdf.page_renderer import PageContext, parse_color

logger = logging.getLogger(__name__)


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in (text or ""))


def needs_complex_script(obj: dict[str, Any]) -> bool:
    content = (obj.get("content") or "").strip()
    if not content:
        return False
    meta = obj.get("meta") or {}
    render = meta.get("render") or {}
    text_spec = render.get("text") or {}
    family = str(text_spec.get("font_family") or obj.get("font_family") or "")
    fam = normalize_font_family(family)
    return fam == "devanagari" or _has_devanagari(content)


def _fit_font_size(text: str, font_name: str, font_size: float, max_width: float, canvas: Canvas) -> float:
    if max_width <= 1 or not text:
        return max(6.0, font_size)
    size = max(6.0, font_size)
    for _ in range(20):
        w = canvas.stringWidth(text, font_name, size)
        if w <= max_width or size <= 6.0:
            break
        size *= 0.92
    return size


def render_text_object(canvas: Canvas, obj: dict[str, Any], ctx: PageContext) -> bool:
    """
    Draw a scene TEXT object as editable PDF text.
    Returns True if drawn.
    Devanagari is deferred to PyMuPDF (ReportLab lacks Indic shaping).
    """
    content = (obj.get("content") or "").strip()
    if not content or not obj.get("visibility", True):
        return False

    if needs_complex_script(obj):
        return False

    meta = obj.get("meta") or {}
    render = meta.get("render") or {}
    text_spec = render.get("text") or {}

    bold = float(text_spec.get("bold") or 0.0)
    italic = float(text_spec.get("italic") or 0.0)
    family = str(text_spec.get("font_family") or obj.get("font_family") or "serif")
    font_name = resolve_font(bold=bold, italic=italic, family=family)

    x = float(obj.get("x") or 0)
    y = float(obj.get("y") or 0)
    w = float(obj.get("width") or 0)
    h = float(obj.get("height") or 0)
    rotation = float(obj.get("rotation") or text_spec.get("rotation") or 0)
    if abs(rotation) < 5.0:
        rotation = 0.0
    opacity = float(obj.get("opacity") or 1.0)
    alignment = str(obj.get("alignment") or text_spec.get("alignment") or "left").lower()
    font_size_scene = float(obj.get("font_size") or text_spec.get("font_size") or max(10.0, h * 0.75))
    if h > 1:
        font_size_scene = min(font_size_scene, h * 0.92)
    font_size = max(6.0, font_size_scene * ctx.scale_y)
    color = parse_color(obj.get("font_color") or text_spec.get("font_color"), "#111111", opacity)

    px, py, pw, ph = ctx.scene_rect_to_pdf(x, y, w, h)

    lines = content.split("\n")
    line_spacing = float(text_spec.get("line_height") or 1.2)
    char_spacing = float(text_spec.get("character_spacing") or 0.0) * ctx.scale_x

    canvas.saveState()
    try:
        origin_x = px
        origin_y = py + ph

        if abs(rotation) > 0.05:
            canvas.translate(origin_x, origin_y)
            canvas.rotate(-rotation)
            draw_x, draw_y_base = 0.0, 0.0
        else:
            draw_x, draw_y_base = origin_x, origin_y

        canvas.setFillColor(color)
        try:
            canvas.setFillAlpha(opacity)
            canvas.setStrokeAlpha(opacity)
        except Exception:
            pass

        cursor_y = draw_y_base - font_size * 0.85
        for line in lines:
            size = _fit_font_size(line, font_name, font_size, max(pw, 1.0), canvas)
            canvas.setFont(font_name, size)
            text_w = canvas.stringWidth(line, font_name, size)
            if alignment == "center":
                lx = draw_x + max(0.0, (pw - text_w) / 2.0)
            elif alignment == "right":
                lx = draw_x + max(0.0, pw - text_w)
            else:
                lx = draw_x

            if abs(char_spacing) > 0.05 and len(line) > 1:
                cx = lx
                for ch in line:
                    canvas.drawString(cx, cursor_y, ch)
                    cx += canvas.stringWidth(ch, font_name, size) + char_spacing
            else:
                canvas.drawString(lx, cursor_y, line)

            if float(text_spec.get("underline") or 0) >= 0.75:
                canvas.setStrokeColor(color)
                canvas.setLineWidth(max(0.5, size * 0.06))
                uy = cursor_y - size * 0.12
                canvas.line(lx, uy, lx + text_w, uy)

            cursor_y -= size * line_spacing
    finally:
        canvas.restoreState()

    return True


def render_complex_script_texts(
    pdf_path: Path,
    objects: list[dict[str, Any]],
    ctx: PageContext,
) -> int:
    """
    Insert Devanagari text via PyMuPDF HTML (OpenType shaping).
    Returns number of lines drawn.
    """
    import fitz

    pending = [o for o in objects if needs_complex_script(o) and o.get("visibility", True)]
    if not pending:
        return 0

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        drawn = 0
        page_h = float(page.rect.height)
        aatext = resolve_aatext_font_path()
        face_css = ""
        archive = None
        if aatext is not None:
            # Local TTF must be loaded via Archive + @font-face url(filename)
            archive = fitz.Archive(str(aatext.parent))
            face_css = (
                f"@font-face{{font-family:AAText;src:url('{aatext.name}');}}"
                f"@font-face{{font-family:AAText;src:url('{aatext.name}');font-weight:700;}}"
            )
        for obj in pending:
            content = (obj.get("content") or "").strip()
            if not content:
                continue
            meta = obj.get("meta") or {}
            render = meta.get("render") or {}
            text_spec = render.get("text") or {}
            alignment = str(obj.get("alignment") or text_spec.get("alignment") or "left").lower()
            align_css = {"center": "center", "right": "right", "justify": "justify"}.get(
                alignment, "left"
            )
            bold = float(text_spec.get("bold") or 0.0) >= 0.45
            weight = "700" if bold else "400"
            color = str(obj.get("font_color") or text_spec.get("font_color") or "#111111")
            x = float(obj.get("x") or 0)
            y = float(obj.get("y") or 0)
            w = float(obj.get("width") or 0)
            h = float(obj.get("height") or 0)
            font_size_scene = float(
                obj.get("font_size") or text_spec.get("font_size") or max(10.0, h * 0.75)
            )
            if h > 1:
                font_size_scene = min(font_size_scene, h * 0.92)
            font_size = max(6.0, font_size_scene * ctx.scale_y)

            px, py, pw, ph = ctx.scene_rect_to_pdf(x, y, w, h)
            x0 = px
            y0 = page_h - (py + ph)
            x1 = px + max(pw, 2.0)
            y1 = page_h - py
            rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 2)
            safe = html.escape(content)
            # text-align-last helps justify single OCR lines that fill their box width
            last_align = "justify" if align_css == "justify" else align_css
            css_html = (
                f"<div style=\"font-family:AAText,'Nirmala UI',Nirmala,Mangal,sans-serif;"
                f"font-size:{font_size:.2f}px;font-weight:{weight};color:{color};"
                f"text-align:{align_css};text-align-last:{last_align};"
                f"line-height:1.15;white-space:nowrap;overflow:hidden;\">{safe}</div>"
            )
            try:
                kwargs = {"css": face_css} if face_css else {}
                if archive is not None:
                    kwargs["archive"] = archive
                page.insert_htmlbox(rect, css_html, **kwargs)
                drawn += 1
            except Exception as exc:
                logger.warning("Devanagari htmlbox failed for %s: %s", obj.get("id"), exc)
                try:
                    fontfile = str(aatext) if aatext else r"C:\Windows\Fonts\Nirmala.ttc"
                    page.insert_font(fontfile=fontfile, fontname="aatext")
                    page.insert_text(
                        (x0, min(y1 - 2, y0 + font_size)),
                        content,
                        fontname="aatext",
                        fontsize=font_size,
                        color=_hex_to_rgb01(color),
                    )
                    drawn += 1
                except Exception as exc2:
                    logger.warning("Devanagari fallback failed: %s", exc2)
        doc.saveIncr()
        return drawn
    finally:
        doc.close()


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    c = (color or "#111111").lstrip("#")
    if len(c) < 6:
        return (0.07, 0.07, 0.07)
    try:
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return (r / 255.0, g / 255.0, b / 255.0)
    except Exception:
        return (0.07, 0.07, 0.07)
