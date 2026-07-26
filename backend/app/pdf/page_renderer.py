"""
Page geometry — PDF points, scene↔source mapping, ReportLab helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reportlab.lib.pagesizes import A4, LETTER, landscape, portrait
from reportlab.lib.colors import Color, HexColor


@dataclass
class PageContext:
    """Coordinate transform from scene/source space → PDF points (origin bottom-left)."""

    pdf_width: float
    pdf_height: float
    scene_width: float
    scene_height: float
    scale_x: float  # scene → pdf
    scale_y: float
    # source image → scene page
    src_scale_x: float = 1.0
    src_scale_y: float = 1.0
    src_offset_x: float = 0.0
    src_offset_y: float = 0.0
    page_format: str = "A4"
    orientation: str = "portrait"
    margins: dict[str, float] | None = None

    def scene_to_pdf_xy(self, x: float, y: float) -> tuple[float, float]:
        """Top-left scene point → PDF point (still top-referenced y; convert with flip)."""
        px = x * self.scale_x
        py_top = y * self.scale_y
        return px, py_top

    def scene_rect_to_pdf(
        self, x: float, y: float, w: float, h: float
    ) -> tuple[float, float, float, float]:
        """Return ReportLab (x, y, w, h) with y at bottom-left of rectangle."""
        px = x * self.scale_x
        pw = w * self.scale_x
        ph = h * self.scale_y
        py_top = y * self.scale_y
        py = self.pdf_height - py_top - ph
        return px, py, pw, ph

    def source_rect_to_pdf(
        self, x: float, y: float, w: float, h: float
    ) -> tuple[float, float, float, float]:
        sx = self.src_offset_x + x * self.src_scale_x
        sy = self.src_offset_y + y * self.src_scale_y
        sw = w * self.src_scale_x
        sh = h * self.src_scale_y
        return self.scene_rect_to_pdf(sx, sy, sw, sh)

    def source_point_to_pdf(self, x: float, y: float) -> tuple[float, float]:
        sx = self.src_offset_x + x * self.src_scale_x
        sy = self.src_offset_y + y * self.src_scale_y
        px = sx * self.scale_x
        py = self.pdf_height - sy * self.scale_y
        return px, py


def detect_page_size(scene_page: dict[str, Any]) -> tuple[float, float, str, str]:
    fmt = str(scene_page.get("page_format") or "A4").upper()
    orient = str(scene_page.get("orientation") or "portrait").lower()
    base = LETTER if fmt == "LETTER" else A4
    if orient == "landscape":
        size = landscape(base)
    else:
        size = portrait(base)
    return float(size[0]), float(size[1]), fmt if fmt in {"A4", "LETTER"} else "A4", orient


def build_page_context(scene: dict[str, Any]) -> PageContext:
    page = scene.get("page") or {}
    pdf_w, pdf_h, fmt, orient = detect_page_size(page)
    scene_w = float(page.get("width") or pdf_w)
    scene_h = float(page.get("height") or pdf_h)
    return PageContext(
        pdf_width=pdf_w,
        pdf_height=pdf_h,
        scene_width=scene_w,
        scene_height=scene_h,
        scale_x=pdf_w / max(scene_w, 1.0),
        scale_y=pdf_h / max(scene_h, 1.0),
        src_scale_x=float(page.get("scale_x") or 1.0),
        src_scale_y=float(page.get("scale_y") or 1.0),
        src_offset_x=float(page.get("offset_x") or 0.0),
        src_offset_y=float(page.get("offset_y") or 0.0),
        page_format=fmt,
        orientation=orient,
        margins=page.get("margins") or {},
    )


def parse_color(value: str | None, default: str = "#000000", alpha: float = 1.0) -> Color:
    if not value:
        value = default
    try:
        if value.startswith("#") and len(value) >= 7:
            c = HexColor(value[:7])
            return Color(c.red, c.green, c.blue, alpha=max(0.0, min(1.0, alpha)))
    except Exception:
        pass
    return HexColor(default)
