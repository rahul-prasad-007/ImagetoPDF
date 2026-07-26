"""
PDF service — orchestration, validation, metadata via ReportLab + PyMuPDF.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.pdf.font_manager import missing_font_count
from app.pdf.models import (
    RenderCounts,
    RenderValidation,
    RenderValidationIssue,
)
from app.pdf.page_renderer import PageContext
from app.utils.validators import format_file_size

logger = logging.getLogger(__name__)


def validate_render_inputs(
    scene: dict[str, Any],
    vectors: dict[str, Any],
    ctx: PageContext,
) -> RenderValidation:
    issues: list[RenderValidationIssue] = []
    outside = 0
    invalid = 0
    missing_images = 0
    overlapping_text = 0

    texts: list[dict[str, Any]] = []
    for obj in scene.get("objects") or []:
        otype = str(obj.get("type") or "")
        x, y = float(obj.get("x") or 0), float(obj.get("y") or 0)
        w, h = float(obj.get("width") or 0), float(obj.get("height") or 0)
        if w < 0 or h < 0:
            invalid += 1
            issues.append(
                RenderValidationIssue(
                    code="INVALID_COORDINATES",
                    message=f"Object {obj.get('id')} has negative size",
                    object_id=obj.get("id"),
                )
            )
        if otype == "GROUP":
            continue
        # Scene space bounds
        if x < -5 or y < -5 or x + w > ctx.scene_width + 5 or y + h > ctx.scene_height + 5:
            outside += 1
            issues.append(
                RenderValidationIssue(
                    code="OUTSIDE_PAGE",
                    message=f"Object {obj.get('id')} extends outside page",
                    object_id=obj.get("id"),
                )
            )
        if otype in {"IMAGE", "LOGO", "ICON"}:
            path = obj.get("image_path")
            if path and not Path(path).is_file():
                missing_images += 1
                issues.append(
                    RenderValidationIssue(
                        code="MISSING_IMAGE",
                        message=f"Image missing for object {obj.get('id')}",
                        object_id=obj.get("id"),
                    )
                )
        if otype == "TEXT":
            texts.append(obj)

    # Overlapping text (heavy IoU)
    for i in range(len(texts)):
        a = texts[i]
        ax1, ay1 = float(a["x"]), float(a["y"])
        ax2, ay2 = ax1 + float(a["width"]), ay1 + float(a["height"])
        for j in range(i + 1, len(texts)):
            b = texts[j]
            bx1, by1 = float(b["x"]), float(b["y"])
            bx2, by2 = bx1 + float(b["width"]), by1 + float(b["height"])
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            iou = inter / area_a
            if iou >= 0.7:
                overlapping_text += 1
                issues.append(
                    RenderValidationIssue(
                        code="OVERLAPPING_TEXT",
                        message=f"Text objects {a.get('id')} and {b.get('id')} overlap",
                        object_id=a.get("id"),
                    )
                )

    missing_fonts = missing_font_count()
    if missing_fonts:
        issues.append(
            RenderValidationIssue(
                code="MISSING_FONTS",
                message="Preferred TTF fonts not found; falling back to Helvetica",
            )
        )

    ok = invalid == 0 and missing_images == 0
    return RenderValidation(
        ok=ok,
        issues=issues[:60],
        missing_fonts=missing_fonts,
        objects_outside_page=outside,
        invalid_coordinates=invalid,
        overlapping_text=overlapping_text,
        missing_images=missing_images,
    )


def apply_pdf_metadata(
    pdf_path: Path,
    *,
    title: str,
    author: str = "Image to Editable PDF",
    producer: str = "ImgToPDF Editable Renderer",
) -> None:
    """Set PDF info dictionary via PyMuPDF (editable document metadata)."""
    try:
        doc = fitz.open(pdf_path)
        doc.set_metadata(
            {
                "title": title,
                "author": author,
                "producer": producer,
                "creator": producer,
                "creationDate": datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ"),
                "modDate": datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ"),
            }
        )
        doc.saveIncr()
        doc.close()
    except Exception as exc:
        logger.warning("Could not write PDF metadata: %s", exc)


def count_drawn(text_n: int, image_n: int, vector_n: int) -> RenderCounts:
    return RenderCounts(
        total_objects=text_n + image_n + vector_n,
        text_count=text_n,
        image_count=image_n,
        vector_count=vector_n,
    )


def describe_size(path: Path) -> tuple[int, str]:
    size = path.stat().st_size if path.is_file() else 0
    return size, format_file_size(size)
