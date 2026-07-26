"""
Image embedding — photos, logos, icons, QR only (cropped from processed image).
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image as PILImage
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from app.pdf.page_renderer import PageContext

logger = logging.getLogger(__name__)

_IMAGE_TYPES = {"IMAGE", "LOGO", "ICON", "PHOTO", "QR_CODE"}


def _crop_image(path: Path, crop: Optional[dict[str, Any]]) -> Optional[PILImage.Image]:
    try:
        img = PILImage.open(path).convert("RGBA")
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", path, exc)
        return None

    if not crop:
        return img

    x = int(max(0, float(crop.get("x") or 0)))
    y = int(max(0, float(crop.get("y") or 0)))
    w = int(max(1, float(crop.get("width") or img.width)))
    h = int(max(1, float(crop.get("height") or img.height)))
    x2 = min(img.width, x + w)
    y2 = min(img.height, y + h)
    if x2 <= x or y2 <= y:
        return img
    return img.crop((x, y, x2, y2))


def render_image_object(
    canvas: Canvas,
    obj: dict[str, Any],
    ctx: PageContext,
    default_image_path: Optional[Path] = None,
) -> bool:
    """Embed a scene IMAGE/LOGO/ICON as a PDF XObject (not a full-page flatten)."""
    otype = str(obj.get("type") or "").upper()
    if otype not in _IMAGE_TYPES:
        return False
    if not obj.get("visibility", True):
        return False

    image_path = obj.get("image_path")
    path = Path(image_path) if image_path else default_image_path
    if path is None or not path.is_file():
        logger.warning("Missing image for scene object %s", obj.get("id"))
        return False

    crop = obj.get("crop")
    meta = obj.get("meta") or {}
    render = meta.get("render") or {}
    image_spec = render.get("image") or {}
    if not crop and image_spec:
        crop = {
            "x": image_spec.get("crop_x"),
            "y": image_spec.get("crop_y"),
            "width": image_spec.get("crop_width"),
            "height": image_spec.get("crop_height"),
        }

    pil = _crop_image(path, crop)
    if pil is None:
        return False

    x = float(obj.get("x") or 0)
    y = float(obj.get("y") or 0)
    w = float(obj.get("width") or 0)
    h = float(obj.get("height") or 0)
    rotation = float(obj.get("rotation") or 0)
    opacity = float(obj.get("opacity") or 1.0)

    px, py, pw, ph = ctx.scene_rect_to_pdf(x, y, w, h)
    if pw < 1 or ph < 1:
        return False

    buf = BytesIO()
    # Preserve resolution — save cropped region as PNG
    pil_rgb = pil.convert("RGB") if pil.mode == "RGBA" else pil
    pil_rgb.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    reader = ImageReader(buf)

    canvas.saveState()
    try:
        try:
            canvas.setFillAlpha(opacity)
            canvas.setStrokeAlpha(opacity)
        except Exception:
            pass
        if abs(rotation) > 0.05:
            # Rotate around center of box
            cx, cy = px + pw / 2, py + ph / 2
            canvas.translate(cx, cy)
            canvas.rotate(-rotation)
            canvas.drawImage(
                reader,
                -pw / 2,
                -ph / 2,
                width=pw,
                height=ph,
                mask="auto",
                preserveAspectRatio=True,
                anchor="c",
            )
        else:
            canvas.drawImage(
                reader,
                px,
                py,
                width=pw,
                height=ph,
                mask="auto",
                preserveAspectRatio=False,
                anchor="c",
            )
    finally:
        canvas.restoreState()
    return True
