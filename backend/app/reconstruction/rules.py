"""
Decision rules for reconstruction planning.

Maps layout / typography signals → reconstruction_type + confidence + reason.
No PDF/SVG export — planning only.
"""

from __future__ import annotations

from typing import Any, Optional

from app.reconstruction.models import ReconstructionType

# Layer bands (lower drawn first)
LAYER_BACKGROUND = 1
LAYER_PANELS = 2
LAYER_SHAPES = 3
LAYER_IMAGES = 5
LAYER_TEXT = 8

# Structural containers are not reconstructed as drawable content
_IGNORE_TYPES = {
    "PAGE",
    "HEADER",
    "MAIN_CONTENT",
    "FOOTER",
}

_TEXT_TYPES = {
    "TITLE",
    "SUBTITLE",
    "PARAGRAPH",
    "TEXT_BLOCK",
    "LIST",
}

_VECTOR_MAP = {
    "RECTANGLE": ReconstructionType.VECTOR_RECTANGLE,
    "ROUNDED_RECTANGLE": ReconstructionType.VECTOR_ROUNDED_RECTANGLE,
    "LINE": ReconstructionType.VECTOR_LINE,
    "CIRCLE": ReconstructionType.VECTOR_CIRCLE,
    "ELLIPSE": ReconstructionType.VECTOR_ELLIPSE,
    "POLYGON": ReconstructionType.VECTOR_POLYGON,
    "TABLE": ReconstructionType.VECTOR_RECTANGLE,  # table grid → rect scaffold for now
}

_IMAGE_MAP = {
    "PHOTO": ReconstructionType.PHOTO_IMAGE,
    "LOGO": ReconstructionType.LOGO_IMAGE,
    "ICON": ReconstructionType.ICON_IMAGE,
    "QR_CODE": ReconstructionType.ICON_IMAGE,
    "IMAGE": ReconstructionType.IMAGE,
}

_PATH_CANDIDATES = {
    "DECORATIVE_ELEMENT",
}

_VECTOR_CONFIDENCE_FLOOR = 80.0


def decide_reconstruction(
    obj: dict[str, Any],
    typography_by_ocr: Optional[dict[int, dict[str, Any]]] = None,
    text_boxes: Optional[list[tuple[float, float, float, float]]] = None,
) -> tuple[ReconstructionType, float, str, int]:
    """
    Returns (reconstruction_type, confidence 0-100, reason, layer).
    """
    otype = str(obj.get("type") or "")
    conf_in = float(obj.get("confidence") or 0.85)
    # Layout confidence may be 0-1
    if conf_in <= 1.0:
        conf_in *= 100.0

    if otype in _IGNORE_TYPES:
        return ReconstructionType.IGNORE, 100.0, "Structural container — not drawable", 0

    # Drop glyph-fragment shapes that land on OCR text — keep highlight accents
    if text_boxes and otype in {
        "RECTANGLE",
        "ROUNDED_RECTANGLE",
        "LINE",
        "CIRCLE",
        "ELLIPSE",
        "POLYGON",
        "DECORATIVE_ELEMENT",
    }:
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        meta = obj.get("meta") or {}
        color = meta.get("color_bgr")
        is_highlight = bool(meta.get("highlight"))
        if color and len(color) >= 3:
            b, g, r = int(color[0]), int(color[1]), int(color[2])
            mx, mn = max(r, g, b), min(r, g, b)
            sat = (mx - mn) / max(mx, 1)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            if sat > 0.25 and 70 < lum < 245 and g >= r + 20 and g >= b + 10:
                is_highlight = True
        if is_highlight and otype in {"RECTANGLE", "ROUNDED_RECTANGLE"}:
            return (
                ReconstructionType.VECTOR_RECTANGLE,
                min(99.0, max(conf_in, 93.0)),
                "Text highlight accent → vector rectangle",
                LAYER_PANELS,
            )
        if len(bbox) >= 4:
            box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            bw = max(1.0, box[2] - box[0])
            bh = max(1.0, box[3] - box[1])
            area = bw * bh
            covered = 0.0
            for tb in text_boxes:
                ix1 = max(box[0], tb[0])
                iy1 = max(box[1], tb[1])
                ix2 = min(box[2], tb[2])
                iy2 = min(box[3], tb[3])
                covered += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            coverage = covered / max(area, 1.0)
            if coverage >= 0.35:
                return (
                    ReconstructionType.IGNORE,
                    95.0,
                    "Shape overlaps OCR text — likely glyph/panel false positive",
                    0,
                )
            for tb in text_boxes:
                ix1 = max(box[0], tb[0])
                iy1 = max(box[1], tb[1])
                ix2 = min(box[2], tb[2])
                iy2 = min(box[3], tb[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter / area >= 0.4 and area < 25000:
                    return (
                        ReconstructionType.IGNORE,
                        95.0,
                        "Shape overlaps OCR text — likely glyph false positive",
                        0,
                    )

    # Full-page BACKGROUND → background image or soft ignore of duplicate
    if otype == "BACKGROUND":
        return (
            ReconstructionType.BACKGROUND_IMAGE,
            92.0,
            "Full-page background region",
            LAYER_BACKGROUND,
        )

    if otype == "BACKGROUND_SHAPE":
        if conf_in >= _VECTOR_CONFIDENCE_FLOOR:
            return (
                ReconstructionType.VECTOR_RECTANGLE,
                min(99.0, max(conf_in, 90.0)),
                "Background panel as vector rectangle",
                LAYER_PANELS,
            )
        return (
            ReconstructionType.BACKGROUND_IMAGE,
            max(70.0, conf_in),
            "Low-confidence panel → embedded background image",
            LAYER_PANELS,
        )

    # Text — always editable
    if otype in _TEXT_TYPES:
        ocr_ids = obj.get("ocr_block_ids") or []
        typo_conf = None
        if typography_by_ocr and ocr_ids:
            for oid in ocr_ids:
                style = typography_by_ocr.get(int(oid))
                if style:
                    typo_conf = float(style.get("confidence") or typo_conf or 90)
                    break
        score = typo_conf if typo_conf is not None else max(95.0, conf_in)
        score = min(99.5, max(95.0, score))
        return ReconstructionType.TEXT, score, "OCR text → editable text", LAYER_TEXT

    # Simple geometric shapes → vectors
    if otype in _VECTOR_MAP:
        rtype = _VECTOR_MAP[otype]
        if conf_in < _VECTOR_CONFIDENCE_FLOOR:
            return (
                ReconstructionType.IMAGE,
                conf_in,
                f"Low shape confidence ({conf_in:.0f}%) → embedded image",
                LAYER_IMAGES,
            )
        return rtype, min(99.5, max(conf_in, 92.0)), f"Geometric {otype} → vector", LAYER_SHAPES

    # Photos / logos / icons / QR — never embed regions that are mostly OCR text
    if otype in _IMAGE_MAP:
        if text_boxes:
            bbox = obj.get("bbox") or [0, 0, 0, 0]
            if len(bbox) >= 4:
                box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
                covered = 0.0
                for tb in text_boxes:
                    ix1 = max(box[0], tb[0])
                    iy1 = max(box[1], tb[1])
                    ix2 = min(box[2], tb[2])
                    iy2 = min(box[3], tb[3])
                    covered += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if covered / area > 0.2:
                    return (
                        ReconstructionType.IGNORE,
                        96.0,
                        f"{otype} overlaps text — skip raster overlay",
                        0,
                    )
        return (
            _IMAGE_MAP[otype],
            min(99.0, max(conf_in, 90.0)),
            f"{otype} → embedded image",
            LAYER_IMAGES,
        )

    # Decorative curves → attempt path; fall back to image if weak
    if otype in _PATH_CANDIDATES:
        if conf_in >= _VECTOR_CONFIDENCE_FLOOR:
            return (
                ReconstructionType.VECTOR_PATH,
                conf_in,
                "Decorative element → SVG path candidate",
                LAYER_SHAPES,
            )
        return (
            ReconstructionType.IMAGE,
            conf_in,
            "Decorative element low confidence → embedded image",
            LAYER_IMAGES,
        )

    # Unknown → ignore softly or image
    if conf_in >= 70:
        return (
            ReconstructionType.IMAGE,
            conf_in,
            f"Unknown type {otype} → embedded image fallback",
            LAYER_IMAGES,
        )
    return ReconstructionType.IGNORE, conf_in, f"Unknown/low-confidence {otype}", 0


def is_vector_type(rtype: ReconstructionType) -> bool:
    return rtype.value.startswith("VECTOR_")


def is_image_type(rtype: ReconstructionType) -> bool:
    return rtype in {
        ReconstructionType.IMAGE,
        ReconstructionType.LOGO_IMAGE,
        ReconstructionType.PHOTO_IMAGE,
        ReconstructionType.ICON_IMAGE,
        ReconstructionType.BACKGROUND_IMAGE,
    }


def is_background_type(rtype: ReconstructionType) -> bool:
    return rtype == ReconstructionType.BACKGROUND_IMAGE or (
        rtype == ReconstructionType.VECTOR_RECTANGLE
    )  # panels counted separately in planner
